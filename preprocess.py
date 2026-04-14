import os
import gc
import shutil
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tempfile
from tqdm import tqdm


DATA_DIR = "./data/ICBHI_final_database"  
SPLIT_FILE = "./data/ICBHI_Challenge_train_test.txt"
OUTPUT_FILENAME = "icbhi_ast_16k_8s_metadata.npz"

TARGET_SR = 16000 
TARGET_DURATION = 8 
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION 


DEVICE_MAP = {
    'AKGC417L': 0,
    'LittC2SE': 1,
    'Litt3200': 2,
    'Meditron': 3
}

def get_device_id(filename):
    
    parts = filename.split('_')
    dev_name = parts[-1] 
    if dev_name in DEVICE_MAP:
        return DEVICE_MAP[dev_name]
    return -1 

def cyclic_padding(wav, target_len):
  
    curr_len = len(wav)
    if curr_len >= target_len:
        return wav[:target_len] 
    
    
    repeat_count = (target_len // curr_len) + 1
    padded = np.tile(wav, repeat_count)
    return padded[:target_len]


def load_audio_safe(wav_path, target_sr):
    """Load audio without triggering heavy soxr allocations.

    Strategy:
    1) Read raw waveform with soundfile (no resampling).
    2) Convert to mono.
    3) Resample with librosa `kaiser_fast` (resampy backend, not soxr).
    4) If memory is still tight, use linear interpolation fallback.
    """
    audio, sr = sf.read(wav_path, dtype='float32', always_2d=False)

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if sr != target_sr:
        try:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=target_sr,
                res_type='kaiser_fast'
            )
        except MemoryError:
            new_len = int(len(audio) * target_sr / sr)
            if new_len <= 0:
                return np.zeros(0, dtype=np.float32)

            old_x = np.arange(len(audio), dtype=np.float64)
            new_x = np.linspace(0, max(len(audio) - 1, 0), num=new_len, dtype=np.float64)
            audio = np.interp(new_x, old_x, audio).astype(np.float32)

    audio = np.asarray(audio, dtype=np.float32)
    return audio


def count_segments(split_df):
    """First pass: count train/test segments without loading full waveforms."""
    train_count = 0
    test_count = 0
    skipped_files = 0

    for _, row in tqdm(split_df.iterrows(), total=split_df.shape[0], desc="Pass 1/2 (count)"):
        fname = row['filename']
        set_type = row['set_type']

        wav_path = os.path.join(DATA_DIR, fname + '.wav')
        txt_path = os.path.join(DATA_DIR, fname + '.txt')

        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            continue

        try:
            info = sf.info(wav_path)
            duration_sec = float(info.frames) / float(info.samplerate)
            anns = pd.read_csv(txt_path, sep='\t', names=['start', 'end', 'crackle', 'wheeze'])
        except Exception as exc:
            skipped_files += 1
            print(f"[WARN] Counting skip {fname}.wav: {type(exc).__name__} - {exc}")
            continue

        for _, ann in anns.iterrows():
            start_sec = max(0.0, float(ann['start']))
            end_sec = min(duration_sec, float(ann['end']))

            if end_sec <= start_sec:
                continue

            seg_len = int((end_sec - start_sec) * TARGET_SR)
            if seg_len < 100:
                continue

            if set_type == 'train':
                train_count += 1
            else:
                test_count += 1

    return train_count, test_count, skipped_files


def estimate_memmap_bytes(train_count, test_count):
    bytes_per_sample = np.dtype(np.float16).itemsize * TARGET_SAMPLES
    return (train_count + test_count) * bytes_per_sample


def pick_work_parent(required_bytes):
    """Choose a writable directory with enough free space for temporary memmaps."""
    output_parent = os.path.dirname(os.path.abspath(OUTPUT_FILENAME)) or os.getcwd()
    preferred = os.environ.get("ICBHI_PREPROCESS_TMP", "").strip()

    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([tempfile.gettempdir(), output_parent])

    # Keep order while removing duplicates
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_abs = os.path.abspath(c)
        if c_abs not in seen:
            seen.add(c_abs)
            unique_candidates.append(c_abs)

    safety_factor = 1.15
    needed = int(required_bytes * safety_factor)

    for cand in unique_candidates:
        try:
            os.makedirs(cand, exist_ok=True)
            free_bytes = shutil.disk_usage(cand).free
            if free_bytes >= needed:
                print(f"Using temp parent: {cand} (free: {free_bytes / (1024**3):.2f} GiB)")
                return cand
        except Exception:
            continue

    raise OSError(
        f"No space left for temp memmaps. Need about {needed / (1024**3):.2f} GiB free. "
        f"Set ICBHI_PREPROCESS_TMP to a drive/folder with more free space."
    )

def process_data():
    print("🚀 Begins")
    
    
    split_df = pd.read_csv(SPLIT_FILE, sep='\t', names=['filename', 'set_type'])
    
    train_count, test_count, skipped_counting = count_segments(split_df)
    print(f"Counted segments -> train: {train_count}, test: {test_count}")

    required_bytes = estimate_memmap_bytes(train_count, test_count)
    print(f"Estimated temporary memmap size: {required_bytes / (1024**3):.2f} GiB")

    stats = {'Normal': 0, 'Crackle': 0, 'Wheeze': 0, 'Both': 0}
    skipped_files = 0

    work_parent = pick_work_parent(required_bytes)
    tmpdir = tempfile.mkdtemp(prefix="icbhi_preprocess_", dir=work_parent)
    x_train_path = os.path.join(tmpdir, "X_train.dat")
    x_test_path = os.path.join(tmpdir, "X_test.dat")
    X_train_mm = None
    X_test_mm = None

    try:
        x_train_path = os.path.join(tmpdir, "X_train.dat")
        x_test_path = os.path.join(tmpdir, "X_test.dat")

        X_train_mm = np.memmap(x_train_path, dtype=np.float16, mode='w+', shape=(train_count, TARGET_SAMPLES))
        X_test_mm = np.memmap(x_test_path, dtype=np.float16, mode='w+', shape=(test_count, TARGET_SAMPLES))

        y_train = np.empty(train_count, dtype=np.int64)
        device_train = np.empty(train_count, dtype=np.int64)
        y_test = np.empty(test_count, dtype=np.int64)
        device_test = np.empty(test_count, dtype=np.int64)

        train_i = 0
        test_i = 0

        for _, row in tqdm(split_df.iterrows(), total=split_df.shape[0], desc="Pass 2/2 (build)"):
            fname = row['filename']
            set_type = row['set_type']

            wav_path = os.path.join(DATA_DIR, fname + '.wav')
            txt_path = os.path.join(DATA_DIR, fname + '.txt')

            if not os.path.exists(wav_path) or not os.path.exists(txt_path):
                continue

            try:
                audio = load_audio_safe(wav_path, TARGET_SR)
                anns = pd.read_csv(txt_path, sep='\t', names=['start', 'end', 'crackle', 'wheeze'])
            except (RuntimeError, MemoryError, ValueError, OSError) as exc:
                skipped_files += 1
                print(f"[WARN] Skipping {fname}.wav: {type(exc).__name__} - {exc}")
                continue

            dev_id = get_device_id(fname)

            for _, ann in anns.iterrows():
                start = max(0, int(float(ann['start']) * TARGET_SR))
                end = min(len(audio), int(float(ann['end']) * TARGET_SR))

                chunk = audio[start:end]
                if len(chunk) < 100:
                    continue

                processed_wav = cyclic_padding(chunk, TARGET_SAMPLES).astype(np.float16, copy=False)

                c = int(ann['crackle'])
                w = int(ann['wheeze'])

                if c == 0 and w == 0:
                    label = 0
                    stats['Normal'] += 1
                elif c == 1 and w == 0:
                    label = 1
                    stats['Crackle'] += 1
                elif c == 0 and w == 1:
                    label = 2
                    stats['Wheeze'] += 1
                else:
                    label = 3
                    stats['Both'] += 1

                if set_type == 'train':
                    if train_i >= train_count:
                        continue
                    X_train_mm[train_i] = processed_wav
                    y_train[train_i] = label
                    device_train[train_i] = dev_id
                    train_i += 1
                else:
                    if test_i >= test_count:
                        continue
                    X_test_mm[test_i] = processed_wav
                    y_test[test_i] = label
                    device_test[test_i] = dev_id
                    test_i += 1

        X_train_mm.flush()
        X_test_mm.flush()

        print(f"Final filled -> train: {train_i}, test: {test_i}")
        print(f"Class stats: {stats}")
        print(f"Skipped files due to load/resample errors (pass2): {skipped_files}")
        print(f"Skipped files during counting (pass1): {skipped_counting}")

        # Kaydet
        np.savez(
            OUTPUT_FILENAME,
            X_train=X_train_mm[:train_i], y_train=y_train[:train_i], device_train=device_train[:train_i],
            X_test=X_test_mm[:test_i], y_test=y_test[:test_i], device_test=device_test[:test_i]
        )
    finally:
        if X_train_mm is not None:
            del X_train_mm
        if X_test_mm is not None:
            del X_test_mm

        gc.collect()

        for p in (x_train_path, x_test_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except PermissionError:
                print(f"[WARN] Could not delete temp file yet: {p}")
            except OSError as exc:
                print(f"[WARN] Temp cleanup error for {p}: {exc}")

        try:
            if os.path.isdir(tmpdir):
                os.rmdir(tmpdir)
        except OSError:
            # Non-fatal on Windows if still briefly locked.
            pass
    
    print(f"✅ Saved: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    process_data()