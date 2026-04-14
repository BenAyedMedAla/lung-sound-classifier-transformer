import os
import numpy as np
import pandas as pd
import librosa
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

def process_data():
    print("🚀 Begins")
    
    
    split_df = pd.read_csv(SPLIT_FILE, sep='\t', names=['filename', 'set_type'])
    
    X_train, y_train, device_train = [], [], []
    X_test, y_test, device_test = [], [], []
    
    # İstatistikler
    stats = {'Normal': 0, 'Crackle': 0, 'Wheeze': 0, 'Both': 0}
    
    for index, row in tqdm(split_df.iterrows(), total=split_df.shape[0]):
        fname = row['filename']
        set_type = row['set_type'] 
        
        wav_path = os.path.join(DATA_DIR, fname + '.wav')
        txt_path = os.path.join(DATA_DIR, fname + '.txt')
        
        if not os.path.exists(wav_path) or not os.path.exists(txt_path):
            continue
            
       
        audio, _ = librosa.load(wav_path, sr=TARGET_SR)
        
       
        dev_id = get_device_id(fname)
        
        
        anns = pd.read_csv(txt_path, sep='\t', names=['start', 'end', 'crackle', 'wheeze'])
        
        for _, ann in anns.iterrows():
            start = int(ann['start'] * TARGET_SR)
            end = int(ann['end'] * TARGET_SR)
            
           
            chunk = audio[start:end]
            
           
            if len(chunk) < 100: 
                continue
                
            # --- CYCLIC PADDING ---
            processed_wav = cyclic_padding(chunk, TARGET_SAMPLES)
            
            #(0:Normal, 1:Crackle, 2:Wheeze, 3:Both)
            c = ann['crackle']
            w = ann['wheeze']
            
            if c == 0 and w == 0:
                label = 0 # Normal
                stats['Normal'] += 1
            elif c == 1 and w == 0:
                label = 1 # Crackle
                stats['Crackle'] += 1
            elif c == 0 and w == 1:
                label = 2 # Wheeze
                stats['Wheeze'] += 1
            else:
                label = 3 # Both
                stats['Both'] += 1
            
            
            if set_type == 'train':
                X_train.append(processed_wav)
                y_train.append(label)
                device_train.append(dev_id)
            else:
                X_test.append(processed_wav)
                y_test.append(label)
                device_test.append(dev_id)

   
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    device_train = np.array(device_train, dtype=np.int64)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    device_test = np.array(device_test, dtype=np.int64)
    
    
    print(f"Train : {X_train.shape}, Test : {X_test.shape}")
    
    # Kaydet
    np.savez(OUTPUT_FILENAME, 
             X_train=X_train, y_train=y_train, device_train=device_train,
             X_test=X_test, y_test=y_test, device_test=device_test)
    
    print(f"✅ Saved: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    process_data()