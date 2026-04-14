import torch
from torch.utils.data import Dataset
import numpy as np

class ASTDataset(Dataset):
    def __init__(self, X, y, device_ids, processor, train=True):
        self.X = X
        self.y = y
        self.device_ids = device_ids
        self.processor = processor
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        wav = self.X[idx]

        # AST Augmentation (Sadece Eğitim Sırasında)
        if self.train:
            # Hafif Gain (Ses şiddeti) değişimi
            if np.random.random() < 0.5:
                wav = wav * np.random.uniform(0.9, 1.1)
            # Hafif Gaussian Noise
            if np.random.random() < 0.5:
                # Noise seviyesi: 0.0001
                wav = wav + np.random.normal(0, 0.0001, wav.shape) 

        # Processor: Audio -> Spectrogram -> Patches
        # Sampling rate 16kHz sabittir
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)

        # Cihaz ID'sini döndürüyoruz ama eğitimde kullanmıyoruz (sadece analiz için)
        return input_values, torch.tensor(self.y[idx], dtype=torch.long), self.device_ids[idx]