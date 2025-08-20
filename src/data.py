import os, glob, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .audio import load_mono, wav_to_logmel, SR, HOP, N_MELS

class WavMelDataset(Dataset):
    def __init__(self, data_glob: str, crop_frames: int = 512):
        self.files = sorted(sum([glob.glob(p) for p in data_glob.split(",")], []))
        if not self.files:
            raise ValueError("No audio files found. Provide a glob like 'data/**/*.wav'")
        self.crop_frames = crop_frames

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        y = load_mono(self.files[idx])
        m = wav_to_logmel(y)  # [n_mels, T]
        if m.shape[1] < self.crop_frames:
            # loop-pad in time
            reps = int(np.ceil(self.crop_frames / m.shape[1]))
            m = np.tile(m, reps)[:, :self.crop_frames]
        start = random.randint(0, m.shape[1] - self.crop_frames)
        crop = m[:, start:start+self.crop_frames]  # [n_mels, F]
        x = torch.from_numpy(crop).unsqueeze(0)    # [1, n_mels, F]
        return x

def make_loader(glob_str: str, crop_frames=512, bs=16, num_workers=2):
    ds = WavMelDataset(glob_str, crop_frames)
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
