from typing import Tuple
import numpy as np, librosa, soundfile as sf

SR = 16000
N_FFT = 1024
HOP = 256
N_MELS = 80
FMIN = 40
FMAX = 7600

def load_mono(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def save_wav(path: str, y: np.ndarray, sr: int = SR):
    sf.write(path, y, sr)

def wav_to_logmel(y: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0
    )
    logm = librosa.power_to_db(m, ref=np.max)
    # normalize to [-1,1]
    logm = (logm - (-80.0)) / (0.0 - (-80.0))  # [-80,0] -> [0,1]
    logm = logm * 2.0 - 1.0                     # [0,1] -> [-1,1]
    return logm.astype(np.float32)

def logmel_to_wav(logm: np.ndarray, length_samples: int = None) -> np.ndarray:
    # denormalize
    m01 = (logm + 1.0) * 0.5            # [-1,1] -> [0,1]
    db = m01 * (0.0 - (-80.0)) + (-80.0)
    power = librosa.db_to_power(db)
    y = librosa.feature.inverse.mel_to_audio(
        power, sr=SR, n_fft=N_FFT, hop_length=HOP, fmin=FMIN, fmax=FMAX,
        n_iter=32
    )
    if length_samples is not None and len(y) > length_samples:
        y = y[:length_samples]
    return y

def pad_or_crop(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) == target_len: return y
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        return y[start:start+target_len]
    out = np.zeros(target_len, dtype=y.dtype)
    start = (target_len - len(y)) // 2
    out[start:start+len(y)] = y
    return out
