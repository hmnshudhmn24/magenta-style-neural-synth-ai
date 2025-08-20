import os, torch, numpy as np
from .model import TimbreAE
from .audio import load_mono, wav_to_logmel, logmel_to_wav, save_wav, SR, HOP
from .utils import device

def center_crop_frames(logm: np.ndarray, frames: int) -> np.ndarray:
    T = logm.shape[1]
    if T <= frames:
        reps = int(np.ceil(frames/T))
        return np.tile(logm, reps)[:, :frames]
    start = (T - frames)//2
    return logm[:, start:start+frames]

def load_model(ckpt_path: str, latent_dim=128) -> TimbreAE:
    dd = device()
    model = TimbreAE(latent_dim=latent_dim).to(dd)
    sd = torch.load(ckpt_path, map_location=dd)
    model.load_state_dict(sd["model"])
    model.eval()
    return model

def encode(model: TimbreAE, logm: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(logm).unsqueeze(0).unsqueeze(0).to(device())  # [1,1,M,T]
    with torch.no_grad():
        z = model.enc(x)
    return z.squeeze(0)

def decode(model: TimbreAE, z: torch.Tensor, out_frames: int) -> np.ndarray:
    with torch.no_grad():
        x_hat = model.dec(z.unsqueeze(0)).squeeze(0).squeeze(0).cpu().numpy()
    # ensure length
    if x_hat.shape[1] > out_frames:
        x_hat = x_hat[:, :out_frames]
    elif x_hat.shape[1] < out_frames:
        reps = int(np.ceil(out_frames/x_hat.shape[1]))
        x_hat = np.tile(x_hat, reps)[:, :out_frames]
    return x_hat

def blend(ckpt, wav_a, wav_b, alpha=0.5, out_wav="blend.wav", crop_frames=512):
    model = load_model(ckpt)
    ya = load_mono(wav_a); yb = load_mono(wav_b)
    la = center_crop_frames(wav_to_logmel(ya), crop_frames)
    lb = center_crop_frames(wav_to_logmel(yb), crop_frames)
    za = encode(model, la); zb = encode(model, lb)
    z = (1-alpha)*za + alpha*zb
    logm = decode(model, z, crop_frames)
    y = logmel_to_wav(logm, length_samples=crop_frames*HOP)
    save_wav(out_wav, y)
    return out_wav

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--a", required=True, help="Instrument A wav")
    ap.add_argument("--b", required=True, help="Instrument B wav")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default="blend.wav")
    ap.add_argument("--crop", type=int, default=512)
    args = ap.parse_args()
    path = blend(args.ckpt, args.a, args.b, args.alpha, args.out, args.crop)
    print("Saved:", path)
