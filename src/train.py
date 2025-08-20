import os, torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
from .model import TimbreAE
from .data import make_loader
from .utils import device, ensure_dir, seed_everything

def train(
    data_glob: str,
    out_dir: str = "checkpoints",
    epochs: int = 20,
    bs: int = 16,
    lr: float = 2e-4,
    crop_frames: int = 512,
    latent_dim: int = 128,
    seed: int = 42
):
    seed_everything(seed)
    ensure_dir(out_dir)
    dd = device()
    model = TimbreAE(latent_dim=latent_dim).to(dd)
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    loader = make_loader(data_glob, crop_frames=crop_frames, bs=bs)

    best = 1e9
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for x in tqdm(loader, desc=f"Epoch {ep}/{epochs}"):
            x = x.to(dd)  # [B,1,M,T]
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        avg = running/len(loader)
        print(f"Epoch {ep} | MAE: {avg:.4f}")
        ckpt = os.path.join(out_dir, f"ae_ep{ep}.pt")
        torch.save({"model": model.state_dict()}, ckpt)
        if avg < best:
            best = avg
            torch.save({"model": model.state_dict()}, os.path.join(out_dir, "ae_best.pt"))
    print("Done. Best MAE:", best)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Glob(s) to WAV/FLAC/MP3 files, e.g., 'data/**/*.wav'")
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--latent", type=int, default=128)
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.bs, args.lr, args.crop, args.latent)
