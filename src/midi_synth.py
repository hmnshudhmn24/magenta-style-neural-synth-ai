import numpy as np, pretty_midi, librosa, torch
from typing import Tuple
from .model import TimbreAE
from .audio import load_mono, wav_to_logmel, logmel_to_wav, save_wav, SR, HOP
from .blend import load_model, encode
from .utils import device

def decode_timbre(model: TimbreAE, z, frames: int = 512):
    with torch.no_grad():
        logm = model.dec(z.unsqueeze(0)).squeeze(0).squeeze(0).cpu().numpy()
    # ensure frames
    if logm.shape[1] < frames:
        reps = int(np.ceil(frames/logm.shape[1]))
        logm = np.tile(logm, reps)[:, :frames]
    elif logm.shape[1] > frames:
        logm = logm[:, :frames]
    wav = logmel_to_wav(logm, length_samples=frames*HOP)
    return wav

def render_midi(ckpt: str, timbre_wav: str, midi_path: str, out_wav: str = "midi_render.wav"):
    # 1) Load timbre latent from reference wav
    model = load_model(ckpt)
    y = load_mono(timbre_wav)
    z = encode(model, wav_to_logmel(y))
    base = decode_timbre(model, z, frames=512)  # base “note” sample ~0.5–1.0s

    # 2) Parse MIDI and synthesize by pitch-shifting & ADSR
    pm = pretty_midi.PrettyMIDI(midi_path)
    sr = SR
    out_len = int(pm.get_end_time() * sr) + sr
    out = np.zeros(out_len, dtype=np.float32)

    def adsr(n: int, a=0.01, d=0.05, s=0.7, r=0.1):
        A = int(a*sr); D = int(d*sr); R = int(r*sr)
        S = max(1, n - A - D - R)
        env = np.concatenate([
            np.linspace(0, 1, A, False),
            np.linspace(1, s, D, False),
            np.full(S, s),
            np.linspace(s, 0, R, True)
        ])
        if len(env) < n:
            env = np.pad(env, (0, n-len(env)))
        return env[:n].astype(np.float32)

    for inst in pm.instruments:
        for note in inst.notes:
            start = int(note.start * sr)
            end = int(note.end * sr)
            length = max(end - start, int(0.1*sr))
            # semitone shift from A4=69
            n_steps = note.pitch - 69
            seg = librosa.effects.pitch_shift(base, sr=sr, n_steps=n_steps)
            if len(seg) < length:
                reps = int(np.ceil(length/len(seg)))
                seg = np.tile(seg, reps)[:length]
            else:
                seg = seg[:length]
            env = adsr(length, a=0.01, d=0.08, s=min(1.0, note.velocity/100), r=0.12)
            sig = (seg * env).astype(np.float32)
            out[start:start+length] += sig

    # limiter
    peak = np.max(np.abs(out) + 1e-6)
    out = (out / peak * 0.9).astype(np.float32)
    save_wav(out_wav, out)
    return out_wav

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--timbre", required=True, help="Reference WAV for timbre")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", default="midi_render.wav")
    args = ap.parse_args()
    path = render_midi(args.ckpt, args.timbre, args.midi, args.out)
    print("Saved:", path)
