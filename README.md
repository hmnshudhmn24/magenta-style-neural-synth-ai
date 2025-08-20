# 🎛️ Magenta-Style Neural Synthesizer (NSynth-inspired)

Blend instrument timbres using a tiny **neural autoencoder** trained on mel-spectrograms.  
Morph between two sounds by **latent interpolation**, render melodies via **MIDI**, and play with an **interactive web UI**.

> ⚠️ This is a **minimal educational** take on NSynth-style ideas (not the original architecture). It’s compact, hackable, and fun.



## ✨ Features

- 🧠 **Conv Autoencoder** on log-mel spectrograms (learns timbre manifold)
- 🎚️ **Latent morphing**: interpolate between two instruments (A ↔ B)
- 🎹 **MIDI rendering**: pick a timbre, render a MIDI file (pitch-shift + ADSR)
- 🌐 **Gradio web app** for interactive blending
- 🧩 Plain-Python DSP via `librosa` (mel/inversion via Griffin-Lim)



## 🧰 Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```



## 📂 Data

Place your WAVs anywhere (mono or stereo; will be loaded as mono, 16 kHz).  
Good sources: single-note multisamples, short instrument phrases, NSynth subset, personal recordings.

> Tip: Keep files normalized; avoid long silences. More variety improves the learned timbre space.



## 🚆 Train the model

```bash
# Example: train on all WAVs in data/
python -m src.train --data "data/**/*.wav" --out checkpoints --epochs 25 --bs 16 --crop 512 --latent 128
```

- Saves `checkpoints/ae_ep*.pt` and `checkpoints/ae_best.pt`  
- Loss: L1 on normalized log-mels (robust and simple)



## 🧪 Timbre blending (A → B)

```bash
python -m src.blend --ckpt checkpoints/ae_best.pt   --a data/piano_C4.wav   --b data/flute_C4.wav   --alpha 0.35   --out out/piano_flute_blend.wav
```

- `alpha=0.0` → 100% A; `alpha=1.0` → 100% B  
- Output is reconstructed audio via mel inversion (Griffin-Lim)



## 🎼 Render a MIDI with a learned timbre

```bash
python -m src.midi_synth --ckpt checkpoints/ae_best.pt   --timbre data/clarinet_C4.wav   --midi examples/melody.mid   --out out/clarinet_melody.wav
```

How it works (simple but effective):
1. Encode reference WAV → latent → **decode** a base “note” sample  
2. For each MIDI note: **pitch-shift** base note (librosa), apply **ADSR**, and **overlap-add**  
3. Soft limiter → write WAV

> This is a naive renderer (not a neural decoder conditioned on pitch), but it’s surprisingly musical with short, clean sources.



## 🌐 Web UI

```bash
python app.py
```

- Upload `ae_best.pt`  
- Upload two WAVs (A & B)  
- Slide **α** to morph, listen, download



## 🧠 Architecture & DSP details

- **Input**: mono 16 kHz → `N_MELS=80`, `n_fft=1024`, `hop=256`, `fmin=40`, `fmax=7600`  
- **Normalization**: log-mel in dB, scaled from `[-80,0] dB` → `[-1,1]`  
- **Model**: 2D conv encoder/decoder with strided downsamples / deconvs (see `src/model.py`)  
- **Latent**: default `128` dims  
- **Reconstruction**: mel → audio via `librosa.feature.inverse.mel_to_audio` (Griffin-Lim, 32 iters)



## 🛠️ Tips & Tricks

- Train on **single notes** across pitches for cleaner timbre learning  
- Mix families (strings, winds, mallets) for cooler morphs  
- Increase `--latent` to preserve more detail (then retrain)  
- Try longer crops (`--crop 768` or `1024`) for sustained instruments  
- For Raspberry Pi:
  - Use smaller models (e.g., latent 64) and fewer channels  
  - Use the **web UI** from a laptop; Pi just hosts audio I/O or buttons



## 📈 Roadmap

- [ ] Variational AE (KL + sampling for smoother spaces)  
- [ ] Pitch-conditioned decoder (avoid naive pitch-shift)  
- [ ] Real-time audio I/O (sounddevice)  
- [ ] MIDI in → live synth engine  
- [ ] Lightweight on-device UI (Raspberry Pi + rotary encoder)


- [ ] Blend two timbres  
- [ ] Render a MIDI  
- [ ] Launch the UI and have fun 🎧
