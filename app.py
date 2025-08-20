import gradio as gr, os
from src.blend import blend

def do_blend(ckpt, a, b, alpha):
    if ckpt is None or not a or not b:
        return None, "Please upload model checkpoint and two WAVs."
    ckpt_path = ckpt.name if hasattr(ckpt, "name") else ckpt
    out = "blend_ui.wav"
    path = blend(ckpt_path, a, b, alpha, out)
    return out, f"Saved {path}"

with gr.Blocks(title="Magenta-Style Neural Synthesizer") as demo:
    gr.Markdown("# 🎛️ Magenta-Style Neural Synth — NSynth-like Timbre Morphing\nUpload two WAVs and a trained checkpoint, then morph the timbre with the slider.")
    ckpt = gr.File(label="Checkpoint (ae_best.pt)")
    a = gr.Audio(label="Instrument A (WAV)", type="filepath")
    b = gr.Audio(label="Instrument B (WAV)", type="filepath")
    alpha = gr.Slider(0, 1, value=0.5, step=0.05, label="Blend α (0=A, 1=B)")
    btn = gr.Button("Blend")
    out_audio = gr.Audio(label="Blended Output", type="filepath")
    msg = gr.Markdown()
    btn.click(do_blend, inputs=[ckpt, a, b, alpha], outputs=[out_audio, msg])

if __name__ == "__main__":
    demo.launch()
