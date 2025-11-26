import os
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 1. Load processor + model
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small"
).to(device)
model.eval()

# 2. Prompts
descriptions = ["happy rock", "energetic EDM"]
inputs = processor(
    text=descriptions,
    padding=True,
    return_tensors="pt",
).to(device)

# 3. Hook setup (Audiocraft-style, but for Transformers)
activations = {}

def make_hook(name):
    def hook(module, inp, out):
        # For HF decoder layers → out is usually tuple(hidden_states, ...)
        if isinstance(out, tuple):
            out_t = out[0]
        else:
            out_t = out
        activations.setdefault(name, []).append(out_t.detach().cpu())
    return hook

handles = []

# Automatically find all decoder blocks by class name
for name, module in model.named_modules():
    if module.__class__.__name__ == "MusicgenDecoderLayer":
        print("Hooking:", name)
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

# 4. Generate audio (hooks fire during generation)
MAX_NEW_TOKENS = 256

with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=MAX_NEW_TOKENS,
    )

# Remove hooks
for h in handles:
    h.remove()

# 5. Save audio to WAV using *your* exact pattern
os.makedirs("outputs", exist_ok=True)

sampling_rate = model.config.audio_encoder.sampling_rate  # typically 32000 Hz

for i in range(audio_values.shape[0]):
    clip = audio_values[i, 0].cpu().numpy()  # mono
    out_path = f"outputs/{i}.wav"

    # EXACT code you provided:
    wavfile.write(out_path, rate=sampling_rate, data=clip)

    print(f"Saved {out_path}")

# 6. Save activations and print shapes
torch.save(activations, "musicgen_activations.pt")

shapes = {k: [t.shape for t in v] for k, v in activations.items()}
print(shapes)
