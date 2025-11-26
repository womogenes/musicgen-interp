import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("small", device="cuda")
model.set_generation_params(duration=8)

descriptions = ["happy rock", "energetic EDM"]

# hooks
activations = {}

def make_hook(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out_t = out[0]
        else:
            out_t = out
        activations.setdefault(name, []).append(out_t.detach().cpu())
    return hook

for i, block in enumerate(model.lm.transformer.layers):
    block.register_forward_hook(make_hook(f"lm_block_{i}"))

wav = model.generate(descriptions)

for idx, one_wav in enumerate(wav):
    audio_write(f"{idx}", one_wav.cpu(), model.sample_rate, strategy="loudness")

torch.save(activations, "musicgen_activations.pt")

print({k: [t.shape for t in v] for k, v in activations.items()})
