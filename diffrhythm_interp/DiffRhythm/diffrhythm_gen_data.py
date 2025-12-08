import os
import sys
import random
from pathlib import Path

import torch
import torchaudio

# ---------------------------------------------------------------------
# 1. Point Python at the DiffRhythm "infer" folder so we can import
#    infer.py and infer_utils.py, which live in infer/.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
INFER_DIR = REPO_ROOT / "infer"
sys.path.insert(0, str(INFER_DIR))

import infer  # this is infer/infer.py, imported as a module
from infer_utils import (
    prepare_model,
    get_lrc_token,
    get_style_prompt,
    get_negative_style_prompt,
    get_reference_latent,
)

# ---------------------------------------------------------------------
# Global storage for activations collected via hooks
# ---------------------------------------------------------------------
ACTIVATIONS = {}


def make_hook(name):
    """Create a forward hook that stores the *output* of a module."""
    def hook(module, inputs, output):
        # LlamaDecoderLayer returns (hidden_states, ...); grab first element
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        out = out.detach().cpu()  # avoid holding GPU graph

        if name not in ACTIVATIONS:
            ACTIVATIONS[name] = []
        ACTIVATIONS[name].append(out)
    return hook


def register_dit_block_hooks(cfm, also_print=False):
    """
    Register hooks on each DiT block (LlamaDecoderLayer) inside CFM.

    In the official code, CFM has a `transformer` attribute that is a DiT,
    and DiT defines `self.transformer_blocks = nn.ModuleList([...])`.
    We hook the top-level entries `transformer_blocks.0`, `transformer_blocks.1`, ...
    """
    if not hasattr(cfm, "transformer"):
        raise AttributeError("Expected `cfm.transformer` to be the DiT backbone.")

    dit = cfm.transformer

    for name, module in dit.named_modules():
        # We want only the top-level blocks, not their submodules:
        # names like 'transformer_blocks.0', 'transformer_blocks.1', ...
        if name.startswith("transformer_blocks.") and name.count(".") == 1:
            full_name = f"dit.{name}"
            module.register_forward_hook(make_hook(full_name))
            if also_print:
                print(f"Registered DiT block hook on: {full_name}")


def main():
    # -----------------------------------------------------------------
    # 2. Basic config: device + duration
    # -----------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_length = 95  # seconds; 95 uses 2048 frames in the official script
    if audio_length == 95:
        max_frames = 2048
    elif 95 < audio_length <= 285:
        max_frames = 6144
    else:
        raise ValueError("audio_length must be 95 or between 96 and 285 (inclusive).")

    # -----------------------------------------------------------------
    # 3. Load models (same helper used by infer/infer.py)
    #    prepare_model returns: (cfm, tokenizer, muq, vae)
    # -----------------------------------------------------------------
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)

    # -----------------------------------------------------------------
    # 3b. Register hooks on DiT blocks inside CFM
    # -----------------------------------------------------------------
    register_dit_block_hooks(cfm, also_print=True)

    # -----------------------------------------------------------------
    # 4. Lyrics (LRC) – use their example file
    #    If you want instrumental only, set lrc = "" instead.
    # -----------------------------------------------------------------
    lrc_path = INFER_DIR / "example" / "eg_en_full.lrc"
    if lrc_path.is_file():
        with open(lrc_path, "r", encoding="utf-8") as f:
            lrc = f.read()
    else:
        # Fallback: no lyrics → effectively instrumental mode
        lrc = ""

    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc, tokenizer, audio_length, device
    )

    # -----------------------------------------------------------------
    # 5. Style prompt
    #    Change this string to steer genre / mood / rough "key".
    # -----------------------------------------------------------------
    style_prompt = get_style_prompt(
        muq,
        prompt="bright orchestral pop in C major, uplifting, piano and strings",
    )
    negative_style_prompt = get_negative_style_prompt(device)

    # -----------------------------------------------------------------
    # 6. Latent prompt for editing / continuation.
    #    For simple generation (no editing), use edit=False and no ref song.
    # -----------------------------------------------------------------
    latent_prompt, pred_frames = get_reference_latent(
        device,
        max_frames,
        False,      # edit
        None,       # edit_segments
        None,       # ref_song
        vae,
    )

    # -----------------------------------------------------------------
    # 7. Run DiffRhythm inference.
    #    Hooks on DiT blocks will fire during the internal CFM forward calls.
    # -----------------------------------------------------------------
    generated_songs = infer.inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=end_frame,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        pred_frames=pred_frames,
        batch_infer_num=1,
        song_duration=song_duration,
        chunked=True,  # reduce VRAM usage; recommended for 8GB cards
    )

    # Take one sample from the batch (here batch_infer_num=1 so it's just that one)
    song = random.choice(generated_songs)

    # -----------------------------------------------------------------
    # 8. Save audio + activations
    # -----------------------------------------------------------------
    output_dir = REPO_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save WAV
    output_path = output_dir / "diffrhythm_output.wav"
    torchaudio.save(str(output_path), song, sample_rate=44100)
    print(f"Saved DiffRhythm output to: {output_path}")

    # Save DiT block activations
    activations_path = output_dir / "dit_block_activations.pt"
    torch.save(ACTIVATIONS, activations_path)
    print(f"Saved DiT block activations to: {activations_path}")

    # Optional: quick summary
    print("Captured DiT activations (layer -> list of shapes):")
    for name, tensors in ACTIVATIONS.items():
        shapes = [tuple(t.shape) for t in tensors]
        print(f"  {name}: {shapes}")


if __name__ == "__main__":
    main()
