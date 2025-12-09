import os
import sys
import random
import subprocess
from pathlib import Path

import torch
import torchaudio

# ---------------------------------------------------------------------
# 1. Point Python at the DiffRhythm "infer" folder
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
INFER_DIR = REPO_ROOT / "infer"
sys.path.insert(0, str(INFER_DIR))

import infer
from infer_utils import (
    prepare_model,
    get_lrc_token,
    get_style_prompt,
    get_reference_latent,
    get_negative_style_prompt,   # built-in negative prompt
    normalize_audio,             # <-- use this for output normalization
)

# ---------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------
def upload_to_gdrive(local_path, remote_folder):
    """
    Uploads a file to Google Drive (gdrive remote must be configured).
    DOES NOT delete locally.
    """
    local_path = Path(local_path)
    remote_path = f"gdrive:{remote_folder}/"
    print(f"Uploading {local_path} → {remote_path}")
    subprocess.run(["rclone", "copy", str(local_path), remote_path], check=True)
    print("Kept local file:", local_path)


# ---------------------------------------------------------------------
# ACTIVATIONS
# ---------------------------------------------------------------------
ACTIVATIONS = {}

def make_hook(name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        out = out.detach().cpu()
        ACTIVATIONS.setdefault(name, []).append(out)
    return hook

def register_dit_block_hooks(cfm, also_print=False):
    if not hasattr(cfm, "transformer"):
        raise AttributeError("Expected `cfm.transformer` to be the DiT backbone.")
    dit = cfm.transformer
    for name, module in dit.named_modules():
        if name.startswith("transformer_blocks.") and name.count(".") == 1:
            full_name = f"dit.{name}"
            module.register_forward_hook(make_hook(full_name))
            if also_print:
                print("Registered hook on:", full_name)

def key_to_tag(key_str: str) -> str:
    return key_str.replace("#", "s").replace(" ", "_")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    # Duration = 100 seconds
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_length = 100
    max_frames = 2048  # OK for ~100s with v1.2

    # Load models (your local version likely has this signature)
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    register_dit_block_hooks(cfm, also_print=False)

    # Empty lyrics → instrumental mode
    lrc = ""
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc, tokenizer, audio_length, device
    )

    # Latent prompt
    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, False, None, None, vae
    )

    # Keys
    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    all_keys = [f"{n} major" for n in note_names] + \
               [f"{n} minor" for n in note_names]

    # Output dir (local)
    output_dir = REPO_ROOT / "tmp_outputs"
    output_dir.mkdir(exist_ok=True, parents=True)

    global ACTIVATIONS

    # MuQ keyword categories
    GENRE = "Classical"
    MOOD = "Heartfelt"
    INSTRUMENT = "Piano"

    for key_str in all_keys:
        print(f"\n=== Generating {GENRE}/{MOOD}/{INSTRUMENT} in {key_str} (100 sec) ===")
        ACTIVATIONS = {}

        key_tag = key_to_tag(key_str)

        # STYLE PROMPT (with literal word "key")
        style_keywords = f"{GENRE}, {MOOD}, {INSTRUMENT}, key: {key_str}"
        style_prompt = get_style_prompt(muq, prompt=style_keywords)

        # Built-in negative prompt (same as HF)
        negative_style_prompt = get_negative_style_prompt(device)

        # Inference (your local infer.inference signature)
        songs = infer.inference(
            cfm_model=cfm,
            vae_model=vae,
            cond=latent_prompt,
            text=lrc_prompt,                # empty → instrumental conditioning
            duration=end_frame,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            start_time=start_time,
            pred_frames=pred_frames,
            batch_infer_num=1,
            song_duration=song_duration,
            chunked=True,
        )

        # Take first (and only) song
        song = songs[0]   # expected shape: [channels, samples]

        # -----------------------------------------------------------------
        # POST-GEN NORMALIZATION (approx HF behavior)
        # -----------------------------------------------------------------
        # HF CLI does peak normalization before saving; we mimic using
        # the normalize_audio helper (target loudness around -6 dBFS).
        song = normalize_audio(song, target_dbfs=-6)

        # Save WAV locally and KEEP it
        wav_path = output_dir / f"piano_{key_tag}.wav"
        torchaudio.save(str(wav_path), song, sample_rate=44100)
        print("Saved local WAV:", wav_path)

        # Also upload to Drive, but keep local copy
        upload_to_gdrive(wav_path, remote_folder="diffrhythm_piano_wavs")

        # Save & delete activations (to keep disk usage small)
        activations_path = output_dir / f"activations_{key_tag}.pt"
        torch.save(ACTIVATIONS, activations_path)
        activations_path.unlink()
        print("(Deleted activations locally — upload disabled)")

        print(f"Finished {key_str}.\n")


if __name__ == "__main__":
    main()
