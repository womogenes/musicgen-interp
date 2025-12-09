import os
import sys
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
    get_negative_style_prompt,
    normalize_audio,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
TARGET_KEY = "C major"        # generate only this key for now
NUM_CHUNKS = 10               # 10 equal chunks
CHUNK_DURATION_SECONDS = 10   # ~10s per chunk if total ≈ 100s

# ---------------------------------------------------------------------
# Upload helper
# ---------------------------------------------------------------------
def upload_to_gdrive(local_path, remote_folder):
    """
    Uploads a file without deleting the local copy.
    """
    local_path = Path(local_path)
    remote_path = f"gdrive:{remote_folder}/"
    print(f"Uploading {local_path} → {remote_path}")
    subprocess.run(["rclone", "copy", str(local_path), remote_path], check=True)
    print("Kept local file:", local_path)


# ---------------------------------------------------------------------
# ACTIVATIONS (collected via hooks)
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
    """
    Register hooks only on transformer blocks whose index is a multiple of 4:
    transformer_blocks.0, transformer_blocks.4, transformer_blocks.8, ...
    """
    if not hasattr(cfm, "transformer"):
        raise AttributeError("Expected `cfm.transformer` to be the DiT backbone.")
    dit = cfm.transformer

    for name, module in dit.named_modules():
        if name.startswith("transformer_blocks.") and name.count(".") == 1:
            try:
                idx = int(name.split(".")[1])
            except ValueError:
                continue

            if idx % 4 != 0:
                continue

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # About 100 seconds of audio, but now with max_frames divisible by 10
    audio_length = 100
    max_frames = 2100   # multiple of 10 → latent time length more likely divisible by 10

    # Load models
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    register_dit_block_hooks(cfm, also_print=True)

    # Empty lyrics → instrumental behavior
    lrc = ""
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc, tokenizer, audio_length, device
    )

    # Latent conditioning
    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, False, None, None, vae
    )

    # Output dir
    output_dir = REPO_ROOT / "tmp_outputs"
    output_dir.mkdir(exist_ok=True, parents=True)

    global ACTIVATIONS
    ACTIVATIONS = {}

    key_str = TARGET_KEY
    key_tag = key_to_tag(key_str)

    print(f"\n=== Generating Classical / Heartfelt / Piano in {key_str} (~100s) ===")

    # MuQ keyword prompt with explicit "key:"
    GENRE = "Classical"
    MOOD = "Heartfelt"
    INSTRUMENT = "Piano"
    style_keywords = f"{GENRE}, {MOOD}, {INSTRUMENT}, key: {key_str}"
    style_prompt = get_style_prompt(muq, prompt=style_keywords)

    # Built-in negative prompt
    negative_style_prompt = get_negative_style_prompt(device)

    # -----------------------------------------------------------------
    # INFERENCE (~100s)
    # -----------------------------------------------------------------
    songs = infer.inference(
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
        chunked=True,
    )

    song = songs[0]     # [channels, samples]
    song = normalize_audio(song, target_dbfs=-6)

    # -----------------------------------------------------------------
    # AUDIO: split into 10 equal time chunks
    # -----------------------------------------------------------------
    sample_rate = 44100
    total_samples = song.shape[-1]

    print(f"Total samples from model: {total_samples}")

    # Make total_samples divisible by NUM_CHUNKS
    effective_samples = (total_samples // NUM_CHUNKS) * NUM_CHUNKS
    song = song[:, :effective_samples]

    chunk_samples = effective_samples // NUM_CHUNKS
    print(f"Effective samples used: {effective_samples}")
    print(f"Each audio chunk: {chunk_samples} samples (~{chunk_samples / sample_rate:.3f}s)")

    for i in range(NUM_CHUNKS):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk_wave = song[:, start:end]

        chunk_wav_path = output_dir / f"{key_tag}_chunk{i:02d}.wav"
        torchaudio.save(str(chunk_wav_path), chunk_wave, sample_rate)
        print(f"Saved WAV chunk {i}: {chunk_wav_path} ({chunk_wave.shape})")

        upload_to_gdrive(chunk_wav_path, remote_folder="diffrhythm_chunks_wav")


    # -----------------------------------------------------------------
    # ACTIVATIONS: equal-length, time-aligned token chunks
    # -----------------------------------------------------------------
    chunked_activations = [dict() for _ in range(NUM_CHUNKS)]

    print("\n=== ACTIVATION SPLITTING DEBUG INFO (equal tokens per chunk) ===")

    for name, tensors in ACTIVATIONS.items():
        if not tensors:
            continue

        base = tensors[0]

        # Identify time dimension
        if base.ndim >= 3:
            time_dim = 1       # [B, T, C] or similar
        elif base.ndim == 2:
            time_dim = 0       # [T, C]
        else:
            print(f"\n{name}: ndim {base.ndim}, treating as non-time, copying to all chunks.")
            for i in range(NUM_CHUNKS):
                chunked_activations[i][name] = base
            continue

        # Concatenate multiple hook outputs if present
        if len(tensors) > 1:
            full = torch.cat(tensors, dim=time_dim)
        else:
            full = base

        L = full.shape[time_dim]

        # Trim so L_trunc is divisible by NUM_CHUNKS
        L_trunc = (L // NUM_CHUNKS) * NUM_CHUNKS
        chunk_len = L_trunc // NUM_CHUNKS

        cropped = full.narrow(time_dim, 0, L_trunc)

        print(f"\n{name}:")
        print(f"  original shape = {tuple(full.shape)}")
        print(f"  total tokens L = {L}")
        print(f"  L_trunc        = {L_trunc}")
        print(f"  chunk_len      = {chunk_len}")

        # Equal-length token chunks
        for i in range(NUM_CHUNKS):
            start_idx = i * chunk_len
            ch = cropped.narrow(time_dim, start_idx, chunk_len)
            print(f"  chunk {i} shape = {tuple(ch.shape)}")
            chunked_activations[i][name] = ch

    # Save activation chunks
    for i in range(NUM_CHUNKS):
        act_path = output_dir / f"activations_{key_tag}_chunk{i:02d}.pt"
        torch.save(chunked_activations[i], act_path)
        print(f"Saved activations chunk {i}: {act_path}")

    print(f"\nFinished {key_str}.\n")

    # -----------------------------------------------------------------
    # ORIGINAL FULL KEY LOOP (COMMENTED OUT)
    # -----------------------------------------------------------------
    """
    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    all_keys = [f"{n} major" for n in note_names] + \
               [f"{n} minor" for n in note_names]

    for key_str in all_keys:
        # original full-loop logic...
        pass
    """


if __name__ == "__main__":
    main()
