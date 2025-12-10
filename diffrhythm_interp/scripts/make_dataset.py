#!/usr/bin/env python3
"""
DiffRhythm dataset generation script.

Generates 50 music samples using DiffRhythm, chunks each into 5-second intervals,
runs key detection on each chunk, and saves metadata similar to MusicGen dataset.

NOTE: This script does NOT require espeak since we generate instrumentals only
(empty lyrics = zeros tensor, no phonemization needed).
"""

import os
import sys
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import scipy.io.wavfile as wavfile
import librosa
from huggingface_hub import hf_hub_download
from muq import MuQMuLan

# ============================================================================
# CONFIGURATION
# ============================================================================
TEST_MODE = "--test" in sys.argv

if TEST_MODE:
    TARGET_TOTAL_SAMPLES = 5
    CHECKPOINT_EVERY = 2
    print("🧪 TEST MODE: only 5 samples")
else:
    TARGET_TOTAL_SAMPLES = 50
    CHECKPOINT_EVERY = 5

# Output directory
OUTPUT_DIR = Path("/home/wyf/orcd/pool/diffrhythm")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_CHUNKS_DIR = OUTPUT_DIR / "audio_chunks"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
METADATA_DIR = OUTPUT_DIR / "metadata"

# DiffRhythm config
SAMPLING_RATE = 44100
CHUNK_DURATION = 5  # seconds per chunk for key detection
AUDIO_LENGTH = 95   # seconds per generation
MAX_FRAMES = 2048

# Setup paths for DiffRhythm imports
SCRIPT_DIR = Path(__file__).resolve().parent
DIFFRHYTHM_ROOT = SCRIPT_DIR.parent / "DiffRhythm"
INFER_DIR = DIFFRHYTHM_ROOT / "infer"
sys.path.insert(0, str(DIFFRHYTHM_ROOT))
sys.path.insert(0, str(INFER_DIR))

# Change to DiffRhythm root for proper config loading
os.chdir(str(DIFFRHYTHM_ROOT))

# Import only what we need (avoiding tokenizer which needs espeak)
from infer_utils import (
    get_style_prompt,
    get_negative_style_prompt,
    get_reference_latent,
    normalize_audio,
)
from model import DiT, CFM
import infer as diffrhythm_infer

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ============================================================================
# MODEL LOADING (without espeak-dependent tokenizer)
# ============================================================================
def load_checkpoint_no_ema(model, ckpt_path, device):
    """Load checkpoint without EMA."""
    model = model.half()
    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        checkpoint = load_file(ckpt_path)
        checkpoint = {"model_state_dict": checkpoint}
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True)
        if "model_state_dict" not in checkpoint:
            checkpoint = {"model_state_dict": checkpoint}
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return model.to(device)


def prepare_model_instrumental(max_frames, device):
    """
    Load DiffRhythm models WITHOUT the tokenizer (no espeak needed).
    Only works for instrumental generation (empty lyrics).
    """
    # Determine repo based on max_frames
    if max_frames == 2048:
        repo_id = "ASLP-lab/DiffRhythm-1_2"
    else:
        repo_id = "ASLP-lab/DiffRhythm-1_2-full"

    # Load DiT/CFM model
    dit_ckpt_path = hf_hub_download(
        repo_id=repo_id, filename="cfm_model.pt", cache_dir="./pretrained"
    )
    dit_config_path = "./config/diffrhythm-1b.json"
    with open(dit_config_path) as f:
        model_config = json.load(f)

    cfm = CFM(
        transformer=DiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    )
    cfm = cfm.to(device)
    cfm = load_checkpoint_no_ema(cfm, dit_ckpt_path, device)

    # Load MuQ (style encoder) - no espeak needed
    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./pretrained")
    muq = muq.to(device).eval()

    # Load VAE
    vae_ckpt_path = hf_hub_download(
        repo_id="ASLP-lab/DiffRhythm-vae",
        filename="vae_model.pt",
        cache_dir="./pretrained",
    )
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)

    return cfm, muq, vae


def get_empty_lrc_token(max_frames, audio_length, device):
    """
    Create empty lyrics tensor for instrumental generation.
    No tokenizer/espeak needed - just returns zeros.
    """
    end_frame = max_frames if max_frames == 2048 else int(audio_length * (SAMPLING_RATE / 2048))
    end_frame = min(end_frame, max_frames)

    # Empty lyrics = zeros
    lrc_emb = torch.zeros((1, end_frame), dtype=torch.long, device=device)

    # Normalized start time and duration
    normalized_start_time = torch.tensor(0.0, device=device).unsqueeze(0).half()
    normalized_duration = torch.tensor(end_frame / max_frames, device=device).unsqueeze(0).half()

    return lrc_emb, normalized_start_time, end_frame, normalized_duration

# ============================================================================
# KK KEY DEFINITIONS
# ============================================================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ============================================================================
# PROMPT GENERATION (NO KEY IN PROMPT - classical/romantic piano focused)
# Avoiding jazz, contemporary, atonal styles that are hard for key detection
# ============================================================================
MOODS = [
    "romantic", "nostalgic", "heartfelt", "happy", "melancholic",
    "upbeat", "energetic", "uplifting", "hopeful", "dreamy",
    "peaceful", "calm", "dramatic", "tender", "triumphant",
    "reflective", "gentle", "passionate", "serene", "bright",
]

STYLES = [
    "classical", "romantic era", "baroque",
    "hymn-like", "chorale", "folk",
    "pop ballad", "cinematic", "minimalist",
]

TEXTURES = [
    "arpeggios", "broken chords", "block chords",
    "rolling arpeggios", "ostinato pattern", "Alberti bass",
    "flowing melody", "simple triads", "octave doublings",
    "legato phrases", "staccato notes", "pedal sustain",
]

TEMPOS = [
    "slow", "moderate", "medium tempo", "gentle",
    "flowing", "steady", "unhurried", "andante",
]

PROMPT_TEMPLATES = [
    "{style}, piano, {mood}, {texture}",
    "{mood}, {style} piano, {texture}, {tempo}",
    "piano, {style}, {mood}, featuring {texture}",
    "{tempo} {mood} piano piece, {style}, {texture}",
]


def generate_prompt() -> str:
    """Generate a random prompt focused on classical/romantic piano."""
    template = random.choice(PROMPT_TEMPLATES)
    return template.format(
        mood=random.choice(MOODS),
        style=random.choice(STYLES),
        texture=random.choice(TEXTURES),
        tempo=random.choice(TEMPOS),
    )


# ============================================================================
# ACTIVATION HOOKS (capture DiT block outputs)
# ============================================================================
ACTIVATIONS = {}


def make_hook(name):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        ACTIVATIONS.setdefault(name, []).append(out.detach().cpu())
    return hook


def register_dit_block_hooks(cfm):
    """Register hooks on DiT transformer blocks."""
    dit = cfm.transformer
    for name, module in dit.named_modules():
        if name.startswith("transformer_blocks.") and name.count(".") == 1:
            idx = int(name.split(".")[1])
            # Hook every 4th block to reduce memory
            if idx % 4 == 0:
                module.register_forward_hook(make_hook(f"dit.{name}"))
                print(f"Registered hook on: {name}")


# ============================================================================
# KEY DETECTION (librosa chroma-based, no MIDI/BasicPitch needed)
# ============================================================================
def detect_key_from_waveform(wave, sr):
    """Detect key directly from waveform using librosa chroma."""
    if isinstance(wave, torch.Tensor):
        mono = wave.mean(dim=0) if wave.dim() == 2 else wave
        y = mono.cpu().numpy().astype(np.float32)
    else:
        y = wave
        if y.ndim == 2:
            y = y.mean(axis=0)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_energy = chroma.sum(axis=1)

    if np.all(chroma_energy == 0):
        return {"key": None, "note": None, "mode": None, "confidence": 0.0, "error": "No chroma energy"}

    chroma_norm = chroma_energy / (np.linalg.norm(chroma_energy) + 1e-8)

    major = KK_MAJOR / (np.linalg.norm(KK_MAJOR) + 1e-8)
    minor = KK_MINOR / (np.linalg.norm(KK_MINOR) + 1e-8)

    best_score = -999
    best_note = None
    best_mode = None

    for i, note in enumerate(NOTE_NAMES):
        maj = np.roll(major, i)
        minr = np.roll(minor, i)

        ms = np.dot(chroma_norm, maj)
        ns = np.dot(chroma_norm, minr)

        if ms > best_score:
            best_score = ms
            best_note = note
            best_mode = "major"

        if ns > best_score:
            best_score = ns
            best_note = note
            best_mode = "minor"

    return {
        "key": f"{best_note}_{best_mode}",
        "note": best_note,
        "mode": best_mode,
        "confidence": float(best_score),
        "error": None,
    }


# ============================================================================
# AUDIO CHUNKING
# ============================================================================
def chunk_audio(audio_tensor, sample_rate, chunk_duration):
    """Split audio tensor into chunks of specified duration."""
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    chunk_samples = int(chunk_duration * sample_rate)
    total_samples = audio_tensor.shape[-1]

    chunks = []
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_samples, total_samples)
        chunk = audio_tensor[..., start_sample:end_sample]

        # Only include chunks that are at least half the target duration
        if chunk.shape[-1] >= chunk_samples // 2:
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            chunks.append((chunk, start_time, end_time))

        start_sample = end_sample

    return chunks


def chunk_activations(activations_dict, num_chunks):
    """Split activations into chunks matching audio chunks."""
    chunked = [{} for _ in range(num_chunks)]

    for name, tensors in activations_dict.items():
        base = tensors[0]
        if base.ndim >= 3:
            td = 1  # time dimension
        elif base.ndim == 2:
            td = 0
        else:
            for i in range(num_chunks):
                chunked[i][name] = base
            continue

        full = torch.cat(tensors, dim=td) if len(tensors) > 1 else base
        L = full.shape[td]
        L_trunc = (L // num_chunks) * num_chunks
        if L_trunc == 0:
            continue
        clen = L_trunc // num_chunks
        cropped = full.narrow(td, 0, L_trunc)

        for i in range(num_chunks):
            start = i * clen
            chunked[i][name] = cropped.narrow(td, start, clen)

    return chunked


# ============================================================================
# PROGRESS / RESUME
# ============================================================================
def save_progress(total: int):
    progress_path = OUTPUT_DIR / "progress.json"
    with open(progress_path, "w") as f:
        json.dump({"total_samples": int(total)}, f, indent=2)


def load_progress() -> int:
    progress_path = OUTPUT_DIR / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            data = json.load(f)
        return int(data.get("total_samples", 0))
    return 0


def get_starting_clip_id() -> int:
    if METADATA_DIR.exists():
        return len(list(METADATA_DIR.glob("*.json")))
    return 0


# ============================================================================
# MERGE METADATA
# ============================================================================
def merge_metadata():
    """Merge individual metadata JSONs and print key distribution."""
    all_metadata = []
    key_counts = {}

    for meta_file in sorted(METADATA_DIR.glob("*.json")):
        with open(meta_file) as f:
            m = json.load(f)
        all_metadata.append(m)

        # Count keys from chunks
        for chunk in m.get("chunks", []):
            k = chunk.get("key_info", {}).get("key")
            if k is not None:
                key_counts[k] = key_counts.get(k, 0) + 1

    combined_path = OUTPUT_DIR / "dataset_metadata.json"
    with open(combined_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Merged {len(all_metadata)} samples into {combined_path}")
    print("\nDetected-key distribution (across all chunks):")
    for k, v in sorted(key_counts.items(), key=lambda x: x[0]):
        print(f"  {k:12s}: {v}")


# ============================================================================
# GENERATION
# ============================================================================
def generate_diffrhythm_sample(cfm, muq, vae, prompt):
    """Generate a single music sample using DiffRhythm (instrumental only, no espeak)."""
    global ACTIVATIONS
    ACTIVATIONS = {}

    # Empty lyrics for instrumental - no tokenizer needed!
    lrc_prompt, start_time, end_frame, song_duration = get_empty_lrc_token(
        MAX_FRAMES, AUDIO_LENGTH, device
    )

    # Style prompt
    style_prompt = get_style_prompt(muq, prompt=prompt)
    negative_style_prompt = get_negative_style_prompt(device)

    # Latent prompt (no editing)
    latent_prompt, pred_frames = get_reference_latent(
        device, MAX_FRAMES, False, None, None, vae
    )

    # Generate
    generated_songs = diffrhythm_infer.inference(
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

    # Normalize audio
    song = normalize_audio(generated_songs[0], target_dbfs=-6)

    return song, dict(ACTIVATIONS)


# ============================================================================
# MAIN
# ============================================================================
def main():
    # Make dirs
    for d in [OUTPUT_DIR, AUDIO_DIR, AUDIO_CHUNKS_DIR, ACTIVATIONS_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if device != "cuda":
        print("⚠️ CUDA not available; this will be very slow on CPU.")

    # Resume info - start at clip 10
    total_so_far = max(50, load_progress())  # Already have 10 samples
    clip_id = max(50, get_starting_clip_id())  # Start at clip 0010
    print(f"Resuming from total_so_far={total_so_far}, next clip_id={clip_id}")

    if total_so_far >= TARGET_TOTAL_SAMPLES:
        print("Already reached TARGET_TOTAL_SAMPLES, nothing to do.")
        merge_metadata()
        return

    # Load DiffRhythm models (no tokenizer = no espeak needed)
    print("Loading DiffRhythm models (instrumental mode, no espeak required)...")
    cfm, muq, vae = prepare_model_instrumental(MAX_FRAMES, device)
    register_dit_block_hooks(cfm)
    print("Models loaded successfully!")

    start_time_total = time.time()

    while total_so_far < TARGET_TOTAL_SAMPLES:
        clip_id_str = f"{clip_id:04d}"
        prompt = generate_prompt()

        print(f"\n=== [{clip_id_str}] '{prompt}' ===")
        print(f"    (sample {total_so_far + 1}/{TARGET_TOTAL_SAMPLES})")

        try:
            # Generate audio
            song_tensor, activations = generate_diffrhythm_sample(
                cfm, muq, vae, prompt
            )

            # Save full audio (scipy expects shape [samples, channels])
            wav_path = AUDIO_DIR / f"{clip_id_str}.wav"
            audio_np = song_tensor.cpu().numpy().T  # [channels, samples] -> [samples, channels]
            wavfile.write(str(wav_path), SAMPLING_RATE, audio_np)
            print(f"    Saved audio to {wav_path}")

            # Convert to float for chunking
            audio_float = song_tensor.float() / 32767.0

            # Chunk audio
            chunks = chunk_audio(audio_float, SAMPLING_RATE, CHUNK_DURATION)
            num_chunks = len(chunks)
            print(f"    Split into {num_chunks} chunks of ~{CHUNK_DURATION}s each")

            # Chunk activations
            chunked_acts = chunk_activations(activations, num_chunks)

            chunk_metadata = []
            for chunk_idx, (chunk_tensor, chunk_start, chunk_end) in enumerate(chunks):
                chunk_id = f"{clip_id_str}_chunk{chunk_idx:02d}"

                # Save chunk audio (scipy expects shape [samples, channels])
                # chunk_wav_path = AUDIO_CHUNKS_DIR / f"{chunk_id}.wav"
                # chunk_int16 = (chunk_tensor * 32767).clamp(-32768, 32767).to(torch.int16)
                # chunk_np = chunk_int16.cpu().numpy().T  # [channels, samples] -> [samples, channels]
                # wavfile.write(str(chunk_wav_path), SAMPLING_RATE, chunk_np)

                # Save chunk activations
                chunk_act_path = ACTIVATIONS_DIR / f"{chunk_id}.pt"
                if chunk_idx < len(chunked_acts):
                    torch.save(chunked_acts[chunk_idx], str(chunk_act_path))

                # Detect key from waveform using librosa chroma
                key_info = detect_key_from_waveform(chunk_tensor, SAMPLING_RATE)

                if key_info['key']:
                    print(f"      Chunk {chunk_idx} ({chunk_start:.1f}s-{chunk_end:.1f}s): "
                          f"{key_info['note']} {key_info['mode']} (conf: {key_info['confidence']:.3f})")
                else:
                    print(f"      Chunk {chunk_idx} ({chunk_start:.1f}s-{chunk_end:.1f}s): "
                          f"Key detection failed - {key_info['error']}")

                chunk_metadata.append({
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_idx,
                    'start_time': chunk_start,
                    'end_time': chunk_end,
                    'duration': chunk_end - chunk_start,
                    # 'audio_path': str(chunk_wav_path),
                    'activations_path': str(chunk_act_path),
                    'key_info': key_info,
                    'label_key': key_info['key'],
                })

            # Build metadata entry
            metadata = {
                'clip_id': clip_id_str,
                'prompt': prompt,
                'audio_path': str(wav_path),
                'sampling_rate': SAMPLING_RATE,
                'audio_length_seconds': AUDIO_LENGTH,
                'chunk_duration_seconds': CHUNK_DURATION,
                'num_chunks': num_chunks,
                'chunks': chunk_metadata,
            }

            # Save individual metadata file
            meta_path = METADATA_DIR / f"{clip_id_str}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            clip_id += 1
            total_so_far += 1

            if total_so_far % CHECKPOINT_EVERY == 0:
                save_progress(total_so_far)

        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final save
    save_progress(total_so_far)

    elapsed = time.time() - start_time_total
    print("\n" + "=" * 60)
    print("DIFFRHYTHM DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {total_so_far}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"Samples per minute: {total_so_far / (elapsed/60):.2f}")

    merge_metadata()
    print(f"\nDataset saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
