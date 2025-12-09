import os
import sys
from pathlib import Path

import torch
import torchaudio
import numpy as np
import librosa

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
NUM_CHUNKS = 10               # 10 equal chunks per song

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


# ---------------------------------------------------------------------
# SIMPLE K-K STYLE KEY DETECTOR (returns key + score)
# ---------------------------------------------------------------------

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33,
                     4.38, 4.09, 2.52, 5.19,
                     2.39, 3.66, 2.29, 2.88], dtype=np.float32)

KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38,
                     2.60, 3.53, 2.54, 4.75,
                     3.98, 2.69, 3.34, 3.17], dtype=np.float32)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def detect_key_from_waveform_torch(wave: torch.Tensor, sr: int):
    """
    Given a torch waveform [channels, samples] and sample rate sr,
    run a simple Krumhansl–Kessler key detection via librosa chroma_cqt.

    Returns:
        best_name: "C major", "A minor", etc.
        best_score: cosine similarity in [-1, 1], usually 0.1–0.9
    """
    # to mono
    if wave.dim() == 2:
        mono = wave.mean(dim=0)
    else:
        mono = wave

    y = mono.detach().cpu().numpy().astype(np.float32)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_energy = chroma.sum(axis=1)

    if np.all(chroma_energy == 0):
        return "Unknown", 0.0

    chroma_norm = chroma_energy / (np.linalg.norm(chroma_energy) + 1e-8)

    kk_major_norm = KK_MAJOR / (np.linalg.norm(KK_MAJOR) + 1e-8)
    kk_minor_norm = KK_MINOR / (np.linalg.norm(KK_MINOR) + 1e-8)

    best_score = -1e9
    best_name = "Unknown"

    for i, note in enumerate(NOTE_NAMES):
        maj_prof = np.roll(kk_major_norm, i)
        min_prof = np.roll(kk_minor_norm, i)

        maj_score = float(np.dot(chroma_norm, maj_prof))
        min_score = float(np.dot(chroma_norm, min_prof))

        if maj_score > best_score:
            best_score = maj_score
            best_name = f"{note} major"

        if min_score > best_score:
            best_score = min_score
            best_name = f"{note} minor"

    return best_name, best_score


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # About 100 seconds of audio, with max_frames a multiple of 10
    audio_length = 100
    max_frames = 2100   # multiple of 10 → latent time length likely divisible by 10

    # Load models once
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device)
    register_dit_block_hooks(cfm, also_print=True)

    # Output root dir
    output_root = REPO_ROOT / "tmp_outputs"
    output_root.mkdir(exist_ok=True, parents=True)

    # Build all keys list (24 keys)
    note_names = ["C", "C#", "D", "D#", "E", "F",
                  "F#", "G", "G#", "A", "A#", "B"]
    all_keys = [f"{n} major" for n in note_names] + \
               [f"{n} minor" for n in note_names]

    # Moods to loop over
    moods = [
        "Romantic",
        "Nostalgic",
        "Heartfelt",
        "Happy",
        "Melancholic",
        "Love",
        "Upbeat",
        "Energetic",
        "Uplifting",
        "Carefree",
    ]

    # Global sample index across all (mood, key, chunk)
    sample_idx = 0

    for mood in moods:
        for key_str in all_keys:
            print(f"\n=== Generating for mood={mood}, target key={key_str} ===")

            # Reset activations for this (mood, key)
            global ACTIVATIONS
            ACTIVATIONS = {}

            # Empty lyrics → instrumental behavior
            lrc = ""
            lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
                max_frames, lrc, tokenizer, audio_length, device
            )

            # Latent conditioning
            latent_prompt, pred_frames = get_reference_latent(
                device, max_frames, False, None, None, vae
            )

            # Style prompt using this mood + key (only for conditioning, not filenames)
            GENRE = "Classical"
            INSTRUMENT = "Piano"
            style_keywords = f"{GENRE}, {mood}, {INSTRUMENT}, key: {key_str}"
            style_prompt = get_style_prompt(muq, prompt=style_keywords)

            negative_style_prompt = get_negative_style_prompt(device)

            # -----------------------------
            # Inference
            # -----------------------------
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

            # -----------------------------
            # AUDIO: split into 10 equal chunks
            # -----------------------------
            sample_rate = 44100
            total_samples = song.shape[-1]

            print(f"[{mood} | {key_str}] Total samples from model: {total_samples}")

            effective_samples = (total_samples // NUM_CHUNKS) * NUM_CHUNKS
            song = song[:, :effective_samples]

            chunk_samples = effective_samples // NUM_CHUNKS
            print(f"[{mood} | {key_str}] Effective samples used: {effective_samples}")
            print(f"[{mood} | {key_str}] Each audio chunk: {chunk_samples} samples (~{chunk_samples / sample_rate:.3f}s)")

            chunk_audio = []
            chunk_keys = []  # (detected_key, score)

            for i in range(NUM_CHUNKS):
                start = i * chunk_samples
                end = start + chunk_samples
                chunk_wave = song[:, start:end]
                chunk_audio.append(chunk_wave)

                detected_key, key_score = detect_key_from_waveform_torch(chunk_wave, sample_rate)
                chunk_keys.append((detected_key, key_score))
                print(f"[{mood} | {key_str}] Chunk {i}: detected key = {detected_key}, score = {key_score:.4f}")

            # -----------------------------
            # ACTIVATIONS: equal-length token chunks
            # -----------------------------
            chunked_activations = [dict() for _ in range(NUM_CHUNKS)]

            print(f"\n[{mood} | {key_str}] === ACTIVATION SPLITTING DEBUG INFO ===")
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
                    print(f"\n{name}: ndim {base.ndim}, copying whole tensor into all chunks.")
                    for i in range(NUM_CHUNKS):
                        chunked_activations[i][name] = base
                    continue

                # Concatenate multiple hook outputs if present
                if len(tensors) > 1:
                    full = torch.cat(tensors, dim=time_dim)
                else:
                    full = base

                L = full.shape[time_dim]
                L_trunc = (L // NUM_CHUNKS) * NUM_CHUNKS
                chunk_len = L_trunc // NUM_CHUNKS
                cropped = full.narrow(time_dim, 0, L_trunc)

                print(f"\n{name}:")
                print(f"  original shape = {tuple(full.shape)}")
                print(f"  total tokens L = {L}")
                print(f"  L_trunc        = {L_trunc}")
                print(f"  chunk_len      = {chunk_len}")

                for i in range(NUM_CHUNKS):
                    start_idx = i * chunk_len
                    ch = cropped.narrow(time_dim, start_idx, chunk_len)
                    print(f"  chunk {i} shape = {tuple(ch.shape)}")
                    chunked_activations[i][name] = ch

            # -----------------------------
            # SAVE: audio_N.wav, activations_N.pt, key_N.txt (global index)
            # -----------------------------
            for i in range(NUM_CHUNKS):
                idx_str = f"{sample_idx:04d}"

                # Audio
                wav_path = output_root / f"audio_{idx_str}.wav"
                torchaudio.save(str(wav_path), chunk_audio[i], sample_rate)

                # Activations
                act_path = output_root / f"activations_{idx_str}.pt"
                torch.save(chunked_activations[i], act_path)

                # Key prediction + score
                detected_key, key_score = chunk_keys[i]
                key_path = output_root / f"key_{idx_str}.txt"
                with open(key_path, "w") as f:
                    f.write(f"{detected_key}\n")
                    f.write(f"score={key_score:.6f}\n")

                # Progress marker for you to grep / watch
                print(
                    f"tadaqum idx={idx_str} mood={mood} target_key={key_str} "
                    f"chunk={i} detected={detected_key} score={key_score:.4f}"
                )

                sample_idx += 1

    print("\nAll moods and keys processed. Total samples:", sample_idx)


if __name__ == "__main__":
    main()
