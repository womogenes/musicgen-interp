"""
No-key MusicGen dataset generation (single GPU, 1000 samples).

- Prompts NEVER mention a musical key.
- We generate audio from text-only prompts (mood/tempo/style/texture).
- We run BasicPitch + Krumhansl-Kessler on the MIDI to detect key.
- We store that detected key as the label (label_key) plus full key_info.
- Single GPU, simple loop, with resume support via progress.json.
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
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi

# ============================================================================
# CONFIGURATION
# ============================================================================
TEST_MODE = "--test" in sys.argv

if TEST_MODE:
    TARGET_TOTAL_SAMPLES = 10   # tiny for debugging
    CHECKPOINT_EVERY = 2
    print("🧪 TEST MODE: only 10 samples total")
else:
    TARGET_TOTAL_SAMPLES = 1000
    CHECKPOINT_EVERY = 20

# Use a separate dir so you don't collide with key-in-prompt dataset
OUTPUT_DIR = Path("/home/harinit9/orcd/pool/musicgen-data-nokey/")
AUDIO_DIR = OUTPUT_DIR / "audio"
MIDI_DIR = OUTPUT_DIR / "midi"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
METADATA_DIR = OUTPUT_DIR / "metadata"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
# PROMPT GENERATION (NO KEY MENTION)
# ============================================================================
MOODS = [
    "happy", "sad", "melancholic", "dark", "bright", "uplifting",
    "dreamy", "nostalgic", "dramatic", "peaceful", "tense", "playful",
    "cinematic", "introspective", "mysterious", "romantic", "bittersweet",
]

TEMPOS = [
    "very slow", "slow", "medium tempo", "moderate", "up-tempo", "fast",
]

TEXTURES = [
    "simple melody with left-hand triads",
    "broken chord arpeggios",
    "block chords with pedal sustain",
    "sparse single notes",
    "rolling arpeggios spanning octaves",
    "repeating ostinato pattern",
    "steady eighth-note accompaniment",
    "flowing sixteenth-note runs",
    "staccato chord stabs",
    "Alberti bass pattern",
]

STYLES = [
    "classical style", "romantic style", "jazzy voicings", "blues harmony",
    "minimalist", "film-score style", "pop ballad style", "impressionistic",
]

NO_KEY_TEMPLATES = [
    "A {tempo}, {mood} solo piano piece, {style}, featuring {texture}.",
    "Solo piano at {tempo}, {mood} overall, with {texture}.",
    "{mood} solo piano, {tempo}, using {style} and {texture}.",
    "A {mood}, {tempo} piano piece with {texture}, in a {style} idiom.",
]

def generate_prompt_no_key() -> str:
    """Generate a random prompt with NO explicit key tokens."""
    template = random.choice(NO_KEY_TEMPLATES)
    return template.format(
        mood=random.choice(MOODS),
        tempo=random.choice(TEMPOS),
        style=random.choice(STYLES),
        texture=random.choice(TEXTURES),
    )

# ============================================================================
# KEY DETECTION (Krumhansl-Kessler)
# ============================================================================
def best_match_with_key(chroma_vec, profile):
    scores = [np.corrcoef(chroma_vec, np.roll(profile, i))[0, 1]
              for i in range(12)]
    best_idx = int(np.argmax(scores))
    return best_idx, float(scores[best_idx])

def detect_key_from_midi(midi_path: Path) -> dict:
    """Detect key from MIDI using KK algorithm."""
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        pc = np.zeros(12)
        for inst in midi.instruments:
            for note in inst.notes:
                duration = note.end - note.start
                pc[note.pitch % 12] += duration

        if pc.sum() == 0:
            return {
                "key": None,
                "note": None,
                "mode": None,
                "confidence": 0.0,
                "error": "No notes",
            }

        pc_norm = pc / pc.sum()

        major_idx, major_score = best_match_with_key(pc_norm, KK_MAJOR)
        minor_idx, minor_score = best_match_with_key(pc_norm, KK_MINOR)

        if major_score > minor_score:
            key_note = NOTE_NAMES[major_idx]
            key_mode = "major"
            confidence = major_score - minor_score
        else:
            key_note = NOTE_NAMES[minor_idx]
            key_mode = "minor"
            confidence = minor_score - major_score

        return {
            "key": f"{key_note}_{key_mode}",
            "note": key_note,
            "mode": key_mode,
            "confidence": float(confidence),
            "error": None,
        }
    except Exception as e:
        return {
            "key": None,
            "note": None,
            "mode": None,
            "confidence": 0.0,
            "error": str(e),
        }

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

# We'll also set clip_id = current total so filenames are consistent:
def get_starting_clip_id() -> int:
    # If metadata files exist, use their count as starting ID
    if METADATA_DIR.exists():
        return len(list(METADATA_DIR.glob("*.json")))
    return 0

# ============================================================================
# MERGE METADATA
# ============================================================================
def merge_metadata():
    """Merge individual metadata JSONs and print detected-key distribution."""
    all_metadata = []
    key_counts = {}

    for meta_file in sorted(METADATA_DIR.glob("*.json")):
        with open(meta_file) as f:
            m = json.load(f)
        all_metadata.append(m)
        k = m.get("label_key")
        if k is not None:
            key_counts[k] = key_counts.get(k, 0) + 1

    combined_path = OUTPUT_DIR / "dataset_metadata.json"
    with open(combined_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Merged {len(all_metadata)} samples into {combined_path}")
    print("\nDetected-key distribution (label_key):")
    for k, v in sorted(key_counts.items(), key=lambda x: x[0]):
        print(f"  {k:12s}: {v}")

# ============================================================================
# MAIN (single GPU loop)
# ============================================================================
def main():
    # Make dirs
    for d in [OUTPUT_DIR, AUDIO_DIR, MIDI_DIR, ACTIVATIONS_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if device != "cuda":
        print("⚠️ CUDA not available; this will be very slow on CPU.")

    # Resume info
    total_so_far = load_progress()
    clip_id = get_starting_clip_id()
    print(f"Resuming from total_so_far={total_so_far}, next clip_id={clip_id}")

    if total_so_far >= TARGET_TOTAL_SAMPLES:
        print("Already reached TARGET_TOTAL_SAMPLES, nothing to do.")
        merge_metadata()
        return

    # Load model + processor
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large"
    ).to(device)
    model.eval()
    sampling_rate = model.config.audio_encoder.sampling_rate

    # Activation hooks
    activations = {}
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out_t = out[0]
            else:
                out_t = out
            activations.setdefault(name, []).append(out_t.detach().cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == "MusicgenDecoderLayer":
            handles.append(module.register_forward_hook(make_hook(name)))

    bp_model_path = str(ICASSP_2022_MODEL_PATH)

    start_time = time.time()
    local_count = 0

    while total_so_far < TARGET_TOTAL_SAMPLES:
        clip_id_str = f"{clip_id:04d}"
        prompt = generate_prompt_no_key()

        # Generate audio
        activations.clear()
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=256,
            )

        clip = audio_values[0, 0].cpu().numpy()

        # Save audio
        wav_path = AUDIO_DIR / f"{clip_id_str}.wav"
        wavfile.write(str(wav_path), rate=sampling_rate, data=clip)

        # BasicPitch → MIDI
        predict_and_save(
            [str(wav_path)],
            str(MIDI_DIR),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=bp_model_path,
        )

        midi_path = MIDI_DIR / f"{clip_id_str}_basic_pitch.mid"
        if not midi_path.exists():
            alt = MIDI_DIR / f"{wav_path.stem}_basic_pitch.mid"
            if alt.exists():
                midi_path = alt

        key_info = detect_key_from_midi(midi_path)
        detected_key = key_info["key"]

        if detected_key is None:
            print(f"[{clip_id_str}] key detection failed ({key_info['error']}), keeping sample but label_key=None")
        else:
            print(f"[{clip_id_str}] detected={detected_key} "
                  f"(sample {total_so_far+1}/{TARGET_TOTAL_SAMPLES})")

        # Save activations
        act_path = ACTIVATIONS_DIR / f"{clip_id_str}.pt"
        torch.save(dict(activations), str(act_path))

        # Save metadata – note NO prompted_key field
        metadata = {
            "clip_id": clip_id_str,
            "prompt": prompt,
            "label_key": detected_key,
            "detected_key": detected_key,
            "key_info": key_info,
            "audio_path": str(wav_path),
            "midi_path": str(midi_path),
            "activations_path": str(act_path),
            "sampling_rate": sampling_rate,
        }

        meta_path = METADATA_DIR / f"{clip_id_str}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        clip_id += 1
        total_so_far += 1
        local_count += 1

        if local_count % CHECKPOINT_EVERY == 0:
            save_progress(total_so_far)

    # Cleanup
    for h in handles:
        h.remove()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("NO-KEY DATASET GENERATION COMPLETE (single GPU)")
    print("=" * 60)
    print(f"Total samples: {total_so_far} (target {TARGET_TOTAL_SAMPLES})")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"Samples per minute: {total_so_far / (elapsed/60):.1f}")

    save_progress(total_so_far)
    merge_metadata()
    print(f"\nDataset (no-key prompts) saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
