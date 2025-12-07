"""
single gpu for missing keys
"""

import os
import json
import random
from pathlib import Path
import time
import sys

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
    TARGET_PER_KEY = 2      # small for debugging
    print("🧪 TEST MODE: only 2 samples per missing key")
else:
    TARGET_PER_KEY = 55     # how many NEW samples per missing key

OUTPUT_DIR = Path("/home/harinit9/orcd/pool/musicgen-data/")
AUDIO_DIR = OUTPUT_DIR / "audio"
MIDI_DIR = OUTPUT_DIR / "midi"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
METADATA_DIR = OUTPUT_DIR / "metadata"

for d in [OUTPUT_DIR, AUDIO_DIR, MIDI_DIR, ACTIVATIONS_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# KEY DEFINITIONS
# ============================================================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KEY_NAMES = [f"{n}_major" for n in NOTE_NAMES] + [f"{n}_minor" for n in NOTE_NAMES]

# Krumhansl-Kessler profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ============================================================================
# PROMPT GENERATION
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

TEMPLATES = [
    "A {tempo}, {mood} solo piano piece in {key}, {style}, featuring {texture}.",
    "Solo piano in {key} at {tempo}, {mood} overall, with {texture}.",
    "{mood} piano in {key}, {tempo}, using {style} and {texture}.",
    "Piano piece in {key} with {tempo} pacing, {mood} character, {texture}.",
]

def generate_prompt(target_key: str) -> str:
    """Generate a random prompt hinting at target_key for diversity."""
    key_prompt = target_key.replace("_", " ")
    template = random.choice(TEMPLATES)
    return template.format(
        key=key_prompt,
        mood=random.choice(MOODS),
        tempo=random.choice(TEMPOS),
        style=random.choice(STYLES),
        texture=random.choice(TEXTURES),
    )

# ============================================================================
# KEY DETECTION (Krumhansl-Kessler, for metadata only)
# ============================================================================
def best_match_with_key(chroma_vec, profile):
    scores = [np.corrcoef(chroma_vec, np.roll(profile, i))[0, 1]
              for i in range(12)]
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]

def detect_key_from_midi(midi_path: Path) -> dict:
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        pc = np.zeros(12)
        for inst in midi.instruments:
            for note in inst.notes:
                duration = note.end - note.start
                pc[note.pitch % 12] += duration

        if pc.sum() == 0:
            return {"key": None, "note": None, "mode": None,
                    "confidence": 0.0, "error": "No notes"}

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
        return {"key": None, "note": None, "mode": None,
                "confidence": 0.0, "error": str(e)}

# ============================================================================
# MAIN
# ============================================================================
def main():
    # ----------------------------------------------------------------------
    # 1. Figure out which keys are missing based on existing metadata
    # ----------------------------------------------------------------------
    meta_path = OUTPUT_DIR / "dataset_metadata.json"
    present_keys = set()

    if meta_path.exists():
        all_meta = json.loads(meta_path.read_text())
        for m in all_meta:
            pk = m.get("prompted_key")
            if pk is not None:
                present_keys.add(pk)
        print(f"Found existing metadata for {len(all_meta)} samples.")
        print("Present prompted keys:", sorted(present_keys))

        # compute max existing clip_id
        existing_ids = []
        for m in all_meta:
            cid = m.get("clip_id")
            if cid is not None:
                try:
                    existing_ids.append(int(cid))
                except ValueError:
                    pass

        if existing_ids:
            current_id = max(existing_ids) + 1
        else:
            current_id = 0
    else:
        all_meta = []
        print("No existing dataset_metadata.json found; assuming fresh run.")
        current_id = 0

    # which keys are missing?
    missing_keys = [k for k in KEY_NAMES if k not in present_keys]
    if not missing_keys:
        print("No missing keys! All 24 already present in prompted_key.")
        return

    print("\nMissing keys to generate:")
    for k in missing_keys:
        print("  ", k)
    print()

    # extra safety: skip any clip_id that already has files on disk
    while (AUDIO_DIR / f"{current_id:04d}.wav").exists() or \
          (METADATA_DIR / f"{current_id:04d}.json").exists() or \
          (ACTIVATIONS_DIR / f"{current_id:04d}.pt").exists():
        current_id += 1

    print(f"Starting from clip_id {current_id:04d}")

    # ----------------------------------------------------------------------
    # 2. Load model + processor on a single GPU
    # ----------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large"
    ).to(device)
    model.eval()
    sampling_rate = model.config.audio_encoder.sampling_rate

    bp_model_path = str(ICASSP_2022_MODEL_PATH)

    # Set up activation hooks
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

    # ----------------------------------------------------------------------
    # 3. Generate TARGET_PER_KEY samples per missing key
    # ----------------------------------------------------------------------
    start_time = time.time()
    total_new = 0

    for target_key in missing_keys:
        print(f"\n=== Generating for missing key: {target_key} ===")
        for n in range(TARGET_PER_KEY):
            # skip over any id that somehow already has files
            while (AUDIO_DIR / f"{current_id:04d}.wav").exists() or \
                  (METADATA_DIR / f"{current_id:04d}.json").exists() or \
                  (ACTIVATIONS_DIR / f"{current_id:04d}.pt").exists():
                current_id += 1

            clip_id_str = f"{current_id:04d}"
            prompt = generate_prompt(target_key)
            print(f"[{target_key}] Sample {clip_id_str} (#{n+1}/{TARGET_PER_KEY})")

            # Clear activations
            activations.clear()

            # Prepare inputs
            inputs = processor(text=[prompt], padding=True,
                               return_tensors="pt").to(device)

            # Generate audio
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=256,
                )

            # Save audio
            clip = audio_values[0, 0].cpu().numpy()
            wav_path = AUDIO_DIR / f"{clip_id_str}.wav"
            wavfile.write(str(wav_path), rate=sampling_rate, data=clip)

            # Run BasicPitch
            predict_and_save(
                [str(wav_path)],
                str(MIDI_DIR),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=bp_model_path,
            )

            # MIDI path
            midi_path = MIDI_DIR / f"{clip_id_str}_basic_pitch.mid"
            if not midi_path.exists():
                # Fallback if BasicPitch used different naming
                alt = MIDI_DIR / f"{wav_path.stem}_basic_pitch.mid"
                if alt.exists():
                    midi_path = alt
                else:
                    print(f"  WARNING: MIDI not found for {clip_id_str}, continuing.")
                    detected_info = {
                        "key": None, "note": None, "mode": None,
                        "confidence": 0.0, "error": "no_midi_file"
                    }
            else:
                detected_info = detect_key_from_midi(midi_path)

            detected_key = detected_info.get("key")

            # Save activations
            act_path = ACTIVATIONS_DIR / f"{clip_id_str}.pt"
            torch.save(dict(activations), str(act_path))

            # Save metadata
            metadata = {
                "clip_id": clip_id_str,
                "prompt": prompt,
                "prompted_key": target_key,
                "detected_key": detected_key,
                "key_info": detected_info,
                "audio_path": str(wav_path),
                "midi_path": str(midi_path),
                "activations_path": str(act_path),
                "sampling_rate": sampling_rate,
            }

            meta_file = METADATA_DIR / f"{clip_id_str}.json"
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)

            all_meta.append(metadata)
            current_id += 1
            total_new += 1

    # ----------------------------------------------------------------------
    # 4. Clean up hooks and merge metadata
    # ----------------------------------------------------------------------
    for h in handles:
        h.remove()

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("MISSING-KEY TOP-UP COMPLETE")
    print("="*60)
    print(f"New samples generated: {total_new}")
    print(f"Total samples now: {len(all_meta)}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print("="*60)

    # Merge into dataset_metadata.json
    with open(OUTPUT_DIR / "dataset_metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"Merged metadata saved to {OUTPUT_DIR/'dataset_metadata.json'}")


if __name__ == "__main__":
    main()
