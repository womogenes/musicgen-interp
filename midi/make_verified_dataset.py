"""
Multi-GPU dataset generation pipeline for MusicGen key classification.

Generates audio, verifies key with Krumhansl-Kessler, saves everything.
Uses detected key as ground truth label (no rejected samples!).
"""
import os
import json
import random
from pathlib import Path
from multiprocessing import Process, Manager
import time

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
import sys

# Check for test mode: python make_verified_dataset.py --test
TEST_MODE = "--test" in sys.argv

if TEST_MODE:
    TARGET_PER_KEY = 1           # Just 1 sample per key for testing
    MAX_TOTAL_SAMPLES = 5        # Stop after 5 samples
    CHECKPOINT_EVERY = 1
    print("🧪 TEST MODE: Generating only a few samples")
else:
    TARGET_PER_KEY = 42          # Samples per key (42 × 24 = 1008 total)
    MAX_TOTAL_SAMPLES = 1200     # Stop after this many total (safety limit)
    CHECKPOINT_EVERY = 10        # Save progress every N samples per worker

# Output directories
OUTPUT_DIR = Path("/home/harinit9/orcd/pool/musicgen-data/")
AUDIO_DIR = OUTPUT_DIR / "audio"
MIDI_DIR = OUTPUT_DIR / "midi"
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
METADATA_DIR = OUTPUT_DIR / "metadata"

# ============================================================================
# KEY DEFINITIONS
# ============================================================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KEY_NAMES = [f"{n}_major" for n in NOTE_NAMES] + [f"{n}_minor" for n in NOTE_NAMES]
NUM_KEYS = len(KEY_NAMES)

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
    # Convert internal format (C_major) to prompt format (C major)
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
# KEY DETECTION (Krumhansl-Kessler)
# ============================================================================
def best_match_with_key(chroma_vec, profile):
    """Find best matching key root and score."""
    scores = [np.corrcoef(chroma_vec, np.roll(profile, i))[0, 1]
              for i in range(12)]
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]

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
            return {"key": None, "note": None, "mode": None, "confidence": 0.0, "error": "No notes"}
        
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
            "error": None
        }
    except Exception as e:
        return {"key": None, "note": None, "mode": None, "confidence": 0.0, "error": str(e)}

# ============================================================================
# WORKER FUNCTION (runs on each GPU)
# ============================================================================
def worker(gpu_id: int, key_subset: list, shared_counts: dict, shared_lock, worker_id: int):
    """Worker process that generates samples on a specific GPU."""
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda"
    
    print(f"[Worker {worker_id}] Starting on GPU {gpu_id}, handling keys: {key_subset[:3]}...{key_subset[-1]}")
    
    # Load model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large").to(device)
    model.eval()
    sampling_rate = model.config.audio_encoder.sampling_rate
    
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
    
    bp_model_path = str(ICASSP_2022_MODEL_PATH)
    
    # Local tracking
    local_count = 0
    
    while True:
        # Check if all keys in our subset are done
        with shared_lock:
            all_done = all(shared_counts.get(k, 0) >= TARGET_PER_KEY for k in key_subset)
            total = sum(shared_counts.values())
        
        if all_done or total >= MAX_TOTAL_SAMPLES:
            break
        
        # Pick a key that needs more samples (from our subset)
        with shared_lock:
            candidates = [k for k in key_subset if shared_counts.get(k, 0) < TARGET_PER_KEY]
        
        if not candidates:
            break
        
        target_key = random.choice(candidates)
        prompt = generate_prompt(target_key)
        
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
        
        # Get clip ID
        with shared_lock:
            clip_id = sum(shared_counts.values())
            clip_id_str = f"{clip_id:04d}"
        
        # Save audio
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
        
        # Detect key
        midi_path = MIDI_DIR / f"{clip_id_str}_basic_pitch.mid"
        key_info = detect_key_from_midi(midi_path)
        
        if key_info["key"] is None:
            print(f"[Worker {worker_id}] Sample {clip_id_str}: Key detection failed, skipping")
            wav_path.unlink(missing_ok=True)
            midi_path.unlink(missing_ok=True)
            continue
        
        detected_key = key_info["key"]
        
        # Save activations
        act_path = ACTIVATIONS_DIR / f"{clip_id_str}.pt"
        torch.save(dict(activations), str(act_path))
        
        # Save metadata
        metadata = {
            "clip_id": clip_id_str,
            "prompt": prompt,
            "prompted_key": target_key,
            "detected_key": detected_key,
            "key_info": key_info,
            "audio_path": str(wav_path),
            "midi_path": str(midi_path),
            "activations_path": str(act_path),
            "sampling_rate": sampling_rate,
        }
        
        meta_path = METADATA_DIR / f"{clip_id_str}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update counts (use DETECTED key, not prompted key!)
        with shared_lock:
            shared_counts[detected_key] = shared_counts.get(detected_key, 0) + 1
            current_count = shared_counts[detected_key]
            total = sum(shared_counts.values())
        
        local_count += 1
        
        print(f"[Worker {worker_id}] Sample {clip_id_str}: prompted={target_key}, detected={detected_key} "
              f"({current_count}/{TARGET_PER_KEY}), total={total}")
        
        # Checkpoint
        if local_count % CHECKPOINT_EVERY == 0:
            with shared_lock:
                save_progress(dict(shared_counts))
    
    # Cleanup
    for h in handles:
        h.remove()
    
    print(f"[Worker {worker_id}] Done! Generated {local_count} samples.")

# ============================================================================
# PROGRESS TRACKING
# ============================================================================
def save_progress(counts: dict):
    """Save current progress."""
    progress_path = OUTPUT_DIR / "progress.json"
    with open(progress_path, 'w') as f:
        json.dump(counts, f, indent=2)

def load_progress() -> dict:
    """Load existing progress if resuming."""
    progress_path = OUTPUT_DIR / "progress.json"
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {k: 0 for k in KEY_NAMES}

def merge_metadata():
    """Merge all individual metadata files into one."""
    all_metadata = []
    for meta_file in sorted(METADATA_DIR.glob("*.json")):
        with open(meta_file) as f:
            all_metadata.append(json.load(f))
    
    combined_path = OUTPUT_DIR / "dataset_metadata.json"
    with open(combined_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"Merged {len(all_metadata)} samples into {combined_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Create directories
    for d in [OUTPUT_DIR, AUDIO_DIR, MIDI_DIR, ACTIVATIONS_DIR, METADATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    if num_gpus == 0:
        print("No GPUs found! Exiting.")
        return
    
    # Load existing progress
    initial_counts = load_progress()
    total_existing = sum(initial_counts.values())
    if total_existing > 0:
        print(f"Resuming from {total_existing} existing samples")
    
    # Split keys across GPUs
    keys_per_gpu = [[] for _ in range(num_gpus)]
    for i, key in enumerate(KEY_NAMES):
        keys_per_gpu[i % num_gpus].append(key)
    
    print(f"Key distribution across GPUs:")
    for i, keys in enumerate(keys_per_gpu):
        print(f"  GPU {i}: {len(keys)} keys")
    
    # Shared state
    manager = Manager()
    shared_counts = manager.dict(initial_counts)
    shared_lock = manager.Lock()
    
    # Start workers
    start_time = time.time()
    processes = []
    
    for gpu_id in range(num_gpus):
        p = Process(
            target=worker,
            args=(gpu_id, keys_per_gpu[gpu_id], shared_counts, shared_lock, gpu_id)
        )
        p.start()
        processes.append(p)
    
    # Wait for all workers
    for p in processes:
        p.join()
    
    elapsed = time.time() - start_time
    
    # Final summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    
    final_counts = dict(shared_counts)
    total = sum(final_counts.values())
    
    print(f"Total samples: {total}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Samples per minute: {total / (elapsed/60):.1f}")
    
    print("\nSamples per key:")
    for key in KEY_NAMES:
        count = final_counts.get(key, 0)
        bar = "█" * (count // 2) + "░" * ((TARGET_PER_KEY - count) // 2)
        print(f"  {key:12s}: {count:3d}/{TARGET_PER_KEY} {bar}")
    
    # Save final progress
    save_progress(final_counts)
    
    # Merge metadata
    merge_metadata()
    
    print(f"\nDataset saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

