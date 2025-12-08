"""
Multi-GPU dataset generation pipeline for DiffRhythm key classification.

Generates audio with DiffRhythm, verifies key with Krumhansl-Kessler, saves everything.
Uses detected key as ground truth label (no rejected samples unless key detection fails!).
"""

import os
import json
import random
from pathlib import Path
from multiprocessing import Process, Manager
import time
import sys

import numpy as np
import torch
import scipy.io.wavfile as wavfile
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi

# DiffRhythm imports (assumes DiffRhythm repo / package is installed & on PYTHONPATH)
from diffrhythm.infer.infer_utils import (
    get_reference_latent,
    get_lrc_token,
    get_audio_style_prompt,   # imported for completeness (unused here)
    get_text_style_prompt,
    prepare_model,
    get_negative_style_prompt,
)
from diffrhythm.infer.infer import inference

# ============================================================================
# CONFIGURATION
# ============================================================================
# Check for test mode: python make_verified_dataset_diffrhythm.py --test
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

# DiffRhythm duration setup
# 2048 frames ≈ 95 seconds for 44.1 kHz audio; you can change to 6144 / 285s if desired.
MAX_FRAMES = 2048
MUSIC_DURATION_SECONDS = 95

# Output directories
OUTPUT_DIR = Path("/home/harinit9/orcd/pool/musicgen-data/")  # reuse same path if you like
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
    """Generate a random style prompt hinting at target_key for diversity."""
    key_prompt = target_key.replace("_", " ")  # "C_major" -> "C major"
    template = random.choice(TEMPLATES)
    return template.format(
        key=key_prompt,
        mood=random.choice(MOODS),
        tempo=random.choice(TEMPOS),
        style=random.choice(STYLES),
        texture=random.choice(TEXTURES),
    )

def make_dummy_lyrics() -> str:
    """
    Create a minimal LRC string. DiffRhythm wants [mm:ss.xx] timestamps.
    We don't care about lyrics content for key classification; this just
    satisfies the interface and gives a simple timing structure.
    """
    return (
        "[00:00.00] Instrumental piano intro\n"
        "[00:10.00] Piano continues without vocals\n"
        "[00:30.00] Development section\n"
        "[01:00.00] Climax with fuller chords\n"
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
            key_no_
