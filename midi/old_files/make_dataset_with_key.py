import os
from pathlib import Path
import json

import numpy as np
import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

PROMPTS = [
    "solo piano arpeggios",
    "slow romantic piano ballad",
    "fast piano arpeggios in the right hand",
    "jazzy piano chords with walking bass",
    "simple left hand piano accompaniment with melody on top",
    "minimalist piano ostinato repeating softly",
    "piano waltz with flowing arpeggios",
    "sad piano melody in a minor key",
    "bright uplifting piano theme",
    "piano etude with fast right hand runs",
    "piano chords with syncopated rhythms",
    "gentle lullaby on solo piano",
    "dramatic piano with big low octave hits",
    "piano melody with broken chords in the left hand",
    "jazz piano comping behind a solo",
    "bluesy piano riff with swing feel",
    "simple triad chords on piano, slow tempo",
    "dense piano chords with pedal sustain",
    "staccato piano notes in high register",
    "piano accompaniment pattern like pop ballad",
    "piano playing repeated block chords",
    "piano arpeggios spanning multiple octaves",
    "melancholic piano motif looping",
    "cinematic piano intro with sparse notes",
    "piano ostinato in the middle register",
    "left hand piano bass line with right hand chords",
    "piano playing eighth-note broken chords",
    "piano improvisation over a simple chord progression",
    "piano chords with occasional melodic fills",
    "piano playing slow, wide intervals",
    "soft piano in high register with lots of reverb",
    "aggressive piano with accented chord stabs",
    "piano pattern with repeating sixteenth notes",
    "piano riff emphasizing syncopation",
    "piano melody doubled in octaves",
    "simple piano scales ascending and descending",
    "piano playing block chords in root position",
    "piano arpeggios outlining jazz chords",
    "piano accompaniment in 3/4 time",
    "piano accompaniment in 6/8 time",
    "slow piano chords with long silences",
    "piano melody with broken chord accompaniment in left hand",
    "high-register piano trills and ornaments",
    "low-register piano rumble with octave doublings",
    "piano part with alternating hands in call and response",
    "piano playing repeated two-note intervals",
    "piano comping like a bossa nova tune",
    "piano chords with clustered voicings",
    "simple diatonic piano melody with stepwise motion",
    "piano piece starting soft and gradually getting louder",
]

AUDIO_DIR = Path("data-large/audio")
MIDI_DIR = Path("data-large/midi")
ACTIVATIONS_DIR = Path("data-large/activations")
METADATA_DIR = Path("data-large/metadata")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MIDI_DIR.mkdir(parents=True, exist_ok=True)
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-large"
).to(device)
model.eval()

sampling_rate = model.config.audio_encoder.sampling_rate  # usually 32000
print("Sampling rate:", sampling_rate)

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

BP_MODEL_PATH = str(ICASSP_2022_MODEL_PATH)

# Key detection setup - Krumhansl-Kessler profiles
kk_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
kk_minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def best_match_with_key(chroma_vec, profile):
    """Find best matching key and return (key_idx, score, all_scores)"""
    scores = [np.corrcoef(chroma_vec, np.roll(profile, i))[0, 1]
              for i in range(12)]
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx], scores

def detect_key_from_midi(midi_path):
    """
    Detect key from MIDI using pitch class distribution with KK profiles.
    Returns dict with key label and confidence.
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Build pitch class histogram weighted by duration
        pc = np.zeros(12)
        for inst in midi.instruments:
            for note in inst.notes:
                duration = note.end - note.start
                pc[note.pitch % 12] += duration
        
        # Check if we got any notes
        if pc.sum() == 0:
            return {
                'key': None,
                'note': None,
                'mode': None,
                'confidence': 0.0,
                'error': 'No notes found'
            }
        
        # Normalize pitch class distribution
        pc_norm = pc / pc.sum()
        
        # Find best major and minor keys
        major_key_idx, major_score, _ = best_match_with_key(pc_norm, kk_major)
        minor_key_idx, minor_score, _ = best_match_with_key(pc_norm, kk_minor)
        
        # Determine final key
        if major_score > minor_score:
            key_note = NOTE_NAMES[major_key_idx]
            key_mode = "major"
            confidence = major_score - minor_score
        else:
            key_note = NOTE_NAMES[minor_key_idx]
            key_mode = "minor"
            confidence = minor_score - major_score
        
        return {
            'key': f"{key_note}_{key_mode}",  # e.g., "C_major" for use as label
            'note': key_note,
            'mode': key_mode,
            'confidence': float(confidence),  # higher = more certain
            'error': None
        }
    
    except Exception as e:
        return {
            'key': None,
            'note': None,
            'mode': None,
            'confidence': 0.0,
            'error': str(e)
        }

def main():
    all_metadata = []
    
    for i, prompt in enumerate(PROMPTS):
        clip_id = f"{i:03d}"
        print(f"\n=== {clip_id}: '{prompt}' ===")

        activations.clear()

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=3.0,
                max_new_tokens=256,  # controls length
            )

        clip = audio_values[0, 0].cpu().numpy()

        wav_path = AUDIO_DIR / f"{clip_id}.wav"
        wavfile.write(str(wav_path), rate=sampling_rate, data=clip)
        print(f"Saved audio to {wav_path}")

        predict_and_save(
            [str(wav_path)],
            str(MIDI_DIR),
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=BP_MODEL_PATH,
        )
        print(f"Ran BasicPitch; check {MIDI_DIR} for MIDI files")

        # Detect key from generated MIDI
        midi_path = MIDI_DIR / f"{clip_id}_basic_pitch.mid"
        key_info = detect_key_from_midi(midi_path)
        
        if key_info['key']:
            print(f"Detected key: {key_info['note']} {key_info['mode']} "
                  f"(confidence: {key_info['confidence']:.3f})")
        else:
            print(f"Key detection failed: {key_info['error']}")

        # Save activations
        act_path = ACTIVATIONS_DIR / f"{clip_id}.pt"
        torch.save(activations, str(act_path))
        print(f"Saved activations to {act_path}")

        # Build metadata entry
        metadata = {
            'clip_id': clip_id,
            'prompt': prompt,
            'audio_path': str(wav_path),
            'midi_path': str(midi_path),
            'activations_path': str(act_path),
            'key_info': key_info,
            'sampling_rate': sampling_rate,
        }
        all_metadata.append(metadata)
        
        # Save individual metadata file
        metadata_path = METADATA_DIR / f"{clip_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    # Save combined metadata file
    combined_metadata_path = Path("data-large/dataset_metadata.json")
    with open(combined_metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    print(f"\n=== Saved combined metadata to {combined_metadata_path} ===")

    # Print summary statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Total clips: {len(all_metadata)}")
    
    # Key distribution
    key_counts = {}
    failed = 0
    for m in all_metadata:
        key = m['key_info']['key']
        if key:
            key_counts[key] = key_counts.get(key, 0) + 1
        else:
            failed += 1
    
    print(f"\nKey distribution:")
    for key in sorted(key_counts.keys()):
        print(f"  {key}: {key_counts[key]}")
    if failed > 0:
        print(f"  Failed detections: {failed}")
    
    # Confidence statistics
    confidences = [m['key_info']['confidence'] for m in all_metadata if m['key_info']['key']]
    if confidences:
        print(f"\nKey detection confidence:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        print(f"  Min: {np.min(confidences):.3f}")
        print(f"  Max: {np.max(confidences):.3f}")

    for h in handles:
        h.remove()

    print("\nDone generating dataset with key labels.")

if __name__ == "__main__":
    main()

