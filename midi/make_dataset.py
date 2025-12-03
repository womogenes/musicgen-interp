import os
from pathlib import Path

import torch
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

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

AUDIO_DIR = Path("data/audio")
MIDI_DIR = Path("data/midi")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MIDI_DIR.mkdir(parents=True, exist_ok=True)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small"
).to(device)
model.eval()

sampling_rate = model.config.audio_encoder.sampling_rate  # usually 32000
print("Sampling rate:", sampling_rate)

BP_MODEL_PATH = str(ICASSP_2022_MODEL_PATH)

def main():
    for i, prompt in enumerate(PROMPTS):
        clip_id = f"{i:03d}"
        print(f"\n=== {clip_id}: '{prompt}' ===")

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

    print("\nDone generating dataset.")

if __name__ == "__main__":
    main()
