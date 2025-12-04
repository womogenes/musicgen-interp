# MusicGen interpretability

https://musicgen.com

## Setup

1. Install uv (https://docs.astral.sh/uv)
2. `uv sync`

## Reference

https://github.com/facebookresearch/audiocraft

## MIDI

`midi/make_dataset.py` generates audio files with basic prompts and puts them in `data/audio/*.wav`, `data/midi/*.midi`, and `data/activations/*.pt`.

## MusicGen Architecture

- EnCodec encoder: converts raw audio samples to "cookbook codec" (4 streams, each at 50 Hz with a dictionary of 2048 tokens).
- Embedding layer: converts tokens into vectors (standard)
- Transformer (decoder): turns tokens into next tokens (one for each of the 4 streams)
