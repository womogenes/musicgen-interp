# MusicGen interpretability

https://musicgen.com

## Setup

1. Install uv (https://docs.astral.sh/uv)
2. `uv sync`

## Reference

https://github.com/facebookresearch/audiocraft

## MIDI

`midi/make_dataset.py` generates audio files with basic prompts and puts them in `data/audio/*.wav` and `data/midi/*.midi`.

## TODO

Along with generating MIDI data, we must save activations (see source in `main2.py`).
