import torch
import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from numpy.lib.format import open_memmap

N_LAYERS = 48
LAYERS = [f"decoder.model.decoder.layers.{n}" for n in range(N_LAYERS)]
METADATA_PATH = "/home/harinit9/orcd/pool/musicgen-data-nokey/dataset_metadata.json"
ACTS_BY_LAYER_PATH = Path("/home/wyf/orcd/pool/musicgen-activations-nokey/acts_by_layer")

os.makedirs(ACTS_BY_LAYER_PATH, exist_ok=True)

with open(METADATA_PATH) as fin:
    metadata = json.load(fin)

N_CLIPS = len(metadata)

def get_clip_np(clip: dict):
    acts = torch.load(clip["activations_path"])
    layer_acts = [torch.cat(acts[layer], dim=1).mean(axis=0).numpy() for layer in LAYERS]
    return np.stack(layer_acts)

first_clip = get_clip_np(metadata[0])
_, T, D = first_clip.shape

layer_mm = [
    open_memmap(
        ACTS_BY_LAYER_PATH / f"layer_{layer_idx:02d}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(N_CLIPS, T, D)
    )
    for layer_idx in range(N_LAYERS)
]

for i, clip in tqdm(enumerate(metadata), total=N_CLIPS):
    clip_acts = get_clip_np(clip).astype(np.float32)
    for layer_idx in range(N_LAYERS):
        layer_mm[layer_idx][i] = clip_acts[layer_idx]

for mm in layer_mm:
    mm.flush()
