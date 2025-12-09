import torch
import os
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from numpy.lib.format import open_memmap

LAYER_IDX = 22
LAYER = f"decoder.model.decoder.layers.{LAYER_IDX}"
METADATA_PATH = "/home/harinit9/orcd/pool/musicgen-data-nokey/dataset_metadata.json"
ACTS_BY_LAYER_PATH = Path("/home/harinit9/orcd/pool/musicgen-activations-nokey/acts_by_layer")

os.makedirs(ACTS_BY_LAYER_PATH, exist_ok=True)

with open(METADATA_PATH) as fin:
    metadata = json.load(fin)

N_CLIPS = len(metadata)

def get_clip_np(clip: dict):
    acts = torch.load(clip["activations_path"])
    return torch.cat(acts[LAYER], dim=1).mean(axis=0).numpy()

first_clip = get_clip_np(metadata[0])
T, D = first_clip.shape

layer_mm = open_memmap(
    ACTS_BY_LAYER_PATH / f"layer_{LAYER_IDX:02d}.npy",
    mode="w+",
    dtype=np.float32,
    shape=(N_CLIPS, T, D)
)

for i, clip in tqdm(enumerate(metadata), total=N_CLIPS):
    layer_mm[i] = get_clip_np(clip).astype(np.float32)

layer_mm.flush()
