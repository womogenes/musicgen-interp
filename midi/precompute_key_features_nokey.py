"""
Precompute pooled features from raw activations for key probing.
Saves one cache file per layer in feature_cache_nokey/.
"""

import json
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
METADATA_PATH = "/home/harinit9/orcd/pool/musicgen-data-nokey/dataset_metadata.json"
FEATURE_CACHE_DIR = Path("feature_cache_nokey_0.2")
MIN_CONFIDENCE = 0.2  # 0.0 = keep all; >0 = filter by detected key confidence

# ----------------------------------------------------------------------
# Label definitions
# ----------------------------------------------------------------------
KEY_NAMES = [
    'C_major', 'C#_major', 'D_major', 'D#_major', 'E_major', 'F_major',
    'F#_major', 'G_major', 'G#_major', 'A_major', 'A#_major', 'B_major',
    'C_minor', 'C#_minor', 'D_minor', 'D#_minor', 'E_minor', 'F_minor',
    'F#_minor', 'G_minor', 'G#_minor', 'A_minor', 'A#_minor', 'B_minor',
]
KEY_TO_IDX = {key: idx for idx, key in enumerate(KEY_NAMES)}

MODE_NAMES = ['major', 'minor']
MODE_TO_IDX = {'major': 0, 'minor': 1}


def get_layer_num(name):
    parts = name.split(".")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0


def get_all_layer_names(metadata):
    """Discover all layer names from first activation file."""
    act_path = None
    for m in metadata:
        if "activations_path" in m:
            act_path = m["activations_path"]
            break
    if act_path is None:
        raise ValueError("No activations_path in metadata!")

    acts = torch.load(act_path, map_location="cpu")
    layer_names = sorted(acts.keys(), key=get_layer_num)
    return layer_names


def select_layers(all_layer_names):
    """Select every 4th layer by index, plus the last layer."""
    indexed_layers = sorted(
        [(get_layer_num(n), n) for n in all_layer_names],
        key=lambda x: x[0],
    )
    layer_indices = [idx for idx, _ in indexed_layers]
    max_idx = max(layer_indices)

    selected_layers = []
    seen = set()
    for idx, name in indexed_layers:
        if idx % 4 == 0:
            selected_layers.append(name)
            seen.add(name)

    # ensure we also include the last layer
    last_layer_name = [n for i, n in indexed_layers if i == max_idx][0]
    if last_layer_name not in seen:
        selected_layers.append(last_layer_name)

    return selected_layers


def pool_activations(layer_act_list):
    """
    4-chunk pooling: concat, reshape to [T, D], mean-pool each chunk, flatten.
    Falls back to global mean if T < 4.
    """
    NUM_CHUNKS = 4
    layer_act = torch.cat(layer_act_list, dim=0)  # [T, D]
    hidden_dim = layer_act.shape[-1]
    layer_act_flat = layer_act.reshape(-1, hidden_dim)  # [T, D]

    T = layer_act_flat.shape[0]
    if T < NUM_CHUNKS:
        # fallback: not enough tokens, just global mean
        pooled = layer_act_flat.mean(dim=0)  # [D]
    else:
        chunk_size = T // NUM_CHUNKS
        chunk_means = []
        for c in range(NUM_CHUNKS):
            start = c * chunk_size
            end = T if c == NUM_CHUNKS - 1 else (c + 1) * chunk_size
            chunk = layer_act_flat[start:end]  # [chunk_T, D]
            chunk_means.append(chunk.mean(dim=0))  # [D]

        chunk_means = torch.stack(chunk_means, dim=0)  # [NUM_CHUNKS, D]
        pooled = chunk_means.flatten()  # [NUM_CHUNKS * D]

    return pooled


def main():
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {METADATA_PATH}")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Filter to entries with valid detected_key
    metadata = [m for m in metadata if m.get("detected_key") in KEY_TO_IDX]
    print(f"Found {len(metadata)} samples with valid detected_key")

    # Optional confidence filter
    if MIN_CONFIDENCE > 0.0:
        before = len(metadata)
        metadata = [
            m for m in metadata
            if m.get("key_info", {}).get("confidence", 0.0) >= MIN_CONFIDENCE
        ]
        print(f"Applied confidence filter >= {MIN_CONFIDENCE}, kept {len(metadata)}/{before}")

    # Discover layers
    all_layer_names = get_all_layer_names(metadata)
    print(f"\nAll layers ({len(all_layer_names)}):")
    for name in all_layer_names:
        print(f"  {name}")

    selected_layers = select_layers(all_layer_names)
    print(f"\nSelected layers for caching ({len(selected_layers)}):")
    for name in selected_layers:
        print(f"  {name}")

    # Initialize per-layer storage
    layer_to_features = {layer_name: [] for layer_name in selected_layers}
    layer_to_labels_key24 = {layer_name: [] for layer_name in selected_layers}
    layer_to_labels_mode2 = {layer_name: [] for layer_name in selected_layers}
    layer_to_clip_ids = {layer_name: [] for layer_name in selected_layers}

    # Process each sample once, updating all layer caches
    print(f"\nProcessing {len(metadata)} samples...")
    for i, meta in enumerate(metadata):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Loading {i+1}/{len(metadata)}...")

        act_path = meta["activations_path"]
        activations = torch.load(act_path, map_location="cpu")

        # Compute labels once per sample
        detected_key = meta["detected_key"]
        key_idx = KEY_TO_IDX[detected_key]
        mode_str = "major" if detected_key.endswith("_major") else "minor"
        mode_idx = MODE_TO_IDX[mode_str]
        clip_id = meta.get("clip_id", str(i))

        # Update each layer's cache
        for layer_name in selected_layers:
            if layer_name not in activations:
                continue

            layer_act_list = activations[layer_name]
            pooled = pool_activations(layer_act_list)

            layer_to_features[layer_name].append(pooled)
            layer_to_labels_key24[layer_name].append(key_idx)
            layer_to_labels_mode2[layer_name].append(mode_idx)
            layer_to_clip_ids[layer_name].append(clip_id)

    # Save cache files for each layer
    print(f"\nSaving cache files...")
    for layer_name in selected_layers:
        features = layer_to_features[layer_name]
        if len(features) == 0:
            print(f"  WARNING: No features for {layer_name}, skipping")
            continue

        features_tensor = torch.stack(features)
        labels_key24_tensor = torch.tensor(layer_to_labels_key24[layer_name], dtype=torch.long)
        labels_mode2_tensor = torch.tensor(layer_to_labels_mode2[layer_name], dtype=torch.long)

        safe_name = layer_name.replace(".", "_")
        save_path = FEATURE_CACHE_DIR / f"{safe_name}.pt"

        torch.save({
            "layer_name": layer_name,
            "features": features_tensor,
            "labels_key24": labels_key24_tensor,
            "labels_mode2": labels_mode2_tensor,
            "clip_ids": layer_to_clip_ids[layer_name],
            "key_names": KEY_NAMES,
            "mode_names": MODE_NAMES,
            "min_confidence": MIN_CONFIDENCE,
        }, save_path)

        print(f"  Saved {save_path}")
        print(f"    Features shape: {features_tensor.shape}")
        print(f"    Samples: {len(features)}")

    # Summary
    first_layer = selected_layers[0]
    num_samples_used = len(layer_to_features[first_layer])
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total metadata entries: {len(metadata)}")
    print(f"  Samples actually cached: {num_samples_used}")
    print(f"  Layers cached: {len(selected_layers)}")
    print(f"  Cache directory: {FEATURE_CACHE_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
