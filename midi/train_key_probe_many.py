"""
Layer sweep: train linear probes (major/minor + 24-key)
for every 4th MusicGen decoder layer, with separate output dirs.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
NUM_KEYS = len(KEY_NAMES)

MODE_NAMES = ['major', 'minor']
MODE_TO_IDX = {'major': 0, 'minor': 1}


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class ActivationDataset(Dataset):
    """
    Dataset of activations with key labels (PROMPTED KEY) - preloads everything into memory.

    Labels:
      - If major_minor_only=True:   0 = major, 1 = minor based on prompted_key
      - If major_minor_only=False:  24-way classification over KEY_NAMES based on prompted_key

    confidence_threshold:
      - If >0, we *optionally* filter using detected key confidence in key_info,
        but labels are still based on prompted_key.
    """
    def __init__(
        self,
        metadata_path,
        layer_name=None,
        confidence_threshold=0.0,
        major_minor_only=False,
    ):
        self.major_minor_only = major_minor_only
        self.num_classes = 2 if major_minor_only else NUM_KEYS
        self.class_names = MODE_NAMES if major_minor_only else KEY_NAMES

        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # keep only samples with a valid prompted_key
        self.metadata = [
            m for m in self.metadata
            if m.get("prompted_key") in KEY_TO_IDX
        ]

        # Optional extra filter by detected key confidence (if present)
        if confidence_threshold > 0.0:
            before = len(self.metadata)
            filtered = []
            for m in self.metadata:
                ki = m.get("key_info", {})
                conf = ki.get("confidence", 0.0)
                if conf >= confidence_threshold:
                    filtered.append(m)
            self.metadata = filtered
            print(
                f"Applied confidence filter >= {confidence_threshold}, "
                f"kept {len(self.metadata)}/{before} samples"
            )

        task_name = "Major/Minor" if major_minor_only else "24-Key"
        print(f"Task: {task_name} classification ({self.num_classes} classes)")
        print(f"Found {len(self.metadata)} usable samples")

        # Load one activations file to discover available layers
        first_act = torch.load(self.metadata[0]["activations_path"], map_location="cpu")

        def get_layer_num(name):
            parts = name.split(".")
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            return 0

        self.available_layers = sorted(first_act.keys(), key=get_layer_num)

        if layer_name is None:
            self.layer_name = self.available_layers[-1]  # use last layer
            print(f"Auto-selected layer: {self.layer_name}")
        else:
            self.layer_name = layer_name
            print(f"Using layer: {self.layer_name}")

        print(f"Available layers: {len(self.available_layers)}")

        # PRELOAD all data into memory
        print("Preloading all activations into memory...")
        self.features = []
        self.labels = []

        for i, meta in enumerate(self.metadata):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Loading {i+1}/{len(self.metadata)}...")

            act_path = meta["activations_path"]
            activations = torch.load(act_path, map_location="cpu")

            if self.layer_name not in activations:
                # Skip if somehow missing
                continue

            layer_act_list = activations[self.layer_name]  # list of tensors

            # Concatenate along first dim
            layer_act = torch.cat(layer_act_list, dim=0)
            hidden_dim = layer_act.shape[-1]
            layer_act_flat = layer_act.reshape(-1, hidden_dim)
            pooled = layer_act_flat.mean(dim=0)  # [hidden_dim]

            self.features.append(pooled)

            # LABELS FROM PROMPTED KEY
            prompted_key = meta["prompted_key"]
            if self.major_minor_only:
                mode_str = "major" if prompted_key.endswith("_major") else "minor"
                self.labels.append(MODE_TO_IDX[mode_str])
            else:
                self.labels.append(KEY_TO_IDX[prompted_key])

        self.features = torch.stack(self.features)  # [N, hidden_dim]
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # [N]
        print(f"Preloaded! Feature shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ----------------------------------------------------------------------
# Model + training utils
# ----------------------------------------------------------------------
class LinearProbe(nn.Module):
    """Simple linear classifier"""
    def __init__(self, input_dim, num_classes=NUM_KEYS):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_probe(
    train_loader,
    val_loader,
    input_dim,
    num_classes=NUM_KEYS,
    num_epochs=50,
    lr=0.001,
):
    """Train a linear probe"""
    model = LinearProbe(input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "best_val_acc": best_val_acc,
    }


def evaluate_model(model, test_loader):
    """Evaluate model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, cm, all_preds, all_labels


def plot_training_curves(history, num_classes, save_path=None):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_losses"], label="Train Loss")
    ax1.plot(history["val_losses"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    random_baseline = 1 / num_classes
    ax2.plot(history["val_accs"], label="Val Accuracy")
    ax2.axhline(
        y=random_baseline,
        color="r",
        linestyle="--",
        label=f"Random ({random_baseline:.3f})",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    figsize = (6, 5) if len(class_names) <= 3 else (12, 10)
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    plt.close(fig)


# ----------------------------------------------------------------------
# Helper: get all layer names from first activation file
# ----------------------------------------------------------------------
def get_all_layer_names(metadata_path: str):
    with open(metadata_path) as f:
        meta = json.load(f)

    # find first sample with activations_path
    act_path = None
    for m in meta:
        if "activations_path" in m:
            act_path = m["activations_path"]
            break
    if act_path is None:
        raise ValueError("No activations_path in metadata!")

    acts = torch.load(act_path, map_location="cpu")

    def get_layer_num(name):
        parts = name.split(".")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0

    layer_names = sorted(acts.keys(), key=get_layer_num)
    return layer_names


# ----------------------------------------------------------------------
# One experiment (one layer + one task)
# ----------------------------------------------------------------------
def run_experiment(
    metadata_path: str,
    layer_name: str,
    major_minor_only: bool,
    output_dir: Path,
    num_epochs: int = 50,
    batch_size: int = 32,
    confidence_threshold: float = 0.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    task_name = "Major/Minor" if major_minor_only else "24-Key"
    print("\n" + "=" * 70)
    print(f"Running experiment: layer={layer_name} | task={task_name}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    dataset = ActivationDataset(
        metadata_path,
        layer_name=layer_name,
        confidence_threshold=confidence_threshold,
        major_minor_only=major_minor_only,
    )

    # Get input dimension from a sample
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]
    print(f"Input dimension: {input_dim}")

    # Split dataset
    TEST_SIZE = 0.2
    VAL_SIZE = 0.125  # of the remaining 0.8

    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=VAL_SIZE, random_state=42
    )

    print("\nDataset split:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    # DataLoaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    num_classes = dataset.num_classes
    class_names = dataset.class_names

    print(f"\nTraining for {num_epochs} epochs...")
    model, history = train_probe(
        train_loader,
        val_loader,
        input_dim,
        num_classes=num_classes,
        num_epochs=num_epochs,
        lr=0.001,
    )

    print("\nEvaluating on test set...")
    test_acc, cm, preds, labels = evaluate_model(model, test_loader)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Layer: {layer_name}")
    print(f"Task : {task_name}")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.4f}")
    print(f"Test Accuracy          : {test_acc:.4f}")
    print(f"Random Baseline        : {1/num_classes:.4f}")

    # Save model + history
    model_path = output_dir / "key_probe.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "major_minor_only": dataset.major_minor_only,
            "layer_name": layer_name,
            "test_acc": test_acc,
            "class_names": class_names,
            "history": history,
        },
        model_path,
    )
    print(f"\nSaved model to {model_path}")

    # Plots
    plot_training_curves(
        history, num_classes, save_path=output_dir / "training_curves.png"
    )
    plot_confusion_matrix(
        cm, class_names, save_path=output_dir / "confusion_matrix.png"
    )

    # Per-class accuracy
    per_class_stats_path = output_dir / "per_class_accuracy.txt"
    with open(per_class_stats_path, "w") as f:
        f.write("Per-class accuracy:\n")
        for i, name in enumerate(class_names):
            mask = np.array(labels) == i
            if mask.sum() > 0:
                acc = (np.array(preds)[mask] == i).mean()
                line = f"{name:12s}: {acc:.3f} ({mask.sum()} samples)\n"
                print("  " + line.strip())
                f.write(line)
    print(f"Saved per-class accuracy to {per_class_stats_path}")


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------
def main():
    METADATA_PATH = "/home/harinit9/orcd/pool/musicgen-data/dataset_metadata.json"
    RESULTS_ROOT = Path("results_sweep")
    RESULTS_ROOT.mkdir(exist_ok=True)

    # Figure out all layer names from one activation file
    all_layer_names = get_all_layer_names(METADATA_PATH)
    print("\nAll layers:")
    for name in all_layer_names:
        print("  ", name)

    # Map layer name -> numeric index
    def get_layer_idx(name):
        parts = name.split(".")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0

    # take every 4th layer by index, plus ensure the last layer is included
    indexed_layers = sorted(
        [(get_layer_idx(n), n) for n in all_layer_names],
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

    print("\nSelected layers for sweep:")
    for name in selected_layers:
        print("  ", name)

    # Hyperparams
    NUM_EPOCHS_MAJOR_MINOR = 50
    NUM_EPOCHS_24KEY = 50
    BATCH_SIZE = 32
    CONFIDENCE_THRESHOLD = 0.0  # keep all

    # Run experiments
    for layer_name in selected_layers:
        layer_idx = get_layer_idx(layer_name)

        # 1) Major/minor probe
        out_dir_mm = RESULTS_ROOT / f"layer_{layer_idx:02d}_major_minor"
        run_experiment(
            metadata_path=METADATA_PATH,
            layer_name=layer_name,
            major_minor_only=True,
            output_dir=out_dir_mm,
            num_epochs=NUM_EPOCHS_MAJOR_MINOR,
            batch_size=BATCH_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        # 2) 24-key probe
        out_dir_24 = RESULTS_ROOT / f"layer_{layer_idx:02d}_24key"
        run_experiment(
            metadata_path=METADATA_PATH,
            layer_name=layer_name,
            major_minor_only=False,
            output_dir=out_dir_24,
            num_epochs=NUM_EPOCHS_24KEY,
            batch_size=BATCH_SIZE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )


if __name__ == "__main__":
    main()
