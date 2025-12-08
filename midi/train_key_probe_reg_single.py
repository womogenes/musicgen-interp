import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# Global device & seed
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# -------------------------------------------------------
# Label definitions
# -------------------------------------------------------
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


# -------------------------------------------------------
# Dataset
# -------------------------------------------------------
class ActivationDataset(Dataset):
    """
    Dataset of activations with key labels (DETECTED KEY).

    Labels:
      - If major_minor_only=True:   0 = major, 1 = minor based on detected_key
      - If major_minor_only=False:  24-way classification over KEY_NAMES based on detected_key

    confidence_threshold:
      - If >0, filter samples by detected key confidence in key_info.
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

        # keep only samples with a valid detected_key
        self.metadata = [
            m for m in self.metadata
            if m.get("detected_key") in KEY_TO_IDX
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
            # default = last layer
            self.layer_name = self.available_layers[-1]
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

            # Concatenate along first dim, flatten tokens, mean pool
            layer_act = torch.cat(layer_act_list, dim=0)
            hidden_dim = layer_act.shape[-1]
            layer_act_flat = layer_act.reshape(-1, hidden_dim)
            pooled = layer_act_flat.mean(dim=0)  # [hidden_dim]

            self.features.append(pooled)

            # LABELS FROM DETECTED KEY
            detected_key = meta["detected_key"]
            if self.major_minor_only:
                mode_str = "major" if detected_key.endswith("_major") else "minor"
                self.labels.append(MODE_TO_IDX[mode_str])
            else:
                self.labels.append(KEY_TO_IDX[detected_key])

        self.features = torch.stack(self.features)  # [N, hidden_dim]
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # [N]
        print(f"Preloaded! Feature shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# -------------------------------------------------------
# Model + training utils
# -------------------------------------------------------
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
    class_weights=None,
):
    """Train a linear probe with optional class-balanced loss."""
    model = LinearProbe(input_dim, num_classes=num_classes).to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class-balanced cross entropy.")
        print("Class weights:", class_weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using unweighted cross entropy.")

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


# -------------------------------------------------------
# Main: single-layer experiment
# -------------------------------------------------------
def main():
    # ================= CONFIG =================
    METADATA_PATH = "/home/harinit9/orcd/pool/musicgen-data-nokey/dataset_metadata.json"

    # If None: use last layer; else put a specific one, e.g.:
    # LAYER_NAME = "decoder.model.decoder.layers.47"
    LAYER_NAME = "decoder.model.decoder.layers.47"

    CONFIDENCE_THRESHOLD = 0.0   # 0.0 = use all; >0 = filter by detected key confidence
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.125  # of the remaining 0.8

    # === TOGGLE THIS ===
    MAJOR_MINOR_ONLY = True  # True = 2 classes, False = 24 classes

    # Output dir
    task_suffix = "major_minor" if MAJOR_MINOR_ONLY else "24key"
    OUTPUT_DIR = Path(f"results_single_layer_nokey/{task_suffix}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Key Classification Probe (DETECTED KEY LABELS)")
    print(f"Layer: {LAYER_NAME}")
    print(f"Task : {'Major/Minor' if MAJOR_MINOR_ONLY else '24-Key'}")
    print("=" * 60)

    # Load dataset
    dataset = ActivationDataset(
        METADATA_PATH,
        layer_name=LAYER_NAME,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        major_minor_only=MAJOR_MINOR_ONLY,
    )

    # Get input dimension from first sample
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]
    print(f"Input dimension: {input_dim}")

    # Compute class weights (for *this* task)
    labels_np = dataset.labels.numpy()
    classes = np.arange(dataset.num_classes)
    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_np,
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)
    print("Computed class weights:", class_weights_np)

    # Split dataset
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=42, stratify=labels_np
    )

    # stratify again within train/val
    labels_train_val = labels_np[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_SIZE,
        random_state=42,
        stratify=labels_train_val,
    )

    print("\nDataset split:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
    )
    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
    )

    num_classes = dataset.num_classes
    class_names = dataset.class_names

    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    model, history = train_probe(
        train_loader,
        val_loader,
        input_dim,
        num_classes=num_classes,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        class_weights=class_weights,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc, cm, preds, labels = evaluate_model(model, test_loader)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Random Baseline: {1/num_classes:.4f}")

    # Save model
    model_path = OUTPUT_DIR / "key_probe.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "major_minor_only": dataset.major_minor_only,
            "layer_name": dataset.layer_name,
            "test_acc": test_acc,
            "class_names": class_names,
            "history": history,
            "class_weights": class_weights_np,
        },
        model_path,
    )
    print(f"\nSaved model to {model_path}")

    # Plot results
    plot_training_curves(
        history, num_classes, save_path=OUTPUT_DIR / "training_curves.png"
    )
    plot_confusion_matrix(
        cm, class_names, save_path=OUTPUT_DIR / "confusion_matrix.png"
    )

    # Per-class accuracy
    print("\nPer-class accuracy:")
    per_class_path = OUTPUT_DIR / "per_class_accuracy.txt"
    with open(per_class_path, "w") as f:
        f.write("Per-class accuracy:\n")
        for i, name in enumerate(class_names):
            mask = np.array(labels) == i
            if mask.sum() > 0:
                acc = (np.array(preds)[mask] == i).mean()
                line = f"  {name:12s}: {acc:.3f} ({mask.sum()} samples)\n"
                print(line.rstrip())
                f.write(line)
    print(f"Saved per-class accuracy to {per_class_path}")


if __name__ == "__main__":
    main()
