"""
Train linear probes to classify musical key from MusicGen activations.
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

class ActivationDataset(Dataset):
    """Dataset of activations with key labels - preloads everything into memory"""
    
    def __init__(self, metadata_path, layer_name=None, confidence_threshold=0.0, major_minor_only=False):
        """
        Args:
            metadata_path: Path to dataset_metadata.json
            layer_name: Which layer to use (e.g., 'decoder.layers.23'). 
                       If None, will use the last layer found.
            confidence_threshold: Filter out examples below this confidence
            major_minor_only: If True, classify major vs minor (2 classes) instead of 24 keys
        """
        self.major_minor_only = major_minor_only
        self.num_classes = 2 if major_minor_only else NUM_KEYS
        self.class_names = MODE_NAMES if major_minor_only else KEY_NAMES
        
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        
        # Filter by confidence
        self.metadata = [
            m for m in self.metadata 
            if m['key_info']['key'] is not None 
            and m['key_info']['confidence'] >= confidence_threshold
        ]
        
        task_name = "Major/Minor" if major_minor_only else "24-Key"
        print(f"Task: {task_name} classification ({self.num_classes} classes)")
        print(f"Found {len(self.metadata)} samples (confidence >= {confidence_threshold})")
        
        first_act = torch.load(self.metadata[0]['activations_path'])
        
        def get_layer_num(name):
            parts = name.split('.')
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            return 0
        
        self.available_layers = sorted(first_act.keys(), key=get_layer_num)
        
        if layer_name is None:
            self.layer_name = self.available_layers[-1] # use last layer
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
            if (i + 1) % 10 == 0:
                print(f"  Loading {i+1}/{len(self.metadata)}...")
            
            activations = torch.load(meta['activations_path'])
            layer_act_list = activations[self.layer_name]
            
            # Concatenate and pool
            layer_act = torch.cat(layer_act_list, dim=0)
            layer_act_flat = layer_act.reshape(layer_act.shape[0], -1)
            pooled = layer_act_flat.mean(dim=0)
            
            self.features.append(pooled)
            
            # Get label based on mode
            if major_minor_only:
                mode = meta['key_info']['mode']  # 'major' or 'minor'
                self.labels.append(MODE_TO_IDX[mode])
            else:
                self.labels.append(KEY_TO_IDX[meta['key_info']['key']])
        
        # Stack into tensors
        self.features = torch.stack(self.features)  # [N, hidden_dim]
        self.labels = torch.tensor(self.labels, dtype=torch.long)  # [N]
        
        print(f"Preloaded! Feature shape: {self.features.shape}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LinearProbe(nn.Module):
    """Simple linear classifier"""
    
    def __init__(self, input_dim, num_classes=NUM_KEYS):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def train_probe(train_loader, val_loader, input_dim, num_classes=NUM_KEYS, num_epochs=50, lr=0.001):
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
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
    
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
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
    
    # Loss curves
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    random_baseline = 1/num_classes
    ax2.plot(history['val_accs'], label='Val Accuracy')
    ax2.axhline(y=random_baseline, color='r', linestyle='--', label=f'Random ({random_baseline:.3f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    figsize = (6, 5) if len(class_names) <= 3 else (12, 10)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def main():
    # Configuration
    METADATA_PATH = "data-large/dataset_metadata.json"
    CONFIDENCE_THRESHOLD = 0.0
    LAYER_NAME = "decoder.model.decoder.layers.9"
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.125  # 0.125 of 0.8 = 0.1 of total
    
    # === TOGGLE THIS ===
    MAJOR_MINOR_ONLY = True  # True = 2 classes (major/minor), False = 24 classes (all keys)
    
    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("="*60)
    print("Training Key Classification Probe")
    print("="*60)
    
    # Load dataset
    dataset = ActivationDataset(
        METADATA_PATH, 
        layer_name=LAYER_NAME,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        major_minor_only=MAJOR_MINOR_ONLY
    )
    
    # Get input dimension from first sample
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[0]
    print(f"Input dimension: {input_dim}")
    
    # Split dataset
    indices = list(range(len(dataset)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=VAL_SIZE, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, 
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )
    val_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx)
    )
    test_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx)
    )
    
    # Train model
    num_classes = dataset.num_classes
    class_names = dataset.class_names
    
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    model, history = train_probe(
        train_loader, val_loader, input_dim,
        num_classes=num_classes,
        num_epochs=NUM_EPOCHS, lr=LEARNING_RATE
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_acc, cm, preds, labels = evaluate_model(model, test_loader)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Random Baseline: {1/num_classes:.4f}")
    
    # Save model
    model_path = OUTPUT_DIR / "key_probe.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'major_minor_only': dataset.major_minor_only,
        'layer_name': dataset.layer_name,
        'test_acc': test_acc,
        'class_names': class_names,
    }, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Plot results
    plot_training_curves(history, num_classes, save_path=OUTPUT_DIR / "training_curves.png")
    plot_confusion_matrix(cm, class_names, save_path=OUTPUT_DIR / "confusion_matrix.png")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for i, name in enumerate(class_names):
        mask = np.array(labels) == i
        if mask.sum() > 0:
            acc = (np.array(preds)[mask] == i).mean()
            print(f"  {name:12s}: {acc:.3f} ({mask.sum()} samples)")


if __name__ == "__main__":
    main()
