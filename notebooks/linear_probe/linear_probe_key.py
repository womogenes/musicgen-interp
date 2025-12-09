# %%
import torch
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
import h5py

# %%
import logging

logging.basicConfig(
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger()

# %%
# Constants
METADATA_PATH = "/home/harinit9/orcd/pool/musicgen-data-nokey-long/dataset_metadata.json"
ACT_BATCHED_PATH = Path("/home/wyf/orcd/pool/musicgen-activations-nokey-long/activations_pooled")
ACTS_BY_LAYER_PATH = Path("/home/wyf/orcd/pool/musicgen-activations-nokey-long/acts_by_layer")

# %%
with open(METADATA_PATH) as fin:
    metadata = json.load(fin)

# %%
# Define main fitting function based on layer idx

class_labels = ["major", "minor"]

def train_linear_probe(
    layer_idx: int,
    pca_dim: int,
    pooling_strat: str,
    test_size: float = 0.3,
    train: bool = True,
):
    acts = np.load(ACTS_BY_LAYER_PATH / f"layer_{layer_idx:02d}.npy")
    logger.info(f"Logistic regression on {layer_idx=}")

    # Pool across time
    logger.info("Pooling...")
    if pooling_strat == "max":
        X_pooled = np.max(acts, axis=1)
    else:
        X_pooled = np.mean(acts, axis=1)

    X_pooled = acts.reshape(acts.shape[0], -1)

    # Normalize per feature
    logger.info("Normalizing...")
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_pooled)

    # Run PCA to reduce dimensionality
    from sklearn.decomposition import PCA

    logger.info("Running pca...")
    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(X_norm)     # shape: (1000, pca_dim)
    y_raw = np.array([
        class_labels.index(clip["key_info"]["mode"])
        for clip in metadata
    ])

    # Make train/test splits
    logger.info("Making test splits...")

    for _ in range(1024 if train else 1):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y_raw,
            test_size=test_size,
            stratify=y_raw,
        )

        # Run logistic regression
        from sklearn.linear_model import LogisticRegression

        # logger.info("Fitting logistic regression...")
        reg = LogisticRegression()
        reg.fit(X_train, y_train)

        if not train:
            # Used in production; return regression coefs
            yield reg, pca, scaler

        # Plot confusion matrix
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

        def evaluate(X, y):
            y_pred = reg.predict(X) >= 0.5

            cm = confusion_matrix(y, y_pred)
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
            # disp.plot()
            
            acc = accuracy_score(y, y_pred)
            return cm, acc
        
        # logger.info("Running eval...")
        _, train_acc = evaluate(X_train, y_train)
        _, test_acc = evaluate(X_test, y_test)

        print(f"[{layer_idx=}]  train_acc={train_acc * 100:.3f}%, test_acc={test_acc * 100:.3f}")
        yield train_acc, test_acc

# %%

if __name__ == "__main__":
    records = []

    # %%
    # logger.setLevel("ERROR")
    logger.setLevel("DEBUG")

    for layer_idx in range(48):
        accs = list(train_linear_probe(layer_idx, 128, "max", train=True))
        records.append((layer_idx, accs))

    # %%
    train_acc = []
    test_acc = []

    for idx, samples in records:
        for train, test in samples:
            train_acc.append((idx, train))
            test_acc.append((idx, test))

    # print(records)

    # %%
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from scipy import stats

    # Organize data by transformer block index
    train_data = defaultdict(list)
    test_data = defaultdict(list)

    for idx, samples in records:
        for train, test in samples:
            train_data[idx].append(train)
            test_data[idx].append(test)

    # Compute mean and 95% CI for each block
    train_means = []
    train_cis = []
    test_means = []
    test_cis = []
    indices = sorted(train_data.keys())

    for idx in indices:
        train_vals = np.array(train_data[idx])
        test_vals = np.array(test_data[idx])
        
        n_train = len(train_vals)
        n_test = len(test_vals)
        
        # Mean
        train_mean = np.mean(train_vals)
        test_mean = np.mean(test_vals)
        
        # 95% CI using t-distribution
        train_ci = stats.t.ppf(0.975, n_train-1) * np.std(train_vals, ddof=1) / np.sqrt(n_train)
        test_ci = stats.t.ppf(0.975, n_test-1) * np.std(test_vals, ddof=1) / np.sqrt(n_test)
        
        train_means.append(train_mean)
        train_cis.append(train_ci)
        test_means.append(test_mean)
        test_cis.append(test_ci)

        print(f"layer {idx:02d} | train: {train_mean*100:.2f}%, train: {test_mean*100:.2f}%")

    logger.setLevel("INFO")

    # Plot with 95% CI error bars
    plt.errorbar(indices, train_means, yerr=train_cis, fmt='o', label="Train (95% CI)")
    plt.errorbar(indices, test_means, yerr=test_cis, fmt='o', label="Test (95% CI)")
    plt.xlabel("Transformer block index")
    plt.ylabel("Accuracy")
    plt.title("Linear probe (95% CI)")
    plt.legend()
    plt.grid()
    plt.show()

    import time
    plt.savefig(f"/home/wyf/musicgen-interp/notebooks/logs/{time.time()}.png")
    