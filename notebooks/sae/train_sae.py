from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

LAYER_IDX = 22
DATA_PATH = Path(f"/home/harinit9/orcd/pool/musicgen-activations-nokey/acts_by_layer/layer_{LAYER_IDX:02d}.npy")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------
# 1. Load activations from memmap .npy file
# --------------------------------------------------------

def load_dataset(max_clips=None, filter_broken=True):
    """
    Load activations from the memmap file created by save_data_one_layer.py.
    Shape: [N_CLIPS, T, D] -> we average over time to get [N_CLIPS, D]
    """
    # Load as memmap for memory efficiency
    acts = np.load(DATA_PATH, mmap_mode='r')  # [N_CLIPS, T, D]
    
    if max_clips is not None:
        acts = acts[:max_clips]
    
    # Average over time dimension
    X = torch.from_numpy(acts.mean(axis=1).astype(np.float32))  # [N_CLIPS, D]
    
    # Filter out broken/low-variance clips
    if filter_broken:
        clip_vars = X.var(dim=1)
        good_mask = clip_vars > 0.001
        good_indices = torch.where(good_mask)[0]
        X = X[good_mask]
        print(f"Filtered out {(~good_mask).sum().item()} broken clips, keeping {len(X)}")
        return X, good_indices.numpy()
    
    return X, np.arange(len(X))


# --------------------------------------------------------
# 2. SAE model
# --------------------------------------------------------

class SAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, top_k: int = 64):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.top_k = top_k

    def forward(self, x):
        z_pre = self.encoder(x)  # [B, m]
        # TopK activation: keep only top_k values, zero the rest
        topk_vals, topk_idx = torch.topk(z_pre, self.top_k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, torch.relu(topk_vals))  # ReLU on top-k only
        x_hat = self.decoder(z)  # [B, d]
        return x_hat, z


def train_sae(
    X: torch.Tensor,
    hidden_mult: float = 2.0,  # Smaller dictionary (2048 -> 4096)
    top_k: int = 32,  # Fewer features = more selective
    batch_size: int = 64,  # Smaller batches for 344 samples
    epochs: int = 200,
    lr: float = 3e-4,
):
    X = X.to(DEVICE)
    N, d_in = X.shape
    d_hidden = int(hidden_mult * d_in)

    # normalize per-dim
    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True) + 1e-6
    Xn = (X - mu) / sigma

    dataset = torch.utils.data.TensorDataset(Xn)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae = SAE(d_in, d_hidden, top_k=top_k).to(DEVICE)
    
    # Initialize encoder with small random weights (encourages diversity)
    nn.init.xavier_uniform_(sae.encoder.weight, gain=0.1)
    nn.init.zeros_(sae.encoder.bias)
    nn.init.xavier_uniform_(sae.decoder.weight, gain=0.1)
    
    opt = optim.Adam(sae.parameters(), lr=lr)

    for epoch in range(epochs):
        sae.train()
        total_loss = 0.0
        all_z = []

        for (batch,) in loader:
            batch = batch.to(DEVICE)
            x_hat, z = sae(batch)
            loss = ((x_hat - batch) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(batch)
            all_z.append((z > 0).float().detach())

        # Track feature diversity
        if (epoch + 1) % 20 == 0:
            z_all = torch.cat(all_z, dim=0)
            freq = z_all.mean(dim=0)
            active = (freq > 0).sum().item()
            max_freq = freq.max().item()
            print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/N:.6f} | active_feats={active} | max_freq={max_freq:.2%}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/N:.6f}")

    ckpt = {
        "sae_state_dict": sae.state_dict(),
        "mu": mu.cpu(),
        "sigma": sigma.cpu(),
        "d_in": d_in,
        "d_hidden": d_hidden,
        "top_k": top_k,
        "layer_idx": LAYER_IDX,
    }
    out_path = Path("checkpoints") / f"sae_layer{LAYER_IDX:02d}.pt"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(ckpt, out_path)
    print(f"Saved SAE checkpoint to {out_path}")

    return sae, mu, sigma


if __name__ == "__main__":
    X, good_indices = load_dataset()  # optionally pass max_clips for debugging
    print(f"Training on {len(X)} good clips")
    train_sae(X)
