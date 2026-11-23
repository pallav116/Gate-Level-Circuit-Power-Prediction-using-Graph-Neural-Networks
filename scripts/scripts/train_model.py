"""
scripts/train_model.py
-----------------------
Full training and evaluation pipeline for the Isomer Pair GNN benchmark.

Steps executed by this script:
  1. Load and normalise the dataset
  2. Split into Train (70%) / Validation (15%) / Test (15%)
  3. Train GraphSAGEPowerPredictor with Adam + early stopping
  4. Train Random Forest baseline on tabular aggregate features
  5. Evaluate both models on the held-out test set
  6. Print a side-by-side comparison table (MAE, RMSE, R²)

Usage:
    python scripts/train_model.py

Must be run from the project root directory.
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # synchronous CUDA errors (safer on Ada GPUs)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from utils.preprocessing import load_and_preprocess, split_dataset
from utils.metrics import compute_metrics
from models.graphsage import GraphSAGEPowerPredictor
from models.random_forest import train_random_forest

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "circuit_power_dataset.pt"
)

SEED       = 42     # random seed for reproducibility
BATCH_SIZE = 32     # number of graphs per mini-batch
EPOCHS     = 200    # maximum training epochs
LR         = 0.001  # Adam learning rate
PATIENCE   = 25     # early stopping: stop after this many epochs without improvement

# Device: default to CPU to avoid CUDA state issues.
# Set FORCE_CPU = False to use GPU if available.
FORCE_CPU  = True
DEVICE     = torch.device("cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu")


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training utilities ────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    Run one epoch of mini-batch gradient descent.

    Returns
    -------
    float — mean MSE loss for this epoch (averaged over all graphs)
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        pred = model(batch)                       # forward pass through GNN
        loss = criterion(pred, batch.y)           # MSE in normalised power space
        loss.backward()                           # backpropagation
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
):
    """
    Evaluate the model on a DataLoader without computing gradients.

    Returns
    -------
    loss   : float       — mean MSE loss
    y_true : np.ndarray  — ground-truth normalised power values
    y_pred : np.ndarray  — predicted normalised power values
    """
    model.eval()
    total_loss = 0.0
    preds_list, trues_list = [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch)
        loss  = criterion(pred, batch.y)

        total_loss   += loss.item() * batch.num_graphs
        preds_list.append(pred.cpu().numpy())
        trues_list.append(batch.y.cpu().numpy())

    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(trues_list)
    return total_loss / len(loader.dataset), y_true, y_pred


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)

    print("=" * 65)
    print("  Circuit Power Prediction: GraphSAGE vs Random Forest")
    print("  Isomer Pair Benchmark")
    print("=" * 65)

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        print(f"\n[ERROR] Dataset not found at:\n  {DATASET_PATH}")
        print("  Please run:  python scripts/generate_dataset.py  first.")
        return

    dataset, _, _, y_mean, y_std = load_and_preprocess(DATASET_PATH)
    train_set, val_set, test_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    # ── 2. Initialise GraphSAGE model ─────────────────────────────────────────
    model     = GraphSAGEPowerPredictor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining GraphSAGE on {DEVICE}  |  Parameters: {num_params:,}")
    print("-" * 65)

    best_val_loss   = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    # ── 3. Training loop with early stopping ──────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate(model, val_loader, criterion)

        # Track best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            tag = " (Best)"
        else:
            epochs_no_improve += 1
            tag = ""

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{tag}")

        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # ── 4. Evaluate GraphSAGE on test set ────────────────────────────────────
    model.load_state_dict(best_model_state)
    _, gnn_y_true, gnn_y_pred = evaluate(model, test_loader, criterion)
    gnn_metrics = compute_metrics(gnn_y_true, gnn_y_pred, y_mean.item(), y_std.item())

    print(f"\nGraphSAGE Test Results:")
    print(f"  MAE:  {gnn_metrics['MAE']:.4f}")
    print(f"  RMSE: {gnn_metrics['RMSE']:.4f}")
    print(f"  R²:   {gnn_metrics['R2']:.4f}")

    # ── 5. Train & evaluate Random Forest baseline ────────────────────────────
    _, rf_y_true, rf_y_pred = train_random_forest(
        train_set, val_set, test_set, y_mean.item(), y_std.item()
    )
    rf_metrics = compute_metrics(rf_y_true, rf_y_pred, y_mean.item(), y_std.item())

    # ── 6. Final comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  FINAL COMPARISON  (Test Set)")
    print("=" * 65)
    print(f"  {'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 65)
    print(f"  {'Random Forest':<22} {rf_metrics['MAE']:>10.4f} {rf_metrics['RMSE']:>10.4f} {rf_metrics['R2']:>10.4f}")
    print(f"  {'GraphSAGE (GNN)':<22} {gnn_metrics['MAE']:>10.4f} {gnn_metrics['RMSE']:>10.4f} {gnn_metrics['R2']:>10.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()scripts/train_model.py
