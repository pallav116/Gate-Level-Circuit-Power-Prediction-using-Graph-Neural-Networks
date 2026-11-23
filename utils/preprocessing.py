"""
utils/preprocessing.py
-----------------------
Data loading, normalisation, and train/val/test splitting.

Normalisation strategy:
  - Targets (power):   z-score standardisation → mean≈0, std≈1
  - Node features:     global z-score across all nodes in the dataset

Why normalise?
  Neural networks train faster and more stably when inputs and outputs
  are roughly in the range [-1, 1]. Raw power values vary widely, so
  normalising ensures the MSE loss is not dominated by outliers.
"""

import torch
import random
from torch_geometric.data import Data
from typing import List, Tuple


def load_and_preprocess(
    path: str,
) -> Tuple[List[Data], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the dataset and apply z-score normalisation to both node
    features and power targets.

    Parameters
    ----------
    path : str — path to the .pt dataset file

    Returns
    -------
    dataset   : list of normalised Data objects
    feat_mean : (10,) tensor — per-feature mean (for inverse transform)
    feat_std  : (10,) tensor — per-feature std  (for inverse transform)
    y_mean    : scalar tensor — target mean      (for inverse transform)
    y_std     : scalar tensor — target std       (for inverse transform)
    """
    print(f"\n[DATA]  Loading dataset from:\n        {path}")
    dataset: List[Data] = torch.load(path, weights_only=False)
    print(f"[DATA]  Loaded {len(dataset)} graphs")

    # ── 1. Normalise power targets ────────────────────────────────────────────
    all_y  = torch.cat([d.y for d in dataset])
    y_mean = all_y.mean()
    y_std  = all_y.std().clamp(min=1e-8)

    for data in dataset:
        data.y = (data.y - y_mean) / y_std

    print(f"[DATA]  Target normalised  (μ={y_mean.item():.2f}, σ={y_std.item():.2f})")

    # ── 2. Normalise node features ────────────────────────────────────────────
    all_x     = torch.cat([d.x for d in dataset], dim=0)   # (N_total, 10)
    feat_mean = all_x.mean(dim=0)
    feat_std  = all_x.std(dim=0).clamp(min=1e-8)

    for data in dataset:
        data.x = (data.x - feat_mean) / feat_std

    print(f"[DATA]  Node features normalised  (dim=10)")

    return dataset, feat_mean, feat_std, y_mean, y_std


def split_dataset(
    dataset:     List[Data],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Shuffle and split the dataset into Train / Validation / Test subsets.

    The shuffle is applied before splitting to prevent ordering bias
    (the generator appends isomer pairs sequentially).

    Split ratios: 70% train | 15% validation | 15% test

    Parameters
    ----------
    dataset      : list of Data objects
    train_ratio  : fraction of data to use for training
    val_ratio    : fraction of data to use for validation

    Returns
    -------
    train_set, val_set, test_set : three lists of Data objects
    """
    shuffled = dataset[:]
    random.shuffle(shuffled)

    n      = len(shuffled)
    n_tr   = int(n * train_ratio)
    n_val  = int(n * val_ratio)

    train_set = shuffled[:n_tr]
    val_set   = shuffled[n_tr : n_tr + n_val]
    test_set  = shuffled[n_tr + n_val :]

    print(f"[DATA]  Split:  train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    return train_set, val_set, test_set
