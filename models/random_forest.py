"""
models/random_forest.py
-----------------------
Random Forest tabular baseline for gate-level power prediction.

This model is the traditional ML baseline. It operates on 5 hand-crafted
aggregate graph statistics, deliberately having NO access to the actual
circuit wiring (graph topology).

On the Isomer Pair benchmark this model fails by design: because both
circuits in a pair share identical aggregate statistics, the RF receives
identical input for both and is forced to predict the same power for both,
resulting in high error. This is the key contrast with GraphSAGE.

Tabular features (5 per graph):
    0: num_gates      — total number of logic gates (nodes)
    1: num_wires      — total number of signal wires (edges)
    2: avg_fan_in     — average number of inputs per gate
    3: avg_fan_out    — average number of outputs per gate
    4: avg_cap        — average gate capacitance across the circuit
"""

import numpy as np
from typing import List, Tuple
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from utils.metrics import compute_metrics


def build_tabular_features(dataset: List[Data]) -> np.ndarray:
    """
    Convert a list of PyG graph objects into a 2-D tabular feature matrix.

    Each graph is reduced to 5 scalar aggregate statistics. This necessarily
    discards all information about which specific gates are connected, making
    the RF provably unable to solve the Isomer Pair benchmark.

    Parameters
    ----------
    dataset : list of Data objects

    Returns
    -------
    np.ndarray of shape (N_graphs, 5), dtype float32
    """
    rows = []
    for data in dataset:
        # Node feature layout:  [one-hot gate type (6)] [alpha (1)] [cap (1)] [fan-in (1)] [fan-out (1)]
        avg_cap     = float(data.x[:, 7].mean())   # column 7 = capacitance
        avg_fan_in  = float(data.x[:, 8].mean())   # column 8 = fan-in
        avg_fan_out = float(data.x[:, 9].mean())   # column 9 = fan-out

        rows.append([
            float(data.num_gates),
            float(data.num_wires),
            avg_fan_in,
            avg_fan_out,
            avg_cap,
        ])

    return np.array(rows, dtype=np.float32)


def train_random_forest(
    train_set: List[Data],
    val_set:   List[Data],
    test_set:  List[Data],
    y_mean:    float = 0.0,
    y_std:     float = 1.0,
) -> Tuple[RandomForestRegressor, np.ndarray, np.ndarray]:
    """
    Train a 100-tree Random Forest on tabular aggregate graph features.

    Targets are in z-score(power) space — identical to GNN training — for a
    fair apples-to-apples metric comparison.

    Parameters
    ----------
    train_set, val_set, test_set : list of Data objects
    y_mean, y_std : float — normalisation constants for inverse-transform

    Returns
    -------
    rf     : trained RandomForestRegressor
    y_true : (N_test,) true normalised-power values
    y_pred : (N_test,) predicted normalised-power values
    """
    print(f"\n{'='*65}")
    print("  Training  Random Forest Baseline")
    print(f"{'='*65}")

    # Build feature matrices from graph aggregates
    X_train = build_tabular_features(train_set)
    X_val   = build_tabular_features(val_set)
    X_test  = build_tabular_features(test_set)

    # Extract normalised power targets (same space as GNN)
    y_train = np.array([d.y.item() for d in train_set])
    y_val   = np.array([d.y.item() for d in val_set])
    y_test  = np.array([d.y.item() for d in test_set])

    # Standardise features (best practice for RF as well)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # Train Random Forest (100 trees, parallelised)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Report validation metrics
    val_pred    = rf.predict(X_val)
    val_metrics = compute_metrics(y_val, val_pred, y_mean, y_std)
    print(f"  RF Val  MAE:  {val_metrics['MAE']:.4f}")
    print(f"  RF Val  RMSE: {val_metrics['RMSE']:.4f}")
    print(f"  RF Val  R²:   {val_metrics['R2']:.4f}")

    # Generate test-set predictions
    y_pred = rf.predict(X_test)
    print("  ✓  Random Forest training complete")

    return rf, y_test, y_pred
