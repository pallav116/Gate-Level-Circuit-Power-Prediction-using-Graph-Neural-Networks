"""
scripts/generate_dataset.py
----------------------------
Isomer Pair Circuit Power Dataset Generator.

Generates a synthetic benchmark dataset for evaluating Graph Neural Networks
on gate-level circuit power prediction.

KEY CONCEPT — Isomer Pairs
--------------------------
Each data point is generated as a PAIR of two circuits (A and B) that have
deliberately identical aggregate statistics:

    ┌──────────────┬────────┬────────┐
    │ Feature      │   A    │   B    │
    ├──────────────┼────────┼────────┤
    │ Gate count   │ same   │ same   │
    │ Wire count   │ same   │ same   │
    │ Avg cap      │ same   │ same   │
    │ Avg fan-in   │ same   │ same   │
    │ Avg fan-out  │ same   │ same   │
    └──────────────┴────────┴────────┘

The ONLY difference is the wiring of three selected nodes:
  Circuit A (hotspot):   n1 → n2 (high-cap → high-cap) → 2× power spike
  Circuit B (safe):      n1 → n3 (high-cap → low-cap)  → normal power

Because the aggregate features are identical, any tabular model (e.g. Random
Forest) receives exactly the same input for both circuits and is forced to
predict the same power for both — resulting in high error. A GNN can look at
the edges and correctly distinguish them.

Circuit model:
    Nodes  = logic gates (AND, OR, NOT, NAND, NOR, XOR)
    Edges  = signal wires (directed: driver → receiver)
    Target = total circuit dynamic power (in arbitrary units)

Node features (10 dimensions):
    [0-5]  Gate type one-hot encoding (6 gate types)
    [6]    Switching activity (alpha)  ∈ [0.1, 0.5]
    [7]    Gate capacitance            ∈ [1.0, 5.0]
    [8]    Fan-in  (number of input wires)
    [9]    Fan-out (number of output wires)
"""

import os
import random
import copy
from typing import List

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_PAIRS  = 2500                               # generates 5000 total graphs
MIN_NODES  = 15                                 # minimum gates per circuit
MAX_NODES  = 100                                # maximum gates per circuit
GATE_TYPES = ["AND", "OR", "NOT", "NAND", "NOR", "XOR"]
NUM_GATE_TYPES = len(GATE_TYPES)               # 6


# ── Helper: DAG Generation ────────────────────────────────────────────────────

def generate_base_dag(num_nodes: int) -> nx.DiGraph:
    """
    Generate a random Directed Acyclic Graph (DAG) representing a circuit.

    Nodes are assigned to random layers (0 → max_depth). Edges are only
    allowed to point from a lower layer to a higher layer, guaranteeing
    acyclicity (no feedback loops — correct for combinational logic).

    Parameters
    ----------
    num_nodes : int — number of logic gates in the circuit

    Returns
    -------
    nx.DiGraph — a connected DAG
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Assign each node a random depth level
    max_depth = int(np.sqrt(num_nodes))
    layers = {i: random.randint(0, max_depth) for i in range(num_nodes)}

    # Add random forward edges (lower layer → higher layer only)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if layers[i] < layers[j] and random.random() < 2.0 / num_nodes:
                G.add_edge(i, j)

    # Ensure no isolated nodes (every node has at least one connection)
    for node in list(nx.isolates(G)):
        if node > 0:
            G.add_edge(random.randint(0, node - 1), node)
        else:
            G.add_edge(node, random.randint(1, num_nodes - 1))

    return G


# ── Main Dataset Generation ───────────────────────────────────────────────────

def generate_dataset(num_pairs: int = NUM_PAIRS) -> List[Data]:
    """
    Generate the full Isomer Pair dataset.

    For each pair, one base DAG and one set of node features are created.
    Then two variants are produced by wiring a specific triplet of nodes
    differently, creating the hotspot (A) and safe (B) circuits.

    Parameters
    ----------
    num_pairs : int — number of isomer pairs to generate (total = 2 × num_pairs)

    Returns
    -------
    list of PyG Data objects (shuffled so pairs are not adjacent)
    """
    print(f"Generating {num_pairs} Isomer Pairs ({num_pairs * 2} total graphs) ...")
    dataset = []

    for _ in tqdm(range(num_pairs), desc="Generating Isomer Pairs"):
        num_nodes = random.randint(MIN_NODES, MAX_NODES)
        base_G    = generate_base_dag(num_nodes)

        # ── Assign base node features ─────────────────────────────────────────
        x_features = np.zeros((num_nodes, 10), dtype=np.float32)
        base_power = 0.0

        for i in range(num_nodes):
            gate_idx = random.randint(0, NUM_GATE_TYPES - 1)
            x_features[i, gate_idx] = 1.0          # one-hot gate type

            alpha = random.uniform(0.1, 0.5)        # switching activity
            cap   = random.uniform(1.0, 5.0)        # capacitance
            x_features[i, 6] = alpha
            x_features[i, 7] = cap
            # fan-in and fan-out are filled per-isomer below

            base_power += alpha * cap               # base power: P = α × C

        # ── Select the three key nodes ────────────────────────────────────────
        # n1: high-cap driver   n2: high-cap receiver   n3: low-cap receiver
        nodes = list(base_G.nodes())
        random.shuffle(nodes)
        n1, n2, n3 = nodes[:3]

        x_features[n1, 7] = 5.0    # force n1 → high capacitance
        x_features[n2, 7] = 5.0    # force n2 → high capacitance
        x_features[n3, 7] = 1.0    # force n3 → low  capacitance

        # Remove any pre-existing edges between the three key nodes
        for u, v in [(n1, n2), (n1, n3)]:
            if base_G.has_edge(u, v):
                base_G.remove_edge(u, v)

        # ── Build Circuit A — HOTSPOT ─────────────────────────────────────────
        # Wiring: n1 → n2  (high-cap drives high-cap → power spike)
        #         n2 → n3  (compensating edge to equalise n3's fan-in)
        G_A = copy.deepcopy(base_G)
        G_A.add_edge(n1, n2)
        G_A.add_edge(n2, n3)

        x_A = copy.deepcopy(x_features)
        for i in range(num_nodes):
            x_A[i, 8] = G_A.in_degree(i)
            x_A[i, 9] = G_A.out_degree(i)

        power_A = base_power * 2.0  # +100% power penalty for the hotspot

        # ── Build Circuit B — SAFE ────────────────────────────────────────────
        # Wiring: n1 → n3  (high-cap drives low-cap  → no spike)
        #         n2 → n3  (compensating edge to equalise n3's fan-in)
        # Both circuits now have IDENTICAL edge count, fan-in, and fan-out.
        G_B = copy.deepcopy(base_G)
        G_B.add_edge(n1, n3)
        G_B.add_edge(n2, n3)

        x_B = copy.deepcopy(x_features)
        for i in range(num_nodes):
            x_B[i, 8] = G_B.in_degree(i)
            x_B[i, 9] = G_B.out_degree(i)

        power_B = base_power * 1.0  # normal power — no hotspot

        # ── Convert to PyG Data and append ───────────────────────────────────
        for G, x, power in [(G_A, x_A, power_A), (G_B, x_B, power_B)]:
            edges = list(G.edges())
            edge_index = (
                torch.tensor(edges, dtype=torch.long).t().contiguous()
                if edges else torch.empty((2, 0), dtype=torch.long)
            )
            data = Data(
                x          = torch.tensor(x, dtype=torch.float),
                edge_index = edge_index,
                y          = torch.tensor([power], dtype=torch.float),
            )
            data.num_gates = num_nodes
            data.num_wires = len(edges)
            dataset.append(data)

    # Shuffle so that paired circuits are not always adjacent in the loader
    random.shuffle(dataset)
    return dataset


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True
    )
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "circuit_power_dataset.pt"
    )

    dataset = generate_dataset(NUM_PAIRS)
    torch.save(dataset, out_path)

    print(f"\nDataset saved  →  {out_path}")
    print(f"Total graphs   :  {len(dataset)}")
    print(f"Isomer pairs   :  {len(dataset) // 2}")
