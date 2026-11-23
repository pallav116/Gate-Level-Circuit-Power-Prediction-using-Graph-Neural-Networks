"""
models/graphsage.py
-------------------
GraphSAGE-based Power Predictor for gate-level circuits.

This model treats a digital circuit as a graph where:
  - Nodes  = logic gates (AND, OR, NOT, NAND, NOR, XOR)
  - Edges  = signal wires (directed: driver gate -> receiver gate)

GraphSAGE allows each gate to aggregate feature information from its
directly connected neighbours, giving the model a structural awareness
that tabular models (e.g. Random Forest) fundamentally cannot have.

Architecture:
    Input (10 node features)
    → Linear projection to hidden_dim
    → 3 x SAGEConv layers  [BatchNorm → ReLU → Dropout]
    → Global Mean Pooling  (collapses node features → single graph vector)
    → MLP regression head  (hidden → hidden/2 → 1)
    → Predicted power (normalised)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data


class GraphSAGEPowerPredictor(nn.Module):
    """
    Graph-level power regression using GraphSAGE message passing.

    Parameters
    ----------
    in_features : int
        Dimensionality of the node feature vectors (default: 10).
    hidden_dim  : int
        Width of each hidden layer (default: 192).
    num_layers  : int
        Number of SAGEConv message-passing layers (default: 3).
    dropout     : float
        Dropout probability applied after every conv layer (default: 0.1).
    """

    def __init__(
        self,
        in_features: int = 10,
        hidden_dim:  int = 192,
        num_layers:  int = 3,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ── Input projection ─────────────────────────────────────────────────
        # Lift raw 10-dim node features to the working hidden dimension.
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
        )

        # ── GraphSAGE convolutional layers ───────────────────────────────────
        # Each SAGEConv layer aggregates neighbour embeddings (mean) and
        # concatenates them with the node's own embedding before updating.
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim, aggr="mean") for _ in range(num_layers)]
        )

        # Batch normalisation stabilises training on variable-size graphs
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # ── Regression MLP head ──────────────────────────────────────────────
        # Takes the pooled graph embedding and regresses to a scalar power value.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            A batched graph with attributes:
              data.x          — (N_total, in_features) node feature matrix
              data.edge_index — (2, E_total) COO edge list
              data.batch      — (N_total,)  batch assignment vector

        Returns
        -------
        torch.Tensor — (B,) predicted normalised power, one value per graph.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Project node features to hidden dimension
        x = self.input_proj(x)

        # 2. Message passing: each gate learns from its wired neighbours
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)   # aggregate neighbour messages
            x = self.bns[i](x)                 # normalise across nodes
            x = F.relu(x)
            x = self.dropout(x)

        # 3. Global mean pooling: collapse all gate embeddings → one graph vector
        x_graph = global_mean_pool(x, batch)    # (B, hidden_dim)

        # 4. Regression head: predict power from graph embedding
        out = self.head(x_graph)                # (B, 1)
        return out.squeeze(-1)                  # (B,)
