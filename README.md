# Gate-Level Circuit Power Prediction using Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyG-2.3+-red)
![License](https://img.shields.io/badge/License-MIT-green)

A graph-based deep learning approach for predicting gate-level digital circuit power consumption using **GraphSAGE**, benchmarked against a traditional **Random Forest** model.

The project investigates whether explicitly learning circuit connectivity through Graph Neural Networks (GNNs) provides an advantage over conventional machine learning models that rely on aggregate circuit statistics.

---

## Research Question

**Can Graph Neural Networks predict circuit power more accurately than traditional machine learning models by exploiting the structural information inherent in digital circuits?**

To investigate this, we develop a controlled synthetic benchmark using *Isomer Pairs*: pairs of circuits with identical aggregate features but different internal connectivity patterns.

---

## Key Results

GraphSAGE significantly outperforms the Random Forest baseline on the synthetic Isomer Pair benchmark.

| Model | MAE ↓ | RMSE ↓ | R² Score ↑ |
|:---|---:|---:|---:|
| Random Forest | 36.64 | 41.41 | 0.1205 |
| **GraphSAGE (GNN)** | **9.88** | **14.19** | **0.8966** |

**Key observations:**
- 3.7× lower Mean Absolute Error with GraphSAGE.
- R² score improves from 0.1205 to 0.8966.
- GraphSAGE captures connectivity-dependent power behaviour that aggregate tabular features cannot distinguish in this benchmark.

---

## Why Represent Circuits as Graphs?

Digital circuits can naturally be represented as directed graphs.

- **Nodes:** Logic gates such as AND, OR, NOT, NAND, NOR and XOR.
- **Edges:** Signal connections between gates.
- **Node Features:** Gate characteristics, capacitance, fan-in, fan-out and other electrical attributes.
- **Graph Target:** Estimated circuit power consumption.

Traditional ML models such as Random Forest operate on fixed-length feature vectors. Converting a circuit into aggregate statistics can discard information about the exact connectivity between gates.

Graph Neural Networks address this limitation by learning representations directly from graph structures.

**GraphSAGE** uses neighbourhood aggregation, allowing each gate to learn from its connected neighbours. Stacking multiple message-passing layers enables the model to capture increasingly larger structural patterns.

---

## The Isomer Pair Benchmark

The central idea behind this project is to construct controlled pairs of circuits with identical aggregate statistics but different connectivity.

Each pair contains two circuits:

| Feature | Circuit A | Circuit B |
|:---|:---:|:---:|
| Number of gates | Same | Same |
| Number of wires | Same | Same |
| Average capacitance | Same | Same |
| Average fan-in | Same | Same |
| Average fan-out | Same | Same |
| Internal connectivity | Different | Different |
| Target power | Higher | Normal |

### Circuit A — High-Power Hotspot

A high-capacitance gate drives another high-capacitance gate, creating a topology-dependent power increase.

### Circuit B — Normal-Power Configuration

The high-capacitance gate is connected to a low-capacitance gate, resulting in normal power consumption.

### Why Is This Important?

The Random Forest receives identical aggregate feature vectors for both circuits. Consequently, it cannot distinguish between the two configurations using these features.

GraphSAGE, however, receives the actual graph connectivity and can learn the structural patterns associated with the power difference.

This creates a controlled experiment for studying the contribution of graph structure to circuit power prediction.

---

## Model Architecture

### 1. GraphSAGE Power Predictor

The GNN uses a three-layer GraphSAGE architecture with global mean pooling for graph-level regression.

```text
Input Graph (10 Node Features)
          |
          v
Linear Projection (10 → 192)
          |
         ReLU
          |
          v
SAGEConv (192 → 192)
          |
     BatchNorm + ReLU
          |
          v
SAGEConv (192 → 192)
          |
     BatchNorm + ReLU
          |
          v
SAGEConv (192 → 192)
          |
     BatchNorm + ReLU
          |
          v
Global Mean Pooling
          |
          v
Linear (192 → 96)
          |
    ReLU + Dropout
          |
          v
Linear (96 → 1)
          |
          v
Predicted Circuit Power
```

**Architecture details:**
- Input representation: 10 node features.
- Graph convolution: 3 GraphSAGE layers.
- Hidden dimension: 192.
- Normalisation: Batch Normalisation.
- Activation: ReLU.
- Graph readout: Global Mean Pooling.
- Regression head: 192 → 96 → 1.

### 2. Random Forest Baseline

The baseline model uses five aggregate features extracted from each circuit:

1. Number of gates.
2. Number of wires.
3. Average fan-in.
4. Average fan-out.
5. Average capacitance.

This provides a conventional tabular ML baseline against which the graph-based model can be evaluated.

---

## Dataset Generation

The dataset is generated programmatically using a custom Isomer Pair DAG generator.

**Dataset configuration:**

| Parameter | Value |
|:---|---:|
| Isomer pairs | 2,500 |
| Total circuits | 5,000 |
| Node features | 10 |
| Graph representation | Directed Acyclic Graph |
| Random seed | 42 |

Each pair contains two circuits with matching aggregate statistics and different connectivity patterns.

The target power values are generated using a controlled synthetic power model designed to incorporate topology-dependent effects.

---

## Project Structure

```text
gate-level-power-gnn/
│
├── data/
│   └── circuit_power_dataset.pt
│
├── models/
│   ├── graphsage.py
│   └── random_forest.py
│
├── scripts/
│   ├── generate_dataset.py
│   └── train_model.py
│
├── utils/
│   ├── preprocessing.py
│   └── metrics.py
│
├── requirements.txt
└── README.md
```

### Directory Overview

| Directory / File | Description |
|:---|:---|
| `data/` | Generated circuit graph dataset |
| `models/graphsage.py` | GraphSAGE regression model |
| `models/random_forest.py` | Random Forest baseline |
| `scripts/generate_dataset.py` | Synthetic Isomer Pair dataset generator |
| `scripts/train_model.py` | Training and evaluation pipeline |
| `utils/preprocessing.py` | Feature normalisation and dataset splitting |
| `utils/metrics.py` | Regression evaluation metrics |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pallav116/Gate-Level-Circuit-Power-Prediction-using-Graph-Neural-Networks.git

cd gate-level-power-gnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Requirements

```text
torch>=2.0.0
torch_geometric>=2.3.0
numpy>=1.24.0
scikit-learn>=1.2.0
networkx>=3.0.0
tqdm>=4.65.0
```

---

## Running the Project

### Step 1: Generate the Dataset

```bash
python scripts/generate_dataset.py
```

Generates 2,500 Isomer Pairs (5,000 circuits) and saves the dataset to:

```text
data/circuit_power_dataset.pt
```

### Step 2: Train and Evaluate Models

```bash
python scripts/train_model.py
```

This script:
1. Loads and preprocesses the generated dataset.
2. Splits the dataset into training, validation and testing sets.
3. Trains the GraphSAGE model.
4. Trains the Random Forest baseline.
5. Evaluates both models using MAE, RMSE and R².
6. Prints a comparative performance report.

### GPU Configuration

Training defaults to CPU.

To enable GPU training, update the following setting in `scripts/train_model.py`:

```python
FORCE_CPU = False
```

---

## Training Configuration

| Hyperparameter | Value |
|:---|---:|
| Optimiser | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Maximum Epochs | 200 |
| Early Stopping Patience | 25 |
| Hidden Dimension | 192 |
| GraphSAGE Layers | 3 |
| Dropout | 0.1 |
| Loss Function | MSE Loss |
| Random Seed | 42 |

---

## Evaluation Metrics

The models are evaluated using three standard regression metrics.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between predicted and actual power values.

**Root Mean Squared Error (RMSE)**

Penalises larger prediction errors more heavily.

**Coefficient of Determination (R²)**

Measures how much of the target variance is explained by the model.

Higher R² and lower MAE/RMSE indicate better predictive performance.

---

## Honest Research Limitations

This project uses a controlled synthetic benchmark rather than real-world EDA datasets or physical circuit measurements.

The power-generation function is intentionally designed to introduce topology-dependent behaviour while keeping aggregate circuit statistics identical within each Isomer Pair.

Therefore:

- The reported performance demonstrates results on a synthetic benchmark, not production-level power estimation.
- The Random Forest baseline is intentionally restricted to five aggregate features.
- The experiment does not establish that GNNs outperform all traditional ML approaches on real circuit designs.
- Generalisation to real circuits requires validation against industry-standard EDA tools and measured or simulated power data.

The purpose of this benchmark is to isolate and demonstrate the potential value of graph structure in circuit-level power prediction.

---

## References

1. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216). NeurIPS.
2. Xu, K., et al. (2019). [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826). ICLR.
3. [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

---

## License

This project is licensed under the MIT License.

---


**Built with Python, PyTorch and PyTorch Geometric.**

*Exploring the intersection of Graph Neural Networks, Digital Circuit Design and Machine Learning.*
