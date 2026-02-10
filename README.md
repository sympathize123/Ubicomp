# Ubicomp Stress Detection Benchmark

This repository contains the source code for benchmarking **Domain Generalization (DG)** and **Domain Adaptation (DA)** methods on wearable stress detection datasets. The project aims to evaluate model performance under user-based domain shifts using a backbone-agnostic approach.

## 📂 Directory Structure

The codebase has been refactored to a `src/` based architecture for better modularity.

```
Ubicomp/
├── src/
│   ├── models.py           # Standard Baselines (XGB, LGB, MLP, ResNet, TabNet...)
│   ├── da_models.py        # Domain Adaptation Models (DANN, CDAN, MCC, DeepCORAL...)
│   ├── domainbed_algos.py  # Domain Generalization Algorithms (IRM, GroupDRO, MixStyle...)
│   ├── backbones.py        # Shared Feature Extractors (MLP, ResNet, Transformer)
│   └── data_loader.py      # Data Loading & Processing (User-wise Normalization)
├── execute_benchmark.py    # Main Entry Point for Running Experiments
└── run_benchmark_all.sh    # Optimization & Batch Execution Script
```

## 🚀 Supported Algorithms

Detailed descriptions and implementation status can be found in [CONTRIBUTING_GUIDE.md](CONTRIBUTING_GUIDE.md).

### 1. Standard Baselines
- **XGBoost**, **LightGBM** (Tree-based)
- **MLP**, **ResNet** (Deep Tabular)
- *Planned*: TabNet, FT-Transformer, TabPFN

### 2. Domain Generalization (DG)
- **ERM** (Empirical Risk Minimization - Baseline)
- **IRM** (Invariant Risk Minimization)
- **V-REx** (Variance Risk Extrapolation)
- **GroupDRO** (Group Distributionally Robust Optimization)
- **MixStyle** (Domain Mixing for features)
- **MLDG** (Meta-Learning for Domain Generalization)
- **MASF** (Domain-Invariant Feature Learning)

### 3. Domain Adaptation (DA)
- **DANN** (Domain-Adversarial Neural Network)
- **CDAN** (Conditional Adversarial Domain Adaptation)
- **MCC** (Minimum Class Confusion)
- **DeepCORAL** (Correlation Alignment)
- *Planned*: ADDA, MCD, SHOT, JAN

## 🛠 Usage

### Prerequisites
Ensure you have the necessary dependencies installed (PyTorch, Scikit-learn, XGBoost, LightGBM).
```bash
conda activate navsim  # Or your preferred environment
```

### Running a Benchmark
Use `execute_benchmark.py` to run a specific model on a dataset.

**Arguments:**
- `--dataset`: `D-1`, `D-2`, `D-3`
- `--model`: Model name (e.g., `DANN`, `IRM`, `XGB`, `ResNet`)
- `--backbone`: `MLP`, `ResNet`, `Transformer` (for DG/DA models)
- `--epochs`: Number of training epochs (default: 50)

**Example 1: Run DANN with ResNet backbone on D-1**
```bash
python3 execute_benchmark.py --dataset D-1 --model DANN --backbone ResNet --epochs 50
```

**Example 2: Run MCC with MLP backbone**
```bash
python3 execute_benchmark.py --dataset D-1 --model MCC --backbone MLP
```

**Example 3: Run XGBoost Baseline**
```bash
python3 execute_benchmark.py --dataset D-1 --model XGB
```

## 📊 Evaluation Protocol
- **Validation Strategy**: User-based Temporal Split (60% Train / 20% Val / 20% Test).
- **Metric**: Accuracy, F1-Score, AUROC.
- **Seeds**: Experiments are averaged over 3 random seeds (42, 0, 1).

## 📅 Roadmap & Planning
Check [PLANNING.md](PLANNING.md) for the detailed implementation roadmap and future phases (Tabular DL, Foundation Models, HPO).

## 🤝 Contributing
See [CONTRIBUTING_GUIDE.md](CONTRIBUTING_GUIDE.md) for instructions on adding new algorithms or datasets.