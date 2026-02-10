# Stress Detection Benchmark: Domain Generalization & Adaptation

## 📌 Project Overview
This repository hosts a rigorous **Within-Dataset Benchmark** for Stress Detection using wearable physiological data. It integrates state-of-the-art **Domain Generalization (DG)** and **Domain Adaptation (DA)** algorithms, alongside modern **Tabular Deep Learning** models and **Foundation Models**.

The goal is to evaluate if advanced alignment techniques can improve generalization across users (subjects) in physiological stress detection tasks, adhering to strict protocols (NeurIPS/DomainBed standards).

---

## 📂 Project Structure

```
Ubicomp/
├── execute_benchmark.py    # MAIN ENTRY POINT: Run experiments
├── PLANNING.md             # Detailed Roadmap, Rules, and Benchmark Protocol
├── CONTRIBUTING_GUIDE.md   # How to implement new DG/DA models
├── src/
│   ├── models.py           # Tabular DL Models (TabNet, SAINT, NODE, etc.)
│   ├── da_models.py        # DA Models (DANN, CDAN, ADDA, MCD, JAN, SHOT)
│   ├── domainbed_algos.py  # DG Models (IRM, VREx, GroupDRO, MixStyle, etc.)
│   ├── backbones.py        # Shared Backbones (MLP, ResNet, Transformer)
│   ├── data_loader.py      # Data loading & preprocessing
│   └── hparams_registry.py # Hyperparameter definitions
└── archive/                # Deprecated files & old analysis
```

---

## 🚀 Supported Algorithms

We support over 20+ algorithms across three categories. See `CONTRIBUTING_GUIDE.md` for implementation details.

### 1. Domain Adaptation (DA)
*Strategies to align Source (Train) and Target (Test) distributions.*
- **Adversarial**: DANN, CDAN, ADDA
- **Statistical**: DeepCORAL, JAN (JMMD), MCC
- **Source-free**: SHOT
- **Discrepancy**: MCD

### 2. Domain Generalization (DG)
*Learning invariant features across training domains to generalize to unseen domains.*
- **Optimization**: IRM, VREx, GroupDRO
- **Feature Alignment**: MASF, MixStyle
- **Meta-Learning**: MLDG
- **Baseline**: ERM (Empirical Risk Minimization)

### 3. Tabular Deep Learning & Foundation Models
*Modern architectures for tabular data.*
- **Tree-based**: XGBoost, LightGBM
- **Deep Learning**: TabNet, TabTransformer, SAINT, NODE, DeepCTR (DCN-v2)
- **Foundation**: TabPFN (Prior-Data Fitted Network)

---

## 🛠️ Usage

### Environment
Ensure you are in the correct Conda environment:
```bash
conda activate navsim
```

### Running the Benchmark
Use `execute_benchmark.py` to run experiments.

**Basic Command:**
```bash
python execute_benchmark.py --dataset D-1 --model DANN --backbone MLP --epochs 50
```

**Arguments:**
- `--dataset`: `D-1`, `D-2`, `D-3`
- `--model`: Choose from supported models (e.g., `XGB`, `TabNet`, `DANN`, `IRM`, `ADDA`, `SHOT`...)
- `--backbone`: `MLP`, `ResNet`, `Transformer` (for DG/DA models)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 64)

---

## 📚 Documentation & Planning

- **[PLANNING.md](PLANNING.md)**: The central "Source of Truth" for the project roadmap, experimental design, and HPO protocols. **Check this for the current project status.**
- **[CONTRIBUTING_GUIDE.md](CONTRIBUTING_GUIDE.md)**: Detailed guide on adding new algorithms and understanding the code structure.

---

## 📧 Contact & Maintenance
Maintained by Minseo (ICLAB).