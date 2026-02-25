# UbiComp Experiment Planning & Status

This document tracks the progress of the Cross-Dataset and Within-Dataset Benchmarking for Stress Detection.

## 1. Objectives
- **Benchmark Domain Generalization (DG) & Domain Adaptation (DA)** methods on stress detection datasets (D-1, D-2, D-3).
- **Backbone Agnostic Evaluation**: Compare algorithms using consistent backbones (`MLP`, `ResNet`, `Transformer`).
- **Reproducibility**: Ensure all methods use standard splits and evaluation protocols.

## 2. Current Progress (Updated 2026-02-10)

### 2.1 Codebase Refactoring
- [x] **Directory Structure**: Moved core logic to `src/` (`src/models.py`, `src/da_models.py`, `src/domainbed_algos.py`, `src/backbones.py`).
- [x] **Backbone Standardization**: Implemented shared `MLPFeaturizer`, `ResNetFeaturizer`, `TransformerFeaturizer` in `src/backbones.py`.
- [x] **Execution Pipeline**: Unified execution via `execute_benchmark.py` supporting `--model`, `--dataset`, `--backbone` arguments.

### 2.2 Algorithm Implementation status
We are building a comprehensive benchmark suite. Detailed guide available in `CONTRIBUTING_GUIDE.md`.

#### Standard Baselines - *Complete*
- [x] **XGBoost** (Gradient Boosting)
- [x] **LightGBM** (Gradient Boosting)
- [x] **MLP** (Simple Feed-Forward)
- [x] **ResNet** (ResNet for Tabular Data - RTDL)

#### Domain Generalization (DG) - *Complete*
- [x] **ERM** (Baseline)
- [x] **IRM** (Invariant Risk Minimization)
- [x] **V-REx** (Variance Risk Extrapolation)
- [x] **GroupDRO** (Distributionally Robust Optimization)
- [x] **MixStyle** (Feature Statistics Mixing)
- [x] **MLDG** (Meta-Learning DG)
- [x] **MASF** (MMD-based DG)
- [x] **Fish** (Gradient Matching) - *Implemented & Verified*
- [x] **CSD** (Common Specific Decomposition) - *Implemented & Verified*
- [x] **SagNet** (Style Agnostic Networks) - *Implemented & Verified*

#### Domain Adaptation (DA) - *In Progress*
- [x] **DANN** (Domain-Adversarial NN) - *Refactored & Verified*
- [x] **CDAN** (Conditional DANN) - *Implemented & Verified*
- [x] **MCC** (Minimum Class Confusion) - *Implemented & Verified*
- [x] **DeepCORAL** (Correlation Alignment) - *Implemented*
- [x] **ADDA** (Adversarial Discriminative DA) - *Implemented*
- [x] **JAN** (Joint Adaptation Network) - *Implemented*
- [x] **MCD** (Maximum Classifier Discrepancy) - *Implemented*
- [x] **SHOT** (Source-Free DA) - *Implemented*
- [x] **CBST** (Class-Balanced Self-Training) - *Implemented & Verified*

### 2.3 Documentation
- [x] `CONTRIBUTING_GUIDE.md`: Detailed instructions for team members to implement remaining algorithms and code sources.

## 3. Detailed Implementation Specifics (From Implementation Plan)
*This section details the technical execution steps for the "Within-Dataset" benchmark.*

### 3.1 Protocol & Preprocessing
- **Datasets**: Using "full feature" versions (`stress_binary_personal-full.pkl`) from `CHI/data/Archived`.
- **Split Strategy**: Temporal Split (60% Train, 20% Val, 20% Test) **per user**.
    - Indices are aggregated to form Global Train/Val/Test sets.
- **Normalization**: Z-score normalization **per user** ($z = \frac{x - \mu}{\sigma}$) before concatenation.
- **Filtering**: Drop users with $< N$ samples or significant class imbalance.

### 3.2 Data Pipeline (`src/data_loader.py`)
- **`StressDataset` Class**:
    - **`filter_users()`**: Logic to remove ineligible users.
    - **`normalize_features()`**: User-wise normalization logic.
    - **`get_splits()`**: Temporal splitting logic (Sort by timestamp -> Split -> Aggregate).

### 3.3 Execution Pipeline
- **Script**: `execute_benchmark.py`
    - Dynamic model loading via `--model`.
    - Supports 3 seeds for robust evaluation.
    - Handles initialization for diverse model types (Torch, Sklearn, etc.).
- **Wrapper**: `run_benchmark_all.sh`
    - Automates iteration over the expanded model list and datasets.

### 3.4 Verification Steps
- [ ] Verify new model classes train for 1 epoch on D-1.
- [ ] Check accuracy is better than random (>50%).

## 4. Benchmark Protocol & Experimental Design (Rationale)
*Rationale for Hyperparameter Optimization (HPO) and Model Selection*

To ensure our benchmark meets top-tier conference standards (e.g., NeurIPS Datasets & Benchmarks Track), we adhere to rigorous evaluation protocols.

### 3.1 Why HPO is Mandatory?
- **Avoid "Tuning on Test"**: Fixed hyperparameters often favor methods that happened to be tuned on specific datasets in their original papers. HPO ensures fair comparison by searching for optimal settings for *every* model on *every* dataset.
- **DomainBed Standard**: We follow the protocol defined by **Gulrajani & Lopez-Paz (2021)** in *"In Search of Lost Domain Generalization"*, which demonstrated that many state-of-the-art DG methods are no better than ERM when hyperparameters are fairly tuned.
- **Tabular Data Sensitivity**: Tree-based models (XGB/LGB) and Deep Tabular models (TabNet/FT-Transformer) have vastly different sensitivities. Random search is preferred over grid search for efficiency and better coverage (Bergstra & Bengio, 2012).

### 3.2 Protocol Details
- **Split Strategy**: Train / Validation (ID) / Test (OOD).
    - *Validation (ID)*: Used for HPO and early stopping.
    - *Test (OOD)*: **Never** used for tuning (No "Oracle" selection).
- **Search Strategy**:
    - **Random Search**: 20 trials per model/dataset combination.
    - **Search Space**: Defined in `src/hparams_registry.py` (to be created, mirroring DomainBed's registry).
- **Evaluation Metric**: Average performance across 3 independent seeds.

### 3.3 Benchmark Design Case Studies (Reference for Manuscript)
We model our protocol after these accepted NeurIPS/ICLR benchmarks:

**1. Tabular Deep Learning Benchmark (NeurIPS 2022)**
*   **Paper**: Grinsztajn et al., *"Why do tree-based models still outperform deep learning on typical tabular data?"*
*   **Protocol**:
    *   Used **Random Search** with up to **400 iterations** per dataset.
    *   Tuned **Model Size** explicitly: Number of layers (1-6), Width (64-1024), Dropout (0-0.5).
    *   **Result**: Showed that without this extensive tuning, DL models significantly underperform XGBoost/LightGBM.

**2. Revisiting Deep Learning for Tabular Data (NeurIPS 2021)**
*   **Paper**: Gorishniy et al. (Yandex Research)
*   **Protocol**:
    *   Unified tuning protocol for ResNet, MLP, and Transformer (FT-Transformer).
    *   Used a fixed **Validation Set** for both early stopping and hyperparameter selection.
    *   Demonstrated that "Model Size" (depth/width) is a critical hyperparameter, not a fixed architectural choice.

**3. DomainBed (ICLR 2021)**
*   **Paper**: Gulrajani & Lopez-Paz
*   **Protocol**:
    *   Random Search over 20 distributions of hyperparameters.
    *   Critique: Proved that complex DG algorithms often fail to beat ERM if ERM is well-tuned and DG is not, or vice versa.

### 3.4 References
1.  **DomainBed**: Gulrajani, I., & Lopez-Paz, D. (2021). *In Search of Lost Domain Generalization*. ICLR.
2.  **Tabular Benchmarks**: Grinsztajn, L., et al. (2022). *Why do tree-based models still outperform deep learning on typical tabular data?*. NeurIPS.
3.  **FT-Transformer**: Gorishniy, Y., et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS.

## 4. Next Steps (Roadmap)

### Phase 3: Complete DA Baselines
- **Goal**: Implement high-priority DA methods (ADDA, JAN, SHOT).
- **Action**: Team to follow `CONTRIBUTING_GUIDE.md` to add remaining models to `src/da_models.py`.

### Phase 4: Tabular Deep Learning & Foundation Models (Alignment with Implementation Plan)
*These models are critical baselines for our "Backbone Agnostic" claim.*
- [ ] **Tabular DL Baselines**:
    - **TabNet** (`pytorch-tabnet`)
    - **TabTransformer** (Attention-based)
    - **NODE** (Neural Oblivious Decision Ensembles)
    - **SAINT** (Self-Attention and Intersample Attention)
    - **FT-Transformer** (Feature Tokenizer + Transformer)
- [ ] **Foundation Models**:
    - **TabPFN** (Prior-Data Fitted Network): A transformer pre-trained on synthetic datasets, acting as a foundation model for tabular data.
- **Action**: Integrate these into `src/models.py` wrappers.

### Phase 5: Large-Scale Benchmarking
- **Goal**: Run full factorial experiments (Grid Search or fixed hyperparams).
- **Matrix**:
    - Datasets: D-1, D-2, D-3
    - Models: All DG/DA list + Tabular DL + Foundation Models
    - Backbones: MLP, ResNet, Transformer
    - Seeds: 3-5 runs
- **Action**: Use `run_benchmark_all.sh` (needs update) to execute batch jobs.

### Phase 6: HPO (Hyperparameter Optimization)
- **Goal**: Optimize `lr`, `dropout`, and algorithm-specific params (`lambda`, `penalty_weight`).
- **Action**: Integrate Optuna or Ray Tune if performance is unsatisfactory with default params.

---
**Legacy Plan (Archived)**
*Previous sections regarding XGBoost pretraining and basic pipeline setup are superseded by the current `src/` based architecture.*

