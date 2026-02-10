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

#### Domain Generalization (DG) - *Complete*
- [x] **ERM** (Baseline)
- [x] **IRM** (Invariant Risk Minimization)
- [x] **V-REx** (Variance Risk Extrapolation)
- [x] **GroupDRO** (Distributionally Robust Optimization)
- [x] **MixStyle** (Feature Statistics Mixing)
- [x] **MLDG** (Meta-Learning DG)
- [x] **MASF** (MMD-based DG)

#### Domain Adaptation (DA) - *In Progress*
- [x] **DANN** (Domain-Adversarial NN) - *Refactored & Verified*
- [x] **CDAN** (Conditional DANN) - *Implemented & Verified*
- [x] **MCC** (Minimum Class Confusion) - *Implemented & Verified*
- [x] **DeepCORAL** (Correlation Alignment) - *Implemented*
- [ ] **ADDA** (Adversarial Discriminative DA)
- [ ] **DAN / JAN** (MMD-based)
- [ ] **MCD** (Classifier Discrepancy)
- [ ] **SHOT** (Source-Free DA)

### 2.3 Documentation
- [x] `CONTRIBUTING_GUIDE.md`: Detailed instructions for team members to implement remaining algorithms and code sources.

## 3. Next Steps (Roadmap)

### Phase 3: Complete DA Baselines
- **Goal**: Implement high-priority DA methods (ADDA, JAN, SHOT).
- **Action**: Team to follow `CONTRIBUTING_GUIDE.md` to add remaining models to `src/da_models.py`.

### Phase 4: Large-Scale Benchmarking
- **Goal**: Run full factorial experiments (Grid Search or fixed hyperparams).
- **Matrix**:
    - Datasets: D-1, D-2, D-3
    - Models: All DG/DA list
    - Backbones: MLP, ResNet, Transformer
    - Seeds: 3-5 runs
- **Action**: Use `run_benchmark_all.sh` (needs update) to execute batch jobs.

### Phase 5: HPO (Hyperparameter Optimization)
- **Goal**: Optimize `lr`, `dropout`, and algorithm-specific params (`lambda`, `penalty_weight`).
- **Action**: Integrate Optuna or Ray Tune if performance is unsatisfactory with default params.

---
**Legacy Plan (Archived)**
*Previous sections regarding XGBoost pretraining and basic pipeline setup are superseded by the current `src/` based architecture.*

