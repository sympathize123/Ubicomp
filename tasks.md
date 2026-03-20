# Project Tasks & Progress

This file tracks the detailed implementation steps for the Stress Detection Benchmark project.
It is synchronized with the high-level roadmap in `PLANNING.md`.

## 1. Environment & Setup
- [x] Check `Ubicomp` directory structure
- [x] Create `PLANNING.md` (Roadmap & HPO Rationale)
- [x] Create `CONTRIBUTING_GUIDE.md` (Dev Guide)
- [x] Clean up deprecated files (`PROGRESS.md`, old `README.md`)

## 2. Model Implementation
### Tabular Deep Learning & Foundation Models
- [x] Install Dependencies (`pytorch-tabnet`, `pytorch-widedeep`, `pytorch-tabular`, `deepctr-torch`)
- [x] Implement Model Wrappers in `src/models.py`
    - [x] TabNet
    - [x] TabTransformer
    - [x] SAINT
    - [x] NODE
    - [x] DCN-v2 (DeepCTR)
- [x] Register Models in `execute_benchmark.py`

### Domain Generalization (DG)
- [x] Verified existing implementations:
    - [x] ERM (Baseline)
    - [x] IRM, VREx, GroupDRO (Optimization)
    - [x] MixStyle, MLDG, MASF (Feature Alignment)

### Domain Adaptation (DA)
- [x] Refactor existing DA models in `src/da_models.py`
    - [x] **DANN** (Domain Adversarial NN)
    - [x] **CDAN** (Conditional DANN)
    - [x] **DeepCORAL** (Correlation Alignment)
    - [x] **MCC** (Minimum Class Confusion)
- [x] Implement Missing DA Models
    - [x] **ADDA** (Adversarial Discriminative DA)
    - [x] **MCD** (Maximum Classifier Discrepancy)
    - [x] **JAN** (Joint Adaptation Network with JMMD)
    - [x] **SHOT** (Source-Free DA)

## 3. Benhcmark Execution (Next Steps)
- [ ] **Implement Remaining Planned Models** (User Request)
    - [ ] **DA**: CBST (Class-Balanced Self-Training)
    - [ ] **DG**: CSD, SagNet, Fish
- [x] **Dry Run**: Verify all models run on D-1 (1 epoch) - *Tabular DL Verified*
- [ ] **Full Benchmark**: Run strictly defined protocols on D-1, D-2, D-3
    - [ ] Phase 1: Tabular DL Baselines
    - [ ] Phase 2: DG Methods
    - [ ] Phase 3: DA Methods
- [x] **HPO Implementation**: Implement `hparams_registry.py` for fair comparison - *Implemented with Optuna integration in execute_benchmark.py*

## 4. Documentation
- [x] Update `README.md` to reflect new benchmark system
- [x] Ensure `CONTRIBUTING_GUIDE.md` matches implementation
