# Domain Generalization & Adaptation Model Contribution Guide

This guide explains how to add new **Domain Generalization (DG)** and **Domain Adaptation (DA)** algorithms to our codebase.

**Core Requirement**: All new implementations must be **Backbone Agnostic**.
- Instead of hardcoding `nn.Linear` layers or `ResNet` layers, you must use the shared backbones defined in `src/backbones.py`.
- Users should be able to switch between `MLP`, `ResNet`, and `Transformer` backbones via the command line argument `--backbone`.

---

## Algorithm Checklist & Rationale

We have selected SOTA algorithms covering different mechanisms of Domain Generalization (DG) and Domain Adaptation (DA).
**Trusted Code Sources**: Please refer to these repositories for implementation details.
1.  **DomainBed**: [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed) (DG Standard)
2.  **Transfer-Learning-Library (TLL)**: [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) (DA Standard)

### Domain Adaptation (DA)
Goal: Access to unlabeled target domain data during training.

**Core Baselines:**
- [x] **DANN (Domain-Adversarial Neural Networks)** (Implemented)
    - **Rationale**: The classic adversarial approach. Aligns feature distributions by confusing a domain discriminator.
    - **Code Source**: Adapted from [shonenkov/DANN-PyTorch](https://github.com/shonenkov/DANN-PyTorch) & TLL.
    - **Paper**: Ganin et al., 2016.
- [x] **DeepCORAL (Correlation Alignment)** (Implemented)
    - **Rationale**: Aligns second-order statistics (covariance) of source and target features. Simple and effective.
    - **Code Source**: Adapted from [SSARCandy/DeepCORAL](https://github.com/SSARCandy/DeepCORAL).
    - **Paper**: Sun et al., 2016.

**Class-wise & Recent Methods:**
*Please implement these by referencing the official TLL repository where possible.*

- [x] **CDAN (Conditional Domain Adversarial Network)** (High Priority) (Implemented)
    - **Rationale**: Condition the domain discriminator on class predictions. Captures multimodal structures crucial for complex shifts.
    - **Code Source**: [thuml/Transfer-Learning-Library/CDAN](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/cdan.py)
    - **Paper**: Long et al., 2018 (NeurIPS).
- [x] **MCC (Minimum Class Confusion)** (Implemented)
    - **Rationale**: Directly minimizes class confusion on the target domain. A non-adversarial, class-wise alignment method.
    - **Code Source**: [thuml/Transfer-Learning-Library/MCC](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/mcc.py)
    - **Paper**: Jin et al., 2020 (ECCV).
- [x] **ADDA (Adversarial Discriminative Domain Adaptation)** (Implemented)
    - **Rationale**: Decouples source and target encoders with GAN loss. (Implemented as Phase 1 Pretraining for Within-Dataset Benchmark).
    - **Code Source**: Adapted from [jvanvugt/pytorch-domain-adaptation](https://github.com/jvanvugt/pytorch-domain-adaptation)
    - **Paper**: Tzeng et al., 2017 (CVPR).
- [x] **JAN (Joint Adaptation Network)** (Implemented)
    - **Rationale**: MMD-based distribution alignment. JAN aligns joint distributions of features and labels using JMMD.
    - **Code Source**: [thuml/Transfer-Learning-Library/DAN](https://github.com/thuml/Transfer-Learning-Library/blob/master/examples/domain_adaptation/image_classification/dan.py)
    - **Paper**: Long et al., 2017 (ICML).
- [x] **MCD (Maximum Classifier Discrepancy)** (Implemented)
    - **Rationale**: Uses two classifiers to align distributions by minimizing their discrepancy on target data.
    - **Code Source**: [mil-tokyo/MCD_DA](https://github.com/mil-tokyo/MCD_DA)
    - **Paper**: Saito et al., 2018 (CVPR).
- [x] **SHOT (Source Hypothesis Transfer)** (Implemented)
    - **Rationale**: Recent SOTA for Source-Free Domain Adaptation (SFDA). Adapts using information maximization and pseudo-labeling.
    - **Code Source**: [tim-learn/SHOT](https://github.com/tim-learn/SHOT)
    - **Paper**: Liang et al., 2020 (ICML).
- [ ] **CBST (Class-Balanced Self-Training)**
    - **Rationale**: Self-training with class balancing to mitigate label shift.
    - **Code Source**: [yzou2/CBST](https://github.com/yzou2/CBST)
    - **Paper**: Zou et al., 2018 (ECCV).

### Domain Generalization (DG)
Goal: Train on source domains to generalize to unseen target domains.

**Implemented (using DomainBed reference):**
*All defined in `src/domainbed_algos.py` following [facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed)*

- [x] **ERM (Empirical Risk Minimization)** (Implemented)
    - **Rationale**: Standard training baseline. Serves as the lower bound for performance.
    - **Code Source**: DomainBed `algorithms.py` (ERM class).
- [x] **IRM (Invariant Risk Minimization)** (Implemented)
    - **Rationale**: Learning invariant features across environments by penalizing gradient norms.
    - **Code Source**: DomainBed `algorithms.py` (IRM class).
- [x] **V-REx (Variance Risk Extrapolation)** (Implemented)
    - **Rationale**: Reduces variance of risks across training domains. robust alternative to IRM.
    - **Code Source**: DomainBed `algorithms.py` (VREx class).
- [x] **GroupDRO (Group Distributionally Robust Optimization)** (Implemented)
    - **Rationale**: Optimizes for the worst-case domain performance. Essential for user heterogeneity.
    - **Code Source**: DomainBed `algorithms.py` (GroupDRO class).
- [x] **MixStyle** (Implemented)
    - **Rationale**: Data augmentation in feature space (mixing statistics). Simple yet effective.
    - **Code Source**: DomainBed `algorithms.py` (MixStyle class).
- [x] **MLDG (Meta-Learning for Domain Generalization)** (Implemented)
    - **Rationale**: Meta-learning to simulate domain shift.
    - **Code Source**: DomainBed `algorithms.py` (MLDG class).
- [x] **MASF (Maximum Mean Discrepancy for DG)** (Implemented)
    - **Rationale**: MMD-based feature alignment for DG.
    - **Code Source**: [douqi/MASF](https://github.com/douqi/MASF)


**Planned DG Methods:**
- [ ] **CSD (Common Specific Decomposition)**
    - **Rationale**: Decomposes features into domain-shared and domain-specific components.
    - **Source**: Piratla et al., 2020 (ICML).
- [ ] **SagNet (Style Agnostic Networks)**
    - **Rationale**: Randomizes style features to focus on content.
    - **Source**: Nam et al., 2021 (CVPR).
- [ ] **Fish (Gradient Matching)**
    - **Rationale**: Aligns gradients across domains.
    - **Source**: Shi et al., 2021 (ICLR).



### Tabular Deep Learning & Transformers
Goal: Modern architectures specifically designed for tabular data, including high-dimensional feature sets.

**Implemented:**
- [x] **TabTransformer** (Implemented)
    - **Rationale**: Transformer encoder for tabular data. Handles categorical embeddings effectively.
    - **Code Source**: `pytorch-widedeep` / Huang et al., 2020.
- [x] **FastFormer** (Implemented)
    - **Rationale**: Efficient Additive Attention ($O(N)$). Crucial for datasets with many features (>100 features).
    - **Code Source**: `pytorch-widedeep` / Wu et al., 2021.
- [x] **Perceiver** (Implemented)
    - **Rationale**: Latent Attention. Decouples compute from input size, enabling handling of very high-dimensional data.
    - **Code Source**: `pytorch-widedeep` / Jaegle et al., 2021 (DeepMind).
- [x] **TabNet** (Implemented)
    - **Rationale**: Attentive interpretable tabular learning.
    - **Code Source**: `pytorch-tabnet` / Arik & Pfister, 2020.
- [x] **NODE (Neural Oblivious Decision Ensembles)** (Implemented)
    - **Rationale**: Deep learning architecture that mimics decision trees.
    - **Code Source**: `pytorch_tabular` / Popov et al., 2019.

---

## 1. Domain Generalization (DG) Algorithm Implementation

All new DG algorithms should be added to `src/domainbed_algos.py`.

### Step 1: Inherit from `DGModel`
Your class must inherit from `DGModel` (or extended from it). The `DGModel` already handles the backbone initialization based on `hparams`.

**Template:**
```python
from src.domainbed_algos import DGModel
import torch.nn.functional as F

class MyNewAlgo(DGModel):
    def __init__(self, input_dim, num_classes=2, hparams=None):
        # 1. Initialize parent (this sets up self.featurizer and self.classifier)
        super(MyNewAlgo, self).__init__(input_dim, num_classes, hparams)
        
        # 2. Add any algorithm-specific parameters/optimizers
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))
        self.my_hyperparam = hparams.get('my_hyperparam', 1.0)

    def update(self, minibatches, unlabeled=None, domain_indices=None):
        """
        Perform one update step.
        minibatches: List of (x, y) tuples, one for each sampled domain.
        """
        # Example logic:
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        # Forward pass (use self.predict or self.network)
        logits = self.predict(all_x)
        loss = F.cross_entropy(logits, all_y)
        
        # ... Add your custom regularization/penalty here ...
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
```

### Key Components:
- `self.featurizer`: The backbone (MLP, ResNet, etc.) initialized by `DGModel`.
- `self.classifier`: The final classification layer.
- `self.network`: Sequential container of `featurizer + classifier`.
- `update()`: The core training step. It receives a list of minibatches from different domains.

---

## 2. Domain Adaptation (DA) Algorithm Implementation

All new DA algorithms (like CDAN, DeepCORAL, etc.) should be added to `src/da_models.py`.

### Step 1: Structure the Model
Unlike DG, we don't have a single `DAModel` base class yet, but you should follow the pattern of `DANN` or reuse the `DGModel` structure if applicable. Ideally, use `src/backbones.py` for the feature extractor.

**Example (Using Shared Backbones):**
```python
import torch.nn as nn
from src.backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer

class MyDAModel(nn.Module):
    def __init__(self, input_dim, num_classes, hparams):
        super(MyDAModel, self).__init__()
        
        # Initialize Backbone manually if not inheriting DGModel
        backbone_name = hparams.get('backbone', 'MLP')
        if backbone_name == 'MLP':
            self.feature_extractor = MLPFeaturizer(input_dim, ...)
        # ... (handle other backbones) ...
        
        self.classifier = nn.Linear(self.feature_extractor.output_dim, num_classes)
        # Add domain discriminator if needed
```

### Step 2: Implement Training Logic
Create a training function (e.g., `train_my_algo`) that handles the source/target data flow. See `train_dann` in `src/da_models.py` for reference.

---

## 3. Registering the Model

Finally, you must register your new model in `execute_benchmark.py` so it can be run via command line.

### Step 1: Import the Class
```python
from src.domainbed_algos import MyNewAlgo
# or
from src.da_models import MyDAModel, train_my_algo
```

### Step 2: Add to Argument Choices
Update the `choices` list for the `--model` argument:
```python
parser.add_argument('--model', ..., choices=[..., 'MyNewAlgo'])
```

### Step 3: Add Initialization Logic
Inside the execution loop (around line 120):
```python
elif args.model == 'MyNewAlgo':
    model = MyNewAlgo(input_dim=input_dim, num_classes=2, hparams={'lr': args.lr, 'backbone': args.backbone})
```

### Step 4: Add Training Logic (if custom)
If your model uses the standard DG loop (`train_dg_model`), add it to the list:
```python
elif args.model in [..., 'MyNewAlgo']:
    model = train_dg_model(model, ...)
```
If it needs a custom loop (like DA often does), call it explicitly:
```python
elif args.model == 'MyDAModel':
    model = train_my_algo(model, ...)
```

---

## Summary Checklist
1.  [ ] **DG**: Add class to `src/domainbed_algos.py` inheriting `DGModel`.
2.  [ ] **DA**: Add class and training function to `src/da_models.py`.
3.  [ ] **Backbone**: Ensure strict usage of `hparams['backbone']` to select `MLP`, `ResNet`, etc.
4.  [ ] **Execution**: Update `execute_benchmark.py` imports, arguments, and training calls.
5.  [ ] **Verify**: Run `python execute_benchmark.py --dataset D-1 --model MyNewAlgo --epochs 1` to test.


---

## Model Limitations & Best Practices

### Tabular Deep Learning Models (SAINT, TabTransformer, etc.)
- Ensure hyperparameters match the specific implementation (e.g., `pytorch-widedeep`).
- Use `n_jobs` for data loading carefully to avoid CPU bottlenecks.
