# Domain Generalization & Adaptation Model Contribution Guide

This guide explains how to add new **Domain Generalization (DG)** and **Domain Adaptation (DA)** algorithms to our codebase.

**Core Requirement**: All new implementations must be **Backbone Agnostic**.
- Instead of hardcoding `nn.Linear` layers or `ResNet` layers, you must use the shared backbones defined in `src/backbones.py`.
- Users should be able to switch between `MLP`, `ResNet`, and `Transformer` backbones via the command line argument `--backbone`.

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
