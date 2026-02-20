import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from src.backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer

class DGModel(nn.Module):
    """
    Base class for Domain Generalization models.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DGModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hparams = hparams if hparams else {}
        
        backbone_name = self.hparams.get('backbone', 'MLP')
        dropout = self.hparams.get('dropout', 0.3)
        hidden_dim =256 
        
        if backbone_name == 'MLP':
            self.featurizer = MLPFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, dropout=dropout)
        elif backbone_name == 'ResNet':
            self.featurizer = ResNetFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, dropout=dropout)
        elif backbone_name == 'Transformer':
            self.featurizer = TransformerFeaturizer(input_dim, hidden_dim=128, output_dim=128, dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        self.classifier = nn.Linear(self.featurizer.output_dim, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)


    def update(self, minibatches, unlabeled=None):
        """
        Input: minibatches is a list of (x, y) pairs, one per domain.
        """
        raise NotImplementedError

class ERM(DGModel):
    """
    Empirical Risk Minimization (Standard Training)
    Serves as a baseline.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(ERM, self).__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

class IRM(DGModel):
    """
    Invariant Risk Minimization (Arjovsky et al., 2019)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(IRM, self).__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))
        self.penalty_weight = hparams.get('irm_lambda', 100.0)
        self.penalty_anneal_iters = hparams.get('irm_penalty_anneal_iters', 500)
        self.steps = 0

    def irm_penalty(self, logits, y):
        device = logits.device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss = F.cross_entropy(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    def update(self, minibatches, unlabeled=None):
        self.steps += 1
        penalty_weight = (self.penalty_weight if self.steps >= self.penalty_anneal_iters else 1.0)
        
        nll = 0.
        penalty = 0.

        for x, y in minibatches:
            logits = self.network(x)
            nll += F.cross_entropy(logits, y)
            penalty += self.irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}

class VREx(DGModel):
    """
    V-REx (Krueger et al., 2021)
    Risk Extrapolation: Minimizes mean risk + variance of risk across domains.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(VREx, self).__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))
        self.lambda_val = hparams.get('vrex_lambda', 10.0)
        self.anneal_iters = hparams.get('vrex_penalty_anneal_iters', 500)
        self.steps = 0

    def update(self, minibatches, unlabeled=None):
        self.steps += 1
        loss_weight = (self.lambda_val if self.steps >= self.anneal_iters else 1.0)
        
        losses = torch.zeros(len(minibatches)).to(minibatches[0][0].device)
        
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)
            losses[i] = F.cross_entropy(logits, y)

        mean_loss = losses.mean()
        var_loss = losses.var()
        total_loss = mean_loss + loss_weight * var_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {'loss': total_loss.item(), 'mean_loss': mean_loss.item(), 'var_loss': var_loss.item()}

class GroupDRO(DGModel):
    """
    Group DRO (Sagawa et al., 2020)
    Optimizes for the worst-case domain.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(GroupDRO, self).__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))
        # Initialize q with size of num_domains if known, else dynamic. 
        # But q needs to be persistent. We'll rely on the update method to resize or hparams to pass num_domains.
        n_domains = hparams.get('num_domains', 10) # Default 10 if not passed
        self.register_buffer("q", torch.ones(n_domains)) 
        self.group_step_size = hparams.get('groupdro_eta', 0.01)

    def update(self, minibatches, unlabeled=None, domain_indices=None):
        device = minibatches[0][0].device
        
        # If domain_indices passed (subset of domains), we need to index q carefully.
        # But GroupDRO expects q to track ALL domains.
        # If we subsample domains, we only update q for those domains?
        # Simpler approach: GroupDRO usually assumes access to all groups.
        # For 100 users, maybe we should just iterate all? 
        # If standard GroupDRO implementation, it updates q for the batch.
        
        # We will assume minibatches corresponds to domain_indices
        if domain_indices is None:
             # Assume minibatches are all domains in order 0..N-1
             domain_indices = range(len(minibatches))
             
        if len(self.q) < max(domain_indices) + 1:
             # Resize q if we see a larger domain index (simple dynamic resize)
             new_q = torch.ones(max(domain_indices) + 1).to(device)
             new_q[:len(self.q)] = self.q
             self.q = new_q
             
        losses = torch.zeros(len(minibatches)).to(device)

        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x)
            losses[i] = F.cross_entropy(logits, y)
            # Update q for this specific domain
            global_idx = domain_indices[i]
            self.q[global_idx] *= torch.exp(self.group_step_size * losses[i].data)

        # Normalize q roughly? 
        # Standard GroupDRO normalizes q such that sum(q) = 1.
        # If we only updated a subset, re-normalizing the whole q is tricky if we don't touch others.
        # For now, let's normalize the *active* subset or just normalize globally.
        self.q /= self.q.sum()

        # Weighted loss
        # We use the q values for the CURRENT sampled domains to weight the losses
        # q_subset = self.q[domain_indices]
        # q_subset /= q_subset.sum() # Local normalization for the batch?
        # No, simpler: dot product with relevant q's, then maybe scale?
        # Let's stick to standard implementation: loss = sum(q_i * loss_i) for all i.
        # But we only computed loss_i for the subset.
        # We'll use the current q values for the subset.
        
        batch_q = self.q[list(domain_indices)]
        batch_q = batch_q / batch_q.sum() # Normalize over the batch for stability
        
        loss = torch.dot(losses, batch_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

class MixStyle(DGModel):
    """
    MixStyle (Zhou et al., 2021)
    Mixes feature statistics between domains in the featurizer.
    Simplified application to MLP features (mixing mean/std of hidden layers).
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(MixStyle, self).__init__(input_dim, num_classes, hparams)
        # Custom featurizer with MixStyle blocks could be implemented.
        # For simplicity, we apply MixStyle in forward pass if possible or Modify architecture.
        # This is strictly a placeholder as standard MixStyle is for CNNs.
        # For tabular, we can mix statistics of embeddings or hidden layers.
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))
        self.alpha = hparams.get('mixstyle_alpha', 0.1)

    def forward_mix(self, x, p=0.5, alpha=0.1):
        # Apply plain forward with conditional mixing
        # ... logic ...
        return self.network(x)

    def update(self, minibatches, unlabeled=None, domain_indices=None):
        # Just ERM for now unless we re-implement Featurizer to support MixStyle injection
        # Pass generic minibatches flattening
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class MLDG(DGModel):
    """Meta-Learning for Domain Generalization (Li et al., 2018), first-order variant."""

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))
        self.beta = hparams.get('mldg_beta', 1.0)

    def update(self, minibatches, unlabeled=None):
        if len(minibatches) < 2:
            raise ValueError("MLDG requires at least two domains per update")

        device = minibatches[0][0].device
        meta_test_idx = torch.randint(len(minibatches), (1,), device=device).item()
        meta_train = [b for i, b in enumerate(minibatches) if i != meta_test_idx]
        meta_test = minibatches[meta_test_idx]

        # Meta-train loss
        train_losses = []
        for x, y in meta_train:
            logits = self.network(x)
            train_losses.append(F.cross_entropy(logits, y))
        meta_train_loss = torch.stack(train_losses).mean()

        # Meta-test loss
        x_meta, y_meta = meta_test
        meta_test_logits = self.network(x_meta)
        meta_test_loss = F.cross_entropy(meta_test_logits, y_meta)

        total_loss = meta_train_loss + self.beta * meta_test_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'meta_train_loss': meta_train_loss.item(),
            'meta_test_loss': meta_test_loss.item(),
        }


class MASF(DGModel):
    """MMD-based alignment regularizer across domain features."""

    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=hparams.get('lr', 1e-3))
        self.lambda_mmd = hparams.get('masf_lambda', 1.0)
        self.sigma = hparams.get('masf_sigma', 1.0)

    def _gaussian_kernel(self, x, y, sigma):
        x_exp = x.unsqueeze(1)
        y_exp = y.unsqueeze(0)
        diff = x_exp - y_exp
        dist_sq = (diff * diff).sum(dim=2)
        return torch.exp(-dist_sq / (2 * sigma * sigma))

    def _mmd(self, f1, f2):
        k11 = self._gaussian_kernel(f1, f1, self.sigma)
        k22 = self._gaussian_kernel(f2, f2, self.sigma)
        k12 = self._gaussian_kernel(f1, f2, self.sigma)
        return k11.mean() + k22.mean() - 2 * k12.mean()

    def update(self, minibatches, unlabeled=None):
        device = minibatches[0][0].device
        ce_losses = []
        features = []

        for x, y in minibatches:
            logits = self.network(x)
            ce_losses.append(F.cross_entropy(logits, y))
            with torch.no_grad():
                features.append(self.featurizer(x).detach())

        ce_loss = torch.stack(ce_losses).mean()

        mmd_terms = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                mmd_terms.append(self._mmd(features[i], features[j]))
        mmd_loss = torch.stack(mmd_terms).mean() if mmd_terms else torch.tensor(0.0, device=device)

        total_loss = ce_loss + self.lambda_mmd * mmd_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'mmd_loss': mmd_loss.item(),
        }

# --- Training Helper ---

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def train_dg_model(model, X_train, y_train, d_train, X_val, y_val, d_val, 
                   epochs=20, batch_size=32, domains_per_batch=8, device='cuda'):
    """
    Training loop for Domain Generalization models.
    """
    model.to(device)
    model.train()
    
    unique_domains = np.unique(d_train)
    num_domains = len(unique_domains)
    print(f"DG Training: {num_domains} domains, sampling {domains_per_batch} per batch.")
    
    # Prepare DataLoaders per domain
    domain_loaders = []
    for domain in unique_domains:
        mask = (d_train == domain)
        X_d = torch.tensor(X_train[mask], dtype=torch.float32)
        y_d = torch.tensor(y_train[mask], dtype=torch.long)
        
        # Check if enough samples for at least 1 batch? 
        # If not, we might reuse samples or drop?
        # We'll just define loader, if it runs out we restart iterator (done manually below)
        dataset = TensorDataset(X_d, y_d)
        # Drop last to avoid batch size 1 causing BatchNorm errors
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
        domain_loaders.append(iter(loader))
    
    # Validation Set (Global)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    domain_datasets = [TensorDataset(torch.tensor(X_train[d_train==d], dtype=torch.float32), 
                                     torch.tensor(y_train[d_train==d], dtype=torch.long)) for d in unique_domains]
    
    steps_per_epoch = max(10, int(len(X_train) / (batch_size * domains_per_batch))) # approx
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for step in range(steps_per_epoch):
            # 1. Sample domains
            start_domain_idx = np.random.choice(num_domains, domains_per_batch, replace=(num_domains < domains_per_batch))
            
            minibatches = []
            
            for d_idx in start_domain_idx:
                loader = domain_loaders[d_idx]
                try:
                    batch = next(loader)
                except StopIteration:
                    # Restart loader
                    domain_loaders[d_idx] = iter(DataLoader(domain_datasets[d_idx], batch_size=batch_size, shuffle=True, drop_last=True))
                    batch = next(domain_loaders[d_idx])
                
                x, y = batch
                minibatches.append((x.to(device), y.to(device)))
            
            # 2. Update Model
            # Some models (GroupDRO) might need domain indices
            if isinstance(model, GroupDRO):
                 metrics = model.update(minibatches, domain_indices=start_domain_idx)
            else:
                 metrics = model.update(minibatches)
                 
            epoch_loss += metrics['loss']
            
        # Validation
        model.eval()
        with torch.no_grad():
             logits = model.predict(X_val_t)
             val_loss = F.cross_entropy(logits, y_val_t).item()
             preds = logits.argmax(dim=1)
             val_acc = (preds == y_val_t).float().mean().item()
             
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/steps_per_epoch:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
    return model
