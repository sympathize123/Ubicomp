import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer


# =============================================================================
# Base Domain Adaptation Model
# =============================================================================

class DAModel(nn.Module):
    """
    Base class for Domain Adaptation models with interchangeable backbones.
    """
    def __init__(self, input_dim, num_classes=2, num_domains=2, hparams=None):
        super(DAModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams if hparams else {}
        
        # Backbone selection
        backbone_name = self.hparams.get('backbone', 'MLP')
        dropout = self.hparams.get('dropout', 0.3)
        hidden_dim = self.hparams.get('hidden_dim', 256)
        
        if backbone_name == 'MLP':
            self.featurizer = MLPFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, dropout=dropout)
        elif backbone_name == 'ResNet':
            self.featurizer = ResNetFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, dropout=dropout)
        elif backbone_name == 'Transformer':
            self.featurizer = TransformerFeaturizer(input_dim, hidden_dim=128, output_dim=128, dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        self.backbone_name = backbone_name

    def predict(self, x):
        """Forward pass for prediction. To be implemented by subclasses."""
        raise NotImplementedError

    def forward(self, x, *args, **kwargs):
        """Forward pass."""
        return self.predict(x, *args, **kwargs)

    def update(self, source_batch, target_batch, unlabeled=None):
        """
        Perform one update step given source and target batches.
        
        Args:
            source_batch: Tuple of (x_source, y_source, d_source) - labeled source data with domain labels
            target_batch: Tuple of (x_target, y_target) or (x_target,) - target data (may be unlabeled)
            unlabeled: Optional additional unlabeled data
            
        Returns:
            dict: Metrics from this update step
        """
        raise NotImplementedError


# =============================================================================
# DANN: Domain Adversarial Neural Network
# =============================================================================

class GradientReversalLayer(autograd.Function):
    """Gradient Reversal Layer for DANN."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(DAModel):
    """
    Domain Adversarial Neural Network (Ganin et al., 2016).
    Uses adversarial training to learn domain-invariant features.
    """
    def __init__(self, input_dim, num_classes=2, num_domains=2, hparams=None):
        super(DANN, self).__init__(input_dim, num_classes, num_domains, hparams)
        
        dropout = self.hparams.get('dropout', 0.3)
        
        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(self.featurizer.output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.featurizer.output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_domains)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.get('lr', 1e-3) if self.hparams else 1e-3
        )
        
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.current_alpha = 1.0

    def predict(self, x, alpha=1.0):
        """Forward pass returning class predictions."""
        features = self.featurizer(x)
        class_output = self.label_classifier(features)
        return class_output

    def forward(self, x, alpha=1.0):
        """Full forward pass with domain prediction."""
        features = self.featurizer(x)
        
        # Label prediction
        class_output = self.label_classifier(features)
        
        # Domain prediction with Gradient Reversal
        reverse_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output

    def update(self, source_batch, target_batch, unlabeled=None):
        """
        Update step for DANN.
        
        Args:
            source_batch: Tuple of (x_source, y_source, d_source)
            target_batch: Tuple of (x_target, d_target) - target domain labels for domain classification
            unlabeled: Not used
            
        Returns:
            dict: Metrics {'loss', 'class_loss', 'domain_loss'}
        """
        x_source, y_source, d_source = source_batch
        x_target, d_target = target_batch
        
        # Combine source and target for domain classification
        x_combined = torch.cat([x_source, x_target], dim=0)
        d_combined = torch.cat([d_source, d_target], dim=0)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        class_out, domain_out = self.forward(x_combined, alpha=self.current_alpha)
        
        # Split outputs
        class_out_source = class_out[:len(x_source)]
        domain_out_all = domain_out
        
        # Compute losses
        class_loss = self.class_criterion(class_out_source, y_source)
        domain_loss = self.domain_criterion(domain_out_all, d_combined)
        
        total_loss = class_loss + domain_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'class_loss': class_loss.item(),
            'domain_loss': domain_loss.item()
        }


# =============================================================================
# DeepCORAL: Deep Correlation Alignment
# =============================================================================

def coral_loss(source_features, target_features):
    """
    Compute CORAL loss between source and target feature distributions.
    
    Args:
        source_features: Features from source domain (batch_size, feature_dim)
        target_features: Features from target domain (batch_size, feature_dim)
        
    Returns:
        torch.Tensor: CORAL loss value
    """
    d = source_features.size(1)
    
    # Source covariance
    xm = torch.mean(source_features, 0, keepdim=True) - source_features
    xc = xm.t() @ xm
    
    # Target covariance
    xmt = torch.mean(target_features, 0, keepdim=True) - target_features
    xct = xmt.t() @ xmt
    
    # Frobenius norm
    loss = torch.sum(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)
    
    return loss


class DeepCORAL(DAModel):
    """
    Deep CORAL: Correlation Alignment for Deep Domain Adaptation (Sun & Saenko, 2016).
    Minimizes feature covariance difference between source and target domains.
    """
    def __init__(self, input_dim, num_classes=2, num_domains=2, hparams=None):
        super(DeepCORAL, self).__init__(input_dim, num_classes, num_domains, hparams)
        
        dropout = self.hparams.get('dropout', 0.3)
        self.coral_lambda = self.hparams.get('coral_lambda', 1.0)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.featurizer.output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.get('lr', 1e-3) if self.hparams else 1e-3
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def predict(self, x):
        """Forward pass for prediction."""
        features = self.featurizer(x)
        output = self.classifier(features)
        return output

    def forward(self, x):
        """Forward pass returning both predictions and features."""
        features = self.featurizer(x)
        output = self.classifier(features)
        return output, features

    def update(self, source_batch, target_batch, unlabeled=None):
        """
        Update step for DeepCORAL.
        
        Args:
            source_batch: Tuple of (x_source, y_source, d_source) - d_source not used
            target_batch: Tuple of (x_target,) or (x_target, d_target) - target features only
            unlabeled: Not used
            
        Returns:
            dict: Metrics {'loss', 'class_loss', 'coral_loss'}
        """
        x_source, y_source, _ = source_batch
        x_target = target_batch[0] if isinstance(target_batch, tuple) else target_batch
        
        self.optimizer.zero_grad()
        
        # Forward pass
        source_out, source_features = self.forward(x_source)
        _, target_features = self.forward(x_target)
        
        # Classification loss on source
        class_loss = self.criterion(source_out, y_source)
        
        # CORAL loss between source and target features
        coral = coral_loss(source_features, target_features)
        
        # Total loss
        total_loss = class_loss + self.coral_lambda * coral
        
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'class_loss': class_loss.item(),
            'coral_loss': coral.item()
        }


# =============================================================================
# Training Helper Functions
# =============================================================================

def train_da_model(model, X_source, y_source, d_source, X_target, y_target=None, d_target=None,
                   X_val=None, y_val=None, epochs=50, batch_size=64, patience=5,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Training loop for Domain Adaptation models.
    
    Args:
        model: DAModel instance (DANN or DeepCORAL)
        X_source: Source domain features (numpy array)
        y_source: Source domain labels (numpy array)
        d_source: Source domain indices (numpy array)
        X_target: Target domain features (numpy array)
        y_target: Optional target domain labels for validation (numpy array)
        d_target: Target domain indices for domain classification (numpy array)
        X_val: Optional validation features
        y_val: Optional validation labels
        epochs: Number of training epochs
        batch_size: Batch size
        patience: Early stopping patience
        device: Device to train on
        
    Returns:
        Trained model
    """
    model = model.to(device)
    model.train()
    
    # Create datasets
    source_dataset = TensorDataset(
        torch.tensor(X_source, dtype=torch.float32),
        torch.tensor(y_source, dtype=torch.long),
        torch.tensor(d_source, dtype=torch.long)
    )
    
    if d_target is not None:
        target_dataset = TensorDataset(
            torch.tensor(X_target, dtype=torch.float32),
            torch.tensor(d_target, dtype=torch.long)
        )
    else:
        target_dataset = TensorDataset(
            torch.tensor(X_target, dtype=torch.float32)
        )
    
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Validation setup
    X_val_t = None
    y_val_t = None
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    
    best_val_loss = float('inf') if X_val is not None and y_val is not None else None
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_metrics = {'loss': 0.0, 'class_loss': 0.0}
        
        # Update alpha for DANN (progressive domain confusion)
        if isinstance(model, DANN):
            p = epoch / epochs
            model.current_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        # Iterate over source and target loaders
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        num_batches = min(len(source_loader), len(target_loader))
        
        for _ in range(num_batches):
            try:
                source_batch = next(source_iter)
                target_batch = next(target_iter)
            except StopIteration:
                break
            
            # Move to device
            source_batch = tuple(t.to(device) for t in source_batch)
            target_batch = tuple(t.to(device) for t in target_batch)
            
            # Update model
            metrics = model.update(source_batch, target_batch)
            
            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Validation
        if X_val_t is not None and y_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_out = model.predict(X_val_t)
                val_loss = F.cross_entropy(val_out, y_val_t).item()
                val_preds = val_out.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_metrics['loss']:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if best_val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                import copy
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_metrics['loss']:.4f} - "
                  f"Class Loss: {epoch_metrics.get('class_loss', 0):.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


# =============================================================================
# Backward Compatibility: Legacy Training Function
# =============================================================================

def train_dann_legacy(model, X_train, y_train, d_train, X_val, y_val, d_val,
                      epochs=50, batch_size=64, lr=1e-3, patience=5,
                      device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Legacy training function for backward compatibility.
    Uses the new train_da_model internally.
    
    Note: This assumes single-domain source and single-domain target.
    For multi-domain scenarios, use train_da_model directly.
    """
    # Split training data into source and target (assuming first half is source, second is target)
    # This is a simplified assumption - adjust based on your data
    n_samples = len(X_train)
    n_source = n_samples // 2
    
    X_source = X_train[:n_source]
    y_source = y_train[:n_source]
    d_source = d_train[:n_source]
    
    X_target = X_train[n_source:]
    d_target = d_train[n_source:]
    
    return train_da_model(
        model, X_source, y_source, d_source, X_target, 
        d_target=d_target, X_val=X_val, y_val=y_val,
        epochs=epochs, batch_size=batch_size, patience=patience, device=device
    )
