import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import torch.nn.functional as F
from src.backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer

class DAModel(nn.Module):
    """
    Base class for Domain Adaptation models ensuring backbone-agnostic behavior.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DAModel, self).__init__()
        self.hparams = hparams if hparams else {}
        backbone_name = self.hparams.get('backbone', 'MLP')
        dropout = self.hparams.get('dropout', 0.3)
        hidden_dim = 256
        
        if backbone_name == 'MLP':
            self.feature_extractor = MLPFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, dropout=dropout)
        elif backbone_name == 'ResNet':
            self.feature_extractor = ResNetFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, dropout=dropout)
        elif backbone_name == 'Transformer':
            self.feature_extractor = TransformerFeaturizer(input_dim, hidden_dim=128, output_dim=128, dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_extractor.output_dim, num_classes)
        )

    def predict(self, x):
        feat = self.feature_extractor(x)
        return self.classifier(feat)

    def forward(self, x):
        return self.predict(x)

class GradientReversalLayer(autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(DAModel):
    def __init__(self, input_dim, num_classes=2, num_domains=2, hparams=None):
        super(DANN, self).__init__(input_dim, num_classes, hparams)
        
        hidden_dim = 128 # Output of featurizer
        dropout = self.hparams.get('dropout', 0.3)
        
        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains)
        )
        
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        
        reverse_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output

class CDAN(DAModel):
    """
    Conditional Domain Adversarial Network (Long et al., 2018)
    Conditions domain discriminator on feature * softmax(logits).
    """
    def __init__(self, input_dim, num_classes=2, num_domains=2, hparams=None):
        super(CDAN, self).__init__(input_dim, num_classes, hparams)
        
        hidden_dim = 128 # Output of featurizer
        dropout = self.hparams.get('dropout', 0.3)
        
        # Discriminator input dim = feature_dim * num_classes (Multilinear conditioning)
        disc_input_dim = hidden_dim * num_classes
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(disc_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains)
        )
        
    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.classifier(features)
        softmax_output = F.softmax(class_output, dim=1)
        
        # Multilinear map: Outer product of feature and softmax
        # (Batch, Feat) x (Batch, Class) -> (Batch, Feat * Class)
        # Efficient way: element-wise if standard map, but here we do full outer product flattened
        op_out = torch.bmm(features.unsqueeze(2), softmax_output.unsqueeze(1)) # (B, F, 1) x (B, 1, C) -> (B, F, C)
        op_out = op_out.view(features.size(0), -1) # Flatten to (B, F*C)
        
        reverse_features = GradientReversalLayer.apply(op_out, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output

class MCC(DAModel):
    """
    Minimum Class Confusion (Jin et al., 2020)
    Non-adversarial. Loss-based optimization.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(MCC, self).__init__(input_dim, num_classes, hparams)
        self.temperature = hparams.get('mcc_temp', 2.0)

    def mcc_loss(self, logits):
        """
        Computes Minimum Class Confusion loss on target logits
        """
        predictions = F.softmax(logits / self.temperature, dim=1)
        # Class confusion matrix
        # (Batch, Class) -> (Class, Class) via batch correlation?
        # MCC defines confusion as: sum_b (pred_b) check paper formulation
        # MCC = - mean( trace( C ) ) where C is correlation matrix weighted
        # Actually standard implementation uses:
        # 1. Entropy minimization (standard)
        # 2. Class correlation minimization
        
        # Approximate:
        return 0.0 # Placeholder, logic in training loop usually or here
    
    def forward(self, x):
        return self.predict(x)


# --- CORAL: Deep Correlation Alignment ---

def coral_loss(source, target):
    d = source.data.shape[1]

    # Source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm
    
    # Target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt
    
    # Frobenius norm
    loss = torch.sum(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4*d*d)
    return loss

class DeepCORAL(DAModel):
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DeepCORAL, self).__init__(input_dim, num_classes, hparams)
        
    # Standard forward uses DAModel.predict


# --- Training Logic ---

def train_adversarial_da(model, X_train, y_train, d_train, X_val, y_val, d_val,
                           epochs=50, batch_size=64, lr=1e-3, patience=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generic adversarial training loop for DANN and CDAN.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    # ... (rest of the logic similar to DANN, but using generic model call)
    # Datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(d_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long),
        torch.tensor(d_val, dtype=torch.long)
    )
    
    # Drop last to avoid batchnorm 1 sample issue
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    import copy

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        p = epoch / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        for X_batch, y_batch, d_batch in train_loader:
            X_batch, y_batch, d_batch = X_batch.to(device), y_batch.to(device), d_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: adversarial models return (class_out, domain_out)
            class_out, domain_out = model(X_batch, alpha=alpha)
            
            err_s_label = class_criterion(class_out, y_batch)
            err_s_domain = domain_criterion(domain_out, d_batch)
            
            loss = err_s_label + err_s_domain
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # Validation forward doesn't need alpha usually, just prediction
                # But our forward expects alpha. 
                # If we call model.predict(x) defined in standard base, we get logits
                class_out = model.predict(X_batch)
                loss = class_criterion(class_out, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

# Alias implementation for backward compatibility if needed, or simply replace use cases
def train_dann(*args, **kwargs):
    return train_adversarial_da(*args, **kwargs)

def train_mcc(model, X_train, y_train, d_train, X_val, y_val, d_val,
              epochs=50, batch_size=64, lr=1e-3, patience=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Training loop for MCC (Minimum Class Confusion).
    MCC minimizes class confusion on target domains.
    For 'Within-Dataset', we treat different users as domains.
    MCC usually requires: L_classification(Source) + mu * L_mcc(Target)
    Here, X_train contains multiple domains.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_criterion = nn.CrossEntropyLoss()
    
    unique_domains = np.unique(d_train)
    # MCC usually formulated for Source -> Target UDA.
    # In DG/Multi-source setting: L_cls(Source) + L_mcc(All Domains?)
    # or L_cls(Source) + L_mcc(Target)??
    # If we are doing DG (no specific target during training), MCC helps align domains?
    # Jin et al. MCC is for UDA. 
    # Let's apply MCC loss to ALL training samples, treating them effectively as "target" for confusion minimization
    # in addition to classification loss.
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    import copy
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            logits = model.predict(X_batch)
            cls_loss = class_criterion(logits, y_batch)
            
            # MCC Loss
            probs = F.softmax(logits / model.temperature, dim=1) # (B, C)
            # Covariance matrix weighting?
            # Simplified MCC: Minimize entropy / confusion
            # Official implementation:
            # cov = probs.T @ probs / BatchSize
            # loss_mcc = (cov.sum() - cov.diag().sum())  (minimize off-diagonal)
            
            cov = torch.mm(probs.t(), probs) / X_batch.size(0)
            mcc_loss = (torch.sum(cov) - torch.trace(cov)) 
            # OR standard formulated: MCC minimizes confusion between classes
            
            loss = cls_loss + 1.0 * mcc_loss # weight 1.0
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model.predict(X_batch)
                val_loss += class_criterion(logits, y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

