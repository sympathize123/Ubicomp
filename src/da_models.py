import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

# --- DANN: Domain Adversarial Neural Network ---

class GradientReversalLayer(autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self, input_dim, num_classes=2, num_domains=2, hidden_dim=256, dropout=0.3):
        super(DANN, self).__init__()
        
        # Feature Extractor (Shared)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Label Classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
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
        
        # Label prediction
        class_output = self.label_classifier(features)
        
        # Domain prediction with Gradient Reversal
        reverse_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        return class_output, domain_output

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

class DeepCORAL(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=256, dropout=0.3):
        super(DeepCORAL, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output, features

# --- Training Logic ---

def train_dann(model, X_train, y_train, d_train, X_val, y_val, d_val,
               epochs=50, batch_size=64, lr=1e-3, patience=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    # Create Datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(d_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), 
        torch.tensor(y_val, dtype=torch.long),
        torch.tensor(d_val, dtype=torch.long) # Not strictly used for selection but good for debug
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Calculate alpha for GRL: 2 / (1 + exp(-10 * p)) - 1
        # p goes from 0 to 1
        p = epoch / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        for X_batch, y_batch, d_batch in train_loader:
            X_batch, y_batch, d_batch = X_batch.to(device), y_batch.to(device), d_batch.to(device)
            
            optimizer.zero_grad()
            
            class_out, domain_out = model(X_batch, alpha=alpha)
            
            err_s_label = class_criterion(class_out, y_batch)
            err_s_domain = domain_criterion(domain_out, d_batch)
            
            # Total loss
            loss = err_s_label + err_s_domain
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, d_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                # DANN validation usually focuses on Task Accuracy
                class_out, _ = model(X_batch, alpha=0)
                loss = class_criterion(class_out, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy() # .copy() for dict? state_dict is dict. copy.deepcopy safe.
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model

