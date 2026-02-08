import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
from tabpfn import TabPFNClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# Imports for new libraries: Moved to inside wrappers for lazy loading and memory safety
# import pandas as pd # Already imported at top
# from pytorch_widedeep... 
# from pytorch_tabular...
# from deepctr_torch...

from torch.utils.data import DataLoader, TensorDataset
import copy
from typing import Dict, Any, Optional

# --- Baseline Models ---

class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LightGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMClassifier(**kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(10, verbose=0)] if eval_set else None
        self.model.fit(X, y, eval_set=eval_set, eval_metric='binary_logloss', callbacks=callbacks)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        from pytorch_tabnet.tab_model import TabNetClassifier
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model = TabNetClassifier(verbose=0, **self.kwargs)
        self.model.fit(X, y, eval_set=eval_set, patience=10, max_epochs=50) # Hardcoded epochs for now, can be parameterized
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', N_ensemble_configurations=3, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        from tabpfn import TabPFNClassifier
        # TabPFN doesn't use validation set for early stopping, it's a PFN.
        # It handles small datasets well. Subsampling might be needed for large Train.
        # TabPFN assumes smaller datasets (e.g. < 10k samples). 
        # Our Global Train is ~12k samples. It might be slow or hit memory limits.
        # We will try fitting directly.
        if X.shape[0] > 10000:
             # Subsample for TabPFN if too large? 
             # For benchmark strictness, let's try full. If OOM, we'll subsample.
             pass
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# --- Wrappers for Advanced Tabular DL ---

class WidedeepWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='SAINT', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.trainer = None
        self.preprocessor = None
        self.col_names = None

    def fit(self, X, y, X_val=None, y_val=None):
        from pytorch_widedeep.preprocessing import TabPreprocessor
        from pytorch_widedeep.models import TabMlp, TabTransformer, SAINT, WideDeep
        from pytorch_widedeep.training import Trainer as WideTrainer
        from pytorch_widedeep.metrics import Accuracy as WideAccuracy

        # Convert to DataFrame
        self.col_names = [f"col_{i}" for i in range(X.shape[1])]
        df_train = pd.DataFrame(X, columns=self.col_names)
        df_train['target'] = y
        
        # Preprocessing
        self.preprocessor = TabPreprocessor(continuous_cols=self.col_names, scale=False) # Already scaled
        X_tab = self.preprocessor.fit_transform(df_train)
        
        # Validation
        if X_val is not None:
            df_val = pd.DataFrame(X_val, columns=self.col_names)
            df_val['target'] = y_val
            X_val_tab = self.preprocessor.transform(df_val)
        else:
            X_val_tab = None
            df_val = None

        # Define Model
        if self.model_type == 'SAINT':
            # SAINT expects some categorical mostly, but supports continuous.
            # input_dim for continuous is passed in column_idx
            deeptabular = SAINT(column_idx=self.preprocessor.column_idx, continuous_cols=self.col_names, 
                                input_dim=32, n_heads=4, n_blocks=2, **self.kwargs)
        elif self.model_type == 'TabTransformer':
            # Must set embed_continuous=True if only continuous columns
            deeptabular = TabTransformer(column_idx=self.preprocessor.column_idx, continuous_cols=self.col_names, 
                                         input_dim=32, n_heads=4, n_blocks=2, embed_continuous=True, embed_continuous_method='standard', **self.kwargs)
        elif self.model_type == 'FT-Transformer':
            # FT-Transformer logic if needed (SAINT uses mostly same structure in widedeep?)
            # pytorch-widedeep supports FT-Transformer config via SAINT? No, distinct?
            # Actually SAINT is enough.
            pass
            
        model = WideDeep(deeptabular=deeptabular)
        
        # Trainer
        # Check cuda
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.trainer = WideTrainer(model, objective='binary', metrics=[WideAccuracy], device=device, verbose=0)
        
        # Wrap in dictionary for WideDeep
        X_train_dict = {'X_tab': X_tab, 'target': df_train['target'].values}
        X_val_dict = {'X_tab': X_val_tab, 'target': df_val['target'].values} if X_val_tab is not None else None
        
        self.trainer.fit(X_train=X_train_dict, target=None, 
                         X_val=X_val_dict, target_val=None,
                         n_epochs=20, batch_size=63) # Batch size 63 to avoid last_batch=1 (12929 % 64 == 1)
        return self

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        X_tab = self.preprocessor.transform(df)
        return self.trainer.predict(X_tab={'X_tab': X_tab})

    def predict_proba(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        X_tab = self.preprocessor.transform(df)
        return self.trainer.predict_proba(X_tab={'X_tab': X_tab})

class PytorchTabularWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='NODE', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.tabular_model = None
        self.col_names = None

    def fit(self, X, y, X_val=None, y_val=None):
        from pytorch_tabular import TabularModel
        from pytorch_tabular.models import NodeConfig
        from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig

        self.col_names = [f"col_{i}" for i in range(X.shape[1])]
        df_train = pd.DataFrame(X, columns=self.col_names)
        df_train['target'] = y
        
        df_val = None
        if X_val is not None:
            df_val = pd.DataFrame(X_val, columns=self.col_names)
            df_val['target'] = y_val

        # Configs
        data_config = DataConfig(
            target=['target'],
            continuous_cols=self.col_names,
            num_workers=0
        )
        trainer_config = TrainerConfig(
            max_epochs=20,
            batch_size=15, # Batch size 15 to avoid last_batch=1 (12929 % 15 == 14)
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            early_stopping_patience=5
        )
        
        if self.model_type == 'NODE':
            model_config = NodeConfig(
                task='classification',
                num_layers=2,
                num_trees=256, # Drastically reduced
                depth=4,
                **self.kwargs
            )
            
        self.tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=OptimizerConfig(),
            trainer_config=trainer_config
        )
        
        self.tabular_model.fit(train=df_train, validation=df_val)
        return self

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        # Pytorch Tabular predict returns dataframe usually
        pred_df = self.tabular_model.predict(df)
        return pred_df['prediction'].values

    def predict_proba(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        pred_df = self.tabular_model.predict(df)
        # Returns probability for class 1? Check docs. Usually class_0_probability, class_1_probability
        if '1_probability' in pred_df.columns:
             return pred_df[['0_probability', '1_probability']].values
        return pred_df['prediction'].values # Fallback

class DeepCTRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='DCN', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None
        self.col_names = None

    def fit(self, X, y, X_val=None, y_val=None):
        from deepctr_torch.models import DCN
        from deepctr_torch.inputs import DenseFeat

        self.col_names = [f"col_{i}" for i in range(X.shape[1])]
        
        # DeepCTR expects feature columns definition
        self.feature_columns = [DenseFeat(feat, 1) for feat in self.col_names]
        
        # Prepare input dict
        train_model_input = {feat: X[:, i] for i, feat in enumerate(self.col_names)}
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.model_type == 'DCN':
            # DCN-v2 (Deep & Cross Network)
            self.model = DCN(self.feature_columns, self.feature_columns, task='binary', device=device, **self.kwargs)
            
        self.model.compile("adam", "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
        
        # Fit
        # DeepCTR fit handles training loop
        self.model.fit(train_model_input, y, batch_size=64, epochs=20, validation_split=0.0, verbose=0) 
        # Manual validation logic is hard with DeepCTR's fit if passing tensors. 
        # Passing X_val is supported via validation_data?
        if X_val is not None:
            # We skip validation in fit for simplicity or check if validation_data arg exists
            pass
            
        return self

    def predict(self, X):
        test_model_input = {feat: X[:, i] for i, feat in enumerate(self.col_names)}
        pred_prob = self.model.predict(test_model_input, batch_size=64)
        return (pred_prob > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        test_model_input = {feat: X[:, i] for i, feat in enumerate(self.col_names)}
        pred_prob = self.model.predict(test_model_input, batch_size=64)
        # Construct [p0, p1]
        return np.hstack([1-pred_prob, pred_prob])

# --- Deep Learning Models ---

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2)) # Binary classification
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super(ResNetBlock, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout=0.3):
        super(ResNet, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        return self.output(out)

# --- Training Loop for DL ---

def train_torch_model(model, X_train, y_train, X_val, y_val, 
                      epochs=50, batch_size=64, lr=1e-3, patience=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print(f"Early stopping at epoch {epoch}")
                break
                
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, X_test, y_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    is_torch = isinstance(model, nn.Module)
    
    if is_torch:
        model.eval()
        model.to(device)
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
    else:
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro') # Macro F1 for generic checking
    try:
        if probs.shape[1] == 2:
            auroc = roc_auc_score(y_test, probs[:, 1])
        else:
            auroc = roc_auc_score(y_test, probs, multi_class='ovr')
    except:
        auroc = 0.5
        
    return {"Accuracy": acc, "F1": f1, "AUROC": auroc}
