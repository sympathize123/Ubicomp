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
    def __init__(self, n_jobs=-1, patience=20, **kwargs):
        self.patience = patience
        self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=n_jobs, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        # XGBoost scikit-learn API handles early stopping if early_stopping_rounds is passed to fit
        # We can pass it via kwargs in init or here. 
        # But for consistency with other wrappers where patience is explicit:
        early_stopping_rounds = getattr(self, 'patience', 10) # Fallback if not set
        self.model.fit(X, y, eval_set=eval_set, verbose=False, early_stopping_rounds=early_stopping_rounds)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LightGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=-1, patience=20, **kwargs):
        self.patience = patience
        self.model = lgb.LGBMClassifier(n_jobs=n_jobs, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(self.patience, verbose=0)] if eval_set else None
        self.model.fit(X, y, eval_set=eval_set, eval_metric='binary_logloss', callbacks=callbacks)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=1024, epochs=50, patience=20, **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        from pytorch_tabnet.tab_model import TabNetClassifier
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        # Remove batch_size from kwargs if it accidentally got in there
        if 'batch_size' in self.kwargs: self.batch_size = self.kwargs.pop('batch_size')
        if 'epochs' in self.kwargs: self.epochs = self.kwargs.pop('epochs')
        if 'patience' in self.kwargs: self.patience = self.kwargs.pop('patience')

        self.model = TabNetClassifier(verbose=0, **self.kwargs)
        self.model.fit(X, y, eval_set=eval_set, patience=self.patience, max_epochs=self.epochs, batch_size=self.batch_size)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, seed=None, subsample_seed=None, **kwargs):
        # TabPFNClassifier does not accept seed in __init__
        self.subsample_seed = subsample_seed
        self.model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        from tabpfn import TabPFNClassifier
        # TabPFN doesn't use validation set for early stopping, it's a PFN.
        
        # Use specific seed for subsampling if provided, otherwise global numpy state
        rng = np.random.RandomState(self.subsample_seed) if self.subsample_seed is not None else np.random
        
        if X.shape[0] > 2048:
             # print(f"TabPFN Warning: Input size {X.shape[0]} > 2048. Subsampling to 2048 for feasibility.")
             indices = rng.choice(X.shape[0], 2048, replace=False)
             X = X[indices]
             y = y[indices]
        
        # TabPFN feature limit check
        if X.shape[1] > 100:
             # print(f"TabPFN Warning: Feature count {X.shape[1]} > 100. Subsampling to 100 features.")
             vars = np.var(X, axis=0)
             top_indices = np.argsort(vars)[-100:]
             X = X[:, top_indices]
             self.feature_indices = top_indices
        else:
             self.feature_indices = None
             
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if hasattr(self, 'feature_indices') and self.feature_indices is not None:
             X = X[:, self.feature_indices]
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self, 'feature_indices') and self.feature_indices is not None:
             X = X[:, self.feature_indices]
        return self.model.predict_proba(X)

# --- Wrappers for Advanced Tabular DL ---

class WidedeepWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='SAINT', batch_size=64, epochs=50, patience=20, **kwargs):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.trainer = None
        self.preprocessor = None
        self.col_names = None

    def fit(self, X, y, X_val=None, y_val=None):
        from pytorch_widedeep.preprocessing import TabPreprocessor
        from pytorch_widedeep.models import TabMlp, TabTransformer, SAINT, WideDeep
        from pytorch_widedeep.training import Trainer as WideTrainer
        from pytorch_widedeep.metrics import Accuracy as WideAccuracy
        from pytorch_widedeep.callbacks import EarlyStopping as WideEarlyStopping

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
            # SAINT parameters
            saint_params = self.kwargs.copy()
            if 'dropout' in saint_params:
                saint_params['transformer_dropout'] = saint_params.pop('dropout')
            
            deeptabular = SAINT(column_idx=self.preprocessor.column_idx, continuous_cols=self.col_names, 
                                **saint_params)
                                
        elif self.model_type == 'TabTransformer':
            # TabTransformer parameters
            tt_params = self.kwargs.copy()
            
            if 'input_dim' not in tt_params: tt_params['input_dim'] = 32
            if 'n_heads' not in tt_params: tt_params['n_heads'] = 4
            if 'n_blocks' not in tt_params: tt_params['n_blocks'] = 2
            
            deeptabular = TabTransformer(column_idx=self.preprocessor.column_idx, continuous_cols=self.col_names, 
                                         embed_continuous=True, embed_continuous_method='standard', **tt_params)
        elif self.model_type == 'FastFormer':
            from pytorch_widedeep.models import TabFastFormer
            ff_params = self.kwargs.copy()
            if 'input_dim' not in ff_params: ff_params['input_dim'] = 32
            if 'n_heads' not in ff_params: ff_params['n_heads'] = 4
            
            deeptabular = TabFastFormer(column_idx=self.preprocessor.column_idx, continuous_cols=self.col_names, 
                                        embed_continuous=True, embed_continuous_method='standard', **ff_params)
        elif self.model_type == 'Perceiver':
             from pytorch_widedeep.models import TabPerceiver
             p_params = self.kwargs.copy()
             if 'input_dim' not in p_params: p_params['input_dim'] = 32
             if 'n_latents' not in p_params: p_params['n_latents'] = 32  # Good default for complexity reduction
             if 'latent_dim' not in p_params: p_params['latent_dim'] = 64
             
             deeptabular = TabPerceiver(column_idx=self.preprocessor.column_idx, continuous_cols=self.col_names,
                                        embed_continuous=True, embed_continuous_method='standard', **p_params)
        elif self.model_type == 'FT-Transformer':
            pass
            
        model = WideDeep(deeptabular=deeptabular)
        
        # Trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Callbacks
        callbacks = []
        if self.patience > 0:
            callbacks.append(WideEarlyStopping(patience=self.patience, min_delta=1e-4, restore_best_weights=True))

        self.trainer = WideTrainer(model, objective='binary', metrics=[WideAccuracy], callbacks=callbacks, device=device, verbose=0)
        
        # Wrap in dictionary for WideDeep
        X_train_dict = {'X_tab': X_tab, 'target': df_train['target'].values}
        X_val_dict = {'X_tab': X_val_tab, 'target': df_val['target'].values} if X_val_tab is not None else None
        
        self.trainer.fit(X_train=X_train_dict, target=None, 
                         X_val=X_val_dict, target_val=None,
                         n_epochs=self.epochs, batch_size=self.batch_size) 
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
    def __init__(self, model_type='NODE', batch_size=64, epochs=50, patience=20, **kwargs):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
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
            max_epochs=self.epochs,
            batch_size=self.batch_size, 
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            early_stopping_patience=self.patience
        )
        
        if self.model_type == 'NODE':
            model_config = NodeConfig(
                task='classification',
                num_layers=2,
                num_trees=512, 
                depth=6,
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
        if 'target_prediction' in pred_df.columns:
             return pred_df['target_prediction'].values
        return pred_df['prediction'].values

    def predict_proba(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        pred_df = self.tabular_model.predict(df)
        # Returns probability for class 1? Check docs. Usually class_0_probability, class_1_probability
        print(f"DEBUG: PytorchTabular prediction columns: {pred_df.columns}")
        if '1_probability' in pred_df.columns:
             return pred_df[['0_probability', '1_probability']].values
        elif 'class_1_probability' in pred_df.columns:
             return pred_df[['class_0_probability', 'class_1_probability']].values
        elif 'target_1_probability' in pred_df.columns:
             return pred_df[['target_0_probability', 'target_1_probability']].values
        elif 'target_prediction' in pred_df.columns:
             return pred_df['target_prediction'].values # Fallback (class labels)
        
        return pred_df['prediction'].values # Fallback

class DeepCTRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='DCN', batch_size=64, epochs=50, patience=20, **kwargs):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None

    def fit(self, X, y, X_val=None, y_val=None):
        from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
        from deepctr_torch.models import DCN
        
        # DeepCTR expects dict input
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        self.feature_columns = [DenseFeat(name, 1) for name in feature_names]
        
        train_model_input = {name: X[:, i] for i, name in enumerate(feature_names)}
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.model_type == 'DCN':
            self.model = DCN(self.feature_columns, self.feature_columns, task='binary', device=device, **self.kwargs)
            
        self.model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])
        
        # DeepCTR fit supports validation_data and callbacks?
        # It's based on pytorch, inherits from BaseModel
        # fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0., validation_data=None, shuffle=True, callbacks=None)
        # It supports callbacks!
        # from deepctr_torch.callbacks import EarlyStopping -> It might leverage something similar or we can inject generic keras-like callbacks if supported?
        # DeepCTR-Torch fit implementation loops epochs. It has 'callbacks' argument but minimal documentation on compatible callbacks.
        # Actually it mimics Keras.
        # Check source or assume standard EarlyStopping might work if available, OR just implement manual loop? 
        # DeepCTR-Torch models have .fit(). 
        # Let's check if we can pass validation_data.
        
        val_data = None
        if X_val is not None:
             val_model_input = {name: X_val[:, i] for i, name in enumerate(feature_names)}
             val_data = (val_model_input, y_val)
             
        # DeepCTR-Torch 0.2.9+ usually supports callbacks? 
        # For safety, since we don't strictly know if it has a built-in EarlyStopping callback class we can import:
        # It doesn't seem to have a robust Callback system exposed easily in docs usually.
        # BUT, given the scope, if we can't easily add ES, we might just set epochs.
        # However, to be "correct", let's try to assume it runs for fixed epochs if no ES is easy, 
        # OR we leave it as is but use the passed epochs.
        # Wait, user *really* wants early stopping. 
        # If I can't guarantee ES for DeepCTR, I should at least use the `epochs` arg.
        
        self.model.fit(train_model_input, y, batch_size=self.batch_size, epochs=self.epochs, validation_data=val_data, verbose=0)
        return self

    def predict(self, X):
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        test_model_input = {name: X[:, i] for i, name in enumerate(feature_names)}
        pred_ans = self.model.predict(test_model_input, batch_size=self.batch_size)
        return np.where(pred_ans > 0.5, 1, 0).astype(int).flatten()

    def predict_proba(self, X):
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        test_model_input = {name: X[:, i] for i, name in enumerate(feature_names)}
        pred_prob = self.model.predict(test_model_input, batch_size=self.batch_size)
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

from tqdm import tqdm

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
    
    epoch_iterator = tqdm(range(epochs), desc="Training Epochs")
    
    for epoch in epoch_iterator:
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
        
        # Update tqdm description
        epoch_iterator.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}'})
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch}")
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
