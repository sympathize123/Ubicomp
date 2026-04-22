import numpy as np
import pandas as pd
import inspect
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from typing import Dict, Any, Optional
from tqdm import tqdm
from tensorflow.python.keras.callbacks import Callback

FIXED_BATCH_SIZE = 16

# Imports for new libraries: Moved to inside wrappers for lazy loading and memory safety
# import pandas as pd # Already imported at top
# from pytorch_widedeep... 
# from pytorch_tabular...
# from deepctr_torch...


def attach_training_metadata(model, **updates):
    info = dict(getattr(model, "_training_info", {}))
    for key, value in updates.items():
        if value is None and key in info:
            continue
        info[key] = value
    setattr(model, "_training_info", info)
    return model


def _drop_optuna_helper_params(kwargs):
    return {
        key: value
        for key, value in kwargs.items()
        if not (
            key.endswith("_is_zero")
            or key.endswith("_log2")
            or key.endswith("_log10")
        )
    }


def _filter_supported_kwargs(callable_obj, kwargs):
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)

    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


# --- Baseline Models ---

class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=-1, patience=20, **kwargs):
        import xgboost as xgb
        self.patience = patience
        # Force CPU by default to avoid CUDA errors in some environments.
        # Users can still override by passing tree_method explicitly.
        kwargs.setdefault("tree_method", "hist")
        # XGBoost 2.0+ requires early_stopping_rounds in constructor
        # use_label_encoder is deprecated/removed in 3.0+
        self.model = xgb.XGBClassifier(eval_metric='auc', n_jobs=n_jobs, early_stopping_rounds=patience, **kwargs)

    def fit(self, X, y, X_val=None, y_val=None):
        eval_set = [(X_val, y_val)] if X_val is not None else None
        # XGBoost scikit-learn API handles early stopping if early_stopping_rounds is passed to constructor
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        best_iteration = getattr(self.model, "best_iteration", None)
        n_estimators = getattr(self.model, "n_estimators", None)
        attach_training_metadata(
            self,
            optimizer="xgboost",
            best_epoch=(best_iteration + 1) if best_iteration is not None else None,
            epochs_ran=(best_iteration + 1) if best_iteration is not None else n_estimators,
            max_epochs=n_estimators,
            early_stopped=bool(best_iteration is not None and n_estimators is not None and best_iteration + 1 < n_estimators),
            model_selection_metric="val_auroc",
        )
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
        X_train = X.values if hasattr(X, "values") else X
        y_train = y.values if hasattr(y, "values") else y
        X_valid = X_val.values if (X_val is not None and hasattr(X_val, "values")) else X_val
        y_valid = y_val.values if (y_val is not None and hasattr(y_val, "values")) else y_val

        eval_set = [(X_valid, y_valid)] if X_valid is not None else None
        callbacks = [lgb.early_stopping(self.patience, verbose=True)] if eval_set else None
        self.model.fit(X_train, y_train, eval_set=eval_set, eval_metric='auc', callbacks=callbacks)
        best_iteration = getattr(self.model, "best_iteration_", None)
        n_estimators = getattr(self.model, "n_estimators", None)
        attach_training_metadata(
            self,
            optimizer="lightgbm",
            best_epoch=best_iteration,
            epochs_ran=best_iteration or n_estimators,
            max_epochs=n_estimators,
            early_stopped=bool(best_iteration is not None and n_estimators is not None and best_iteration < n_estimators),
            model_selection_metric="val_auroc",
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=FIXED_BATCH_SIZE, epochs=50, patience=20, **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        from pytorch_tabnet.tab_model import TabNetClassifier
        params = _drop_optuna_helper_params(self.kwargs)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.batch_size = int(self.batch_size or FIXED_BATCH_SIZE)
        
        # Remove batch_size from kwargs if it accidentally got in there
        if 'batch_size' in params: self.batch_size = params.pop('batch_size')
        if 'epochs' in params: self.epochs = params.pop('epochs')
        if 'patience' in params: self.patience = params.pop('patience')

        if 'verbose' in params: verbose = params.pop('verbose')
        else: verbose = 1

        if 'n_d' in params and int(params['n_d']) > 32:
            print(f"[INFO] TabNet: capped n_d from {params['n_d']} to 32.")
            params['n_d'] = 32
        if 'n_a' in params and int(params['n_a']) > 32:
            print(f"[INFO] TabNet: capped n_a from {params['n_a']} to 32.")
            params['n_a'] = 32
        if 'n_steps' in params and int(params['n_steps']) > 6:
            print(f"[INFO] TabNet: capped n_steps from {params['n_steps']} to 6.")
            params['n_steps'] = 6

        tabnet_params = _filter_supported_kwargs(TabNetClassifier.__init__, params)
        self.model = TabNetClassifier(verbose=verbose, **tabnet_params)
        self.model.fit(X, y, eval_set=eval_set, eval_metric=['auc'], patience=self.patience, 
                       max_epochs=self.epochs, batch_size=self.batch_size, num_workers=0)
        best_epoch = getattr(self.model, "best_epoch", None)
        attach_training_metadata(
            self,
            optimizer="Adam",
            best_epoch=(best_epoch + 1) if isinstance(best_epoch, int) else best_epoch,
            epochs_ran=(best_epoch + 1) if isinstance(best_epoch, int) else self.epochs,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            early_stopped=bool(best_epoch is not None and isinstance(best_epoch, int) and best_epoch + 1 < self.epochs),
            model_selection_metric="val_auroc",
            architecture_params=tabnet_params,
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# --- Wrappers for Advanced Tabular DL ---

class WidedeepWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='SAINT', batch_size=FIXED_BATCH_SIZE, epochs=50, patience=20, efficient_attention=False, **kwargs):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.efficient_attention = efficient_attention
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
        from torchmetrics.classification import BinaryAUROC
        training_params = _drop_optuna_helper_params(self.kwargs)
        train_lr = float(training_params.get("lr", 1e-3))
        train_weight_decay = float(training_params.get("weight_decay", 0.0) or 0.0)
        effective_model_params = {}
        self.batch_size = int(self.batch_size or FIXED_BATCH_SIZE)

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

        def _cap_transformer_params(params, model_name, max_input_dim=32, max_heads=4, max_blocks=3,
                                    token_element_budget=24_000_000):
            params.setdefault('input_dim', max_input_dim)
            params.setdefault('n_heads', min(4, max_heads))
            params.setdefault('n_blocks', 2)

            input_dim = min(int(params.get('input_dim', max_input_dim)), max_input_dim)
            n_heads = min(max(1, int(params.get('n_heads', 4))), max_heads, input_dim)
            n_blocks = min(max(1, int(params.get('n_blocks', 2))), max_blocks)

            if input_dim % n_heads != 0:
                input_dim = max(n_heads, (input_dim // n_heads) * n_heads)

            n_features = max(1, len(self.col_names))
            max_dim = max(n_heads, token_element_budget // max(1, n_features * max(1, self.batch_size)))
            max_dim = max(n_heads, (max_dim // n_heads) * n_heads)
            if input_dim > max_dim:
                print(
                    f"[INFO] {model_name}: capped input_dim from {params.get('input_dim')} to {max_dim} "
                    f"for {n_features} features and batch_size={self.batch_size}."
                )
                input_dim = max_dim

            max_batch = max(1, token_element_budget // max(1, n_features * max(1, input_dim)))
            if self.batch_size > max_batch:
                adjusted_batch = next((candidate for candidate in (64, 32, 16, 8) if candidate <= max_batch), 8)
                print(
                    f"[INFO] {model_name}: capped batch_size from {self.batch_size} to {adjusted_batch} "
                    f"for {n_features} features and input_dim={input_dim}."
                )
                self.batch_size = adjusted_batch

            params['input_dim'] = input_dim
            params['n_heads'] = n_heads
            params['n_blocks'] = n_blocks

        # Define Model
        if self.model_type == 'SAINT':
            saint_params = training_params.copy()
            _TRAIN_KEYS = {'lr', 'weight_decay', 'batch_size', 'mlp_dropout', 'num_layers', 'hidden_dim', 'transformer_dropout'}
            for k in _TRAIN_KEYS:
                saint_params.pop(k, None)
            if 'dropout' in saint_params:
                dropout_val = saint_params.pop('dropout')
                saint_params.setdefault('attn_dropout', dropout_val)
                saint_params.setdefault('ff_dropout', dropout_val)
            _cap_transformer_params(saint_params, 'SAINT')
            if self.efficient_attention:
                print("[INFO] SAINT: enabling linear-attention path for SAINT encoder blocks.")
                import einops
                import pytorch_widedeep.models.tabular.transformers.saint as saint_mod
                from pytorch_widedeep.models.tabular.transformers._attention_layers import AddNorm, FeedForward, MultiHeadedAttention

                if not getattr(saint_mod, "_ubicomp_linear_saint_patch", False):
                    class EfficientSaintEncoder(nn.Module):
                        def __init__(
                            self,
                            input_dim: int,
                            n_heads: int,
                            use_bias: bool,
                            attn_dropout: float,
                            ff_dropout: float,
                            ff_factor: int,
                            activation: str,
                            n_feat: int,
                        ):
                            super().__init__()
                            self.n_feat = n_feat

                            self.col_attn = MultiHeadedAttention(
                                input_dim,
                                n_heads,
                                use_bias,
                                attn_dropout,
                                None,
                                True,   # use_linear_attention
                                False,  # use_flash_attention
                            )
                            self.col_attn_ff = FeedForward(input_dim, ff_dropout, ff_factor, activation)
                            self.col_attn_addnorm = AddNorm(input_dim, attn_dropout)
                            self.col_attn_ff_addnorm = AddNorm(input_dim, ff_dropout)

                            # Row attention over samples per feature token (keeps SAINT row-attention intent,
                            # avoids flattening n_feat * input_dim into huge projection matrices).
                            self.row_attn = MultiHeadedAttention(
                                input_dim,
                                n_heads,
                                use_bias,
                                attn_dropout,
                                None,
                                True,   # use_linear_attention
                                False,  # use_flash_attention
                            )
                            self.row_attn_ff = FeedForward(input_dim, ff_dropout, ff_factor, activation)
                            self.row_attn_addnorm = AddNorm(input_dim, attn_dropout)
                            self.row_attn_ff_addnorm = AddNorm(input_dim, ff_dropout)

                        def forward(self, X):
                            x = self.col_attn_addnorm(X, self.col_attn)
                            x = self.col_attn_ff_addnorm(x, self.col_attn_ff)
                            x = einops.rearrange(x, "b n d -> n b d")
                            x = self.row_attn_addnorm(x, self.row_attn)
                            x = self.row_attn_ff_addnorm(x, self.row_attn_ff)
                            x = einops.rearrange(x, "n b d -> b n d")
                            return x

                    saint_mod.SaintEncoder = EfficientSaintEncoder
                    saint_mod._ubicomp_linear_saint_patch = True

            saint_params = _filter_supported_kwargs(SAINT.__init__, saint_params)
            effective_model_params = dict(saint_params)
            deeptabular = SAINT(
                column_idx=self.preprocessor.column_idx,
                continuous_cols=self.col_names,
                **saint_params,
            )

        elif self.model_type == 'TabTransformer':
            tt_params = training_params.copy()
            _TRAIN_KEYS = {'lr', 'weight_decay', 'batch_size', 'mlp_dropout', 'num_layers', 'hidden_dim',
                           'transformer_dropout', 'miscellaneous_dropout'}
            for k in _TRAIN_KEYS:
                tt_params.pop(k, None)
            if 'dropout' in tt_params:
                dropout_val = tt_params.pop('dropout')
                tt_params.setdefault('attn_dropout', dropout_val)
                tt_params.setdefault('ff_dropout', dropout_val)
            tt_params.setdefault('input_dim', 32)
            tt_params.setdefault('n_heads', 4)
            tt_params.setdefault('n_blocks', 2)
            _cap_transformer_params(tt_params, 'TabTransformer')
            tt_params['use_linear_attention'] = bool(self.efficient_attention)
            if self.efficient_attention:
                print("[INFO] TabTransformer: use_linear_attention=True (architecture preserved).")
            tt_params = _filter_supported_kwargs(TabTransformer.__init__, tt_params)
            effective_model_params = dict(tt_params)
            deeptabular = TabTransformer(
                column_idx=self.preprocessor.column_idx,
                continuous_cols=self.col_names,
                embed_continuous_method='standard',
                **tt_params,
            )

        elif self.model_type == 'FTTransformer':
            from pytorch_widedeep.models import FTTransformer
            ft_params = training_params.copy()
            _TRAIN_KEYS = {'lr', 'weight_decay', 'batch_size', 'mlp_dropout', 'num_layers', 'hidden_dim'}
            for k in _TRAIN_KEYS:
                ft_params.pop(k, None)
            if 'dropout' in ft_params:
                dropout_val = ft_params.pop('dropout')
                ft_params.setdefault('attn_dropout', dropout_val)
                ft_params.setdefault('ff_dropout', dropout_val)
            ft_params.setdefault('input_dim', 32)
            ft_params.setdefault('n_heads', 4)
            if int(ft_params.get('n_blocks', 2)) > 3:
                print(f"[INFO] FTTransformer: capped n_blocks from {ft_params.get('n_blocks')} to 3.")
                ft_params['n_blocks'] = 3
            # Multi-head attention requires the embedding width to be divisible by the number of heads.
            # Coerce misconfigured HPO suggestions to the nearest valid multiple instead of failing late.
            input_dim = int(ft_params.get('input_dim', 32))
            n_heads = int(ft_params.get('n_heads', 4))
            if n_heads <= 0:
                n_heads = 4
            if input_dim % n_heads != 0:
                adjusted = max(n_heads, int(round(input_dim / n_heads)) * n_heads)
                if adjusted != input_dim:
                    print(f"[INFO] FTTransformer: adjusted input_dim from {input_dim} to {adjusted} for n_heads={n_heads}.")
                ft_params['input_dim'] = adjusted
                ft_params['n_heads'] = n_heads
                input_dim = adjusted

            max_ft_input_dim = 64
            if input_dim > max_ft_input_dim:
                adjusted = max(n_heads, (max_ft_input_dim // n_heads) * n_heads)
                print(f"[INFO] FTTransformer: capped input_dim from {input_dim} to {adjusted}.")
                ft_params['input_dim'] = adjusted
                input_dim = adjusted

            n_features = max(1, len(self.col_names))
            token_element_budget = 36_000_000
            max_dim = max(n_heads, token_element_budget // max(1, n_features * max(1, self.batch_size)))
            max_dim = max(n_heads, (max_dim // n_heads) * n_heads)
            if input_dim > max_dim:
                print(
                    f"[INFO] FTTransformer: capped input_dim from {input_dim} to {max_dim} "
                    f"for {n_features} features and batch_size={self.batch_size}."
                )
                ft_params['input_dim'] = max_dim
                input_dim = max_dim

            max_batch = max(1, token_element_budget // max(1, n_features * max(1, input_dim)))
            if self.batch_size > max_batch:
                adjusted_batch = next((candidate for candidate in (64, 32, 16, 8) if candidate <= max_batch), 8)
                print(
                    f"[INFO] FTTransformer: capped batch_size from {self.batch_size} to {adjusted_batch} "
                    f"for {n_features} features and input_dim={input_dim}."
                )
                self.batch_size = adjusted_batch
            if self.efficient_attention:
                print("[INFO] FTTransformer: enabling kernel linear-attention path in FT encoder blocks.")
                import pytorch_widedeep.models.tabular.transformers.ft_transformer as ft_mod
                from pytorch_widedeep.models.tabular.transformers._attention_layers import NormAdd, FeedForward, MultiHeadedAttention

                if not getattr(ft_mod, "_ubicomp_linear_ft_patch", False):
                    class EfficientFTTransformerEncoder(nn.Module):
                        def __init__(
                            self,
                            input_dim: int,
                            n_feats: int,
                            n_heads: int,
                            use_bias: bool,
                            attn_dropout: float,
                            ff_dropout: float,
                            ff_factor: float,
                            kv_compression_factor: float,
                            kv_sharing: bool,
                            activation: str,
                            first_block: bool,
                        ):
                            super().__init__()
                            self.first_block = first_block
                            self.attn = MultiHeadedAttention(
                                input_dim,
                                n_heads,
                                use_bias,
                                attn_dropout,
                                None,
                                True,   # use_linear_attention
                                False,  # use_flash_attention
                            )
                            self.ff = FeedForward(input_dim, ff_dropout, ff_factor, activation)
                            self.attn_normadd = NormAdd(input_dim, attn_dropout)
                            self.ff_normadd = NormAdd(input_dim, ff_dropout)

                        def forward(self, X):
                            if self.first_block:
                                x = X + self.attn(X)
                            else:
                                x = self.attn_normadd(X, self.attn)
                            return self.ff_normadd(x, self.ff)

                    ft_mod.FTTransformerEncoder = EfficientFTTransformerEncoder
                    ft_mod._ubicomp_linear_ft_patch = True

            ft_params = _filter_supported_kwargs(FTTransformer.__init__, ft_params)
            effective_model_params = dict(ft_params)
            deeptabular = FTTransformer(
                column_idx=self.preprocessor.column_idx,
                continuous_cols=self.col_names,
                embed_continuous_method='standard',
                **ft_params,
            )
            
        model = WideDeep(deeptabular=deeptabular)
        
        # Trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Callbacks
        callbacks = []
        # Use ROCAUC for early stopping if possible, or Accuracy
        from pytorch_widedeep.metrics import Accuracy
        
        if self.patience > 0:
            # Torchmetrics names are logged as val_<ClassName>, e.g. val_BinaryAUROC.
            callbacks.append(
                WideEarlyStopping(
                    monitor='val_BinaryAUROC',
                    mode='max',
                    patience=self.patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                )
            )

        # Keep Accuracy for visibility and add BinaryAUROC so callbacks can monitor val_BinaryAUROC.
        self.trainer = WideTrainer(
            model,
            objective='binary',
            optimizers=torch.optim.Adam(model.parameters(), lr=train_lr, weight_decay=train_weight_decay),
            metrics=[Accuracy, BinaryAUROC()],
            callbacks=callbacks,
            device=device,
            verbose=0,
            num_workers=0,
        )
        
        
        
        # Defensive Patch: WideDeepDataset sometimes receives X_tab as a dict {'X_tab': array} despite correct usage.
        # This patch unwraps it automatically.
        from pytorch_widedeep.training._wd_dataset import WideDeepDataset
        
        _original_wd_init = WideDeepDataset.__init__
        
        def _patched_wd_init(self, X_wide=None, X_tab=None, X_text=None, X_img=None, target=None, transforms=None):
            if X_tab is not None and isinstance(X_tab, dict) and 'X_tab' in X_tab:
                # print("[DEBUG] Patch triggered: Unwrapping X_tab from dict")
                X_tab = X_tab['X_tab']
            _original_wd_init(self, X_wide=X_wide, X_tab=X_tab, X_text=X_text, X_img=X_img, target=target, transforms=transforms)
            
        WideDeepDataset.__init__ = _patched_wd_init

        X_train_dict = {'X_tab': X_tab, 'target': df_train['target'].values}
        X_val_dict = {'X_tab': X_val_tab, 'target': df_val['target'].values} if X_val_tab is not None else None
        
        self.trainer.fit(X_train=X_train_dict, target=None, 
                         X_val=X_val_dict, target_val=None,
                         n_epochs=self.epochs, batch_size=self.batch_size)
        history = getattr(self.trainer, "history", None)
        best_epoch = getattr(self.trainer, "best_epoch", None)
        attach_training_metadata(
            self,
            optimizer="Adam",
            best_epoch=best_epoch,
            epochs_ran=self.epochs,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=train_lr,
            weight_decay=train_weight_decay,
            early_stopped=bool(best_epoch is not None and isinstance(best_epoch, int) and best_epoch < self.epochs),
            model_selection_metric="val_BinaryAUROC",
            epoch_history=history,
            architecture_params=effective_model_params,
        )
        return self

    def predict(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        X_tab = self.preprocessor.transform(df)
        return self.trainer.predict(X_tab={'X_tab': X_tab})

    def predict_proba(self, X):
        df = pd.DataFrame(X, columns=self.col_names)
        X_tab = self.preprocessor.transform(df)
        return self.trainer.predict_proba(X_tab={'X_tab': X_tab})


class TorchStateDictEarlyStopping(Callback):
    def __init__(self, monitor='val_auc', min_delta=1e-4, patience=0, verbose=0, mode='max', restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.min_delta = abs(min_delta)
        self.patience = int(patience)
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = None
        self.best_state_dict = None

        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = None
        self.best_state_dict = None
        self.best = np.inf if self.monitor_op == np.less else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            return

        self.wait += 1
        if self.wait < self.patience:
            return

        self.stopped_epoch = epoch + 1
        self.model.stop_training = True
        if self.restore_best_weights and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        if self.verbose > 0:
            print(f"Epoch {epoch + 1:05d}: early stopping")

    def _is_improvement(self, current, best):
        return self.monitor_op(current - self.min_delta, best)



class DeepCTRWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='DCN', batch_size=FIXED_BATCH_SIZE, epochs=50, patience=20, **kwargs):
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None
        self.feature_names = None
        self._autoint_bin_edges = None

    def _fit_autoint_input(self, X, feature_names, n_bins):
        from deepctr_torch.inputs import SparseFeat
        bin_edges = []
        encoded_columns = []
        feature_columns = []

        for i, name in enumerate(feature_names):
            col = X[:, i].astype(np.float32)
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(col, qs)
            edges = np.unique(edges)
            if edges.size <= 1:
                edges = np.array([edges[0], edges[0] + 1.0], dtype=np.float32)
            binned = np.digitize(col, edges[1:-1], right=False).astype(np.int64)
            vocab_size = int(max(2, edges.size))
            feature_columns.append(SparseFeat(name, vocabulary_size=vocab_size, embedding_dim=8))
            encoded_columns.append(binned)
            bin_edges.append(edges)

        self._autoint_bin_edges = bin_edges
        model_input = np.ascontiguousarray(np.stack(encoded_columns, axis=1))
        return feature_columns, [model_input]

    def _transform_autoint_input(self, X, feature_names):
        encoded_columns = []
        for i, name in enumerate(feature_names):
            col = X[:, i].astype(np.float32)
            edges = self._autoint_bin_edges[i]
            encoded_columns.append(np.digitize(col, edges[1:-1], right=False).astype(np.int64))
        return [np.ascontiguousarray(np.stack(encoded_columns, axis=1))]

    def _transform_dense_input(self, X):
        X_array = np.asarray(X, dtype=np.float32)
        if not X_array.flags.c_contiguous:
            X_array = np.ascontiguousarray(X_array)
        return [X_array]

    def fit(self, X, y, X_val=None, y_val=None):
        from deepctr_torch.inputs import DenseFeat
        params = _drop_optuna_helper_params(self.kwargs)
        lr = float(params.pop("lr", 1e-3))
        weight_decay = float(params.pop("weight_decay", 0.0) or 0.0)
        effective_model_params = {}
        self.batch_size = int(self.batch_size or FIXED_BATCH_SIZE)

        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        self.feature_names = feature_names
        train_model_input = self._transform_dense_input(X)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.model_type == 'DCN':
            from deepctr_torch.models import DCN
            dcn_params = params.copy()
            if 'n_cross_layers' in dcn_params and 'cross_num' not in dcn_params:
                dcn_params['cross_num'] = dcn_params.pop('n_cross_layers')
            if 'hidden_dropout' in dcn_params and 'dnn_dropout' not in dcn_params:
                dcn_params['dnn_dropout'] = dcn_params.pop('hidden_dropout')
            if 'dropout' in dcn_params and 'dnn_dropout' not in dcn_params:
                dcn_params['dnn_dropout'] = dcn_params.pop('dropout')
            if 'dnn_hidden_units' in dcn_params and not isinstance(dcn_params['dnn_hidden_units'], tuple):
                dcn_params['dnn_hidden_units'] = tuple(dcn_params['dnn_hidden_units'])
            if 'dnn_hidden_units' not in dcn_params and {'layer_size', 'n_hidden_layers'} <= set(dcn_params):
                dcn_params['dnn_hidden_units'] = tuple(
                    [int(dcn_params.pop('layer_size'))] * int(dcn_params.pop('n_hidden_layers'))
                )
            if 'dnn_hidden_units' in dcn_params:
                original_units = tuple(int(v) for v in dcn_params['dnn_hidden_units'])
                capped_units = tuple(min(v, 256) for v in original_units[:4])
                if capped_units != original_units:
                    print(f"[INFO] DCN: capped dnn_hidden_units from {original_units} to {capped_units}.")
                dcn_params['dnn_hidden_units'] = capped_units
            if 'cross_num' in dcn_params and int(dcn_params['cross_num']) > 4:
                print(f"[INFO] DCN: capped cross_num from {dcn_params['cross_num']} to 4.")
                dcn_params['cross_num'] = 4
            dcn_params.pop('cross_dropout', None)
            dcn_params.pop('layer_size', None)
            dcn_params.pop('n_hidden_layers', None)
            dcn_params.setdefault('l2_reg_linear', weight_decay)
            dcn_params.setdefault('l2_reg_embedding', weight_decay)
            dcn_params.setdefault('l2_reg_cross', weight_decay)
            dcn_params.setdefault('l2_reg_dnn', weight_decay)
            # Keep the same raw dense input, but represent it as one wide DenseFeat to reduce
            # DeepCTR feature-index bookkeeping overhead for 9k+ column tables.
            self.feature_columns = [DenseFeat('dense_input', X.shape[1])]
            dcn_params = _filter_supported_kwargs(DCN.__init__, dcn_params)
            effective_model_params = dict(dcn_params)
            self.model = DCN(
                self.feature_columns,
                self.feature_columns,
                task='binary',
                device=device,
                **dcn_params,
            )
        elif self.model_type == 'AutoInt':
            from deepctr_torch.models import AutoInt
            autoint_params = params.copy()
            n_bins = int(autoint_params.pop('autoint_bins', 16))
            if n_bins > 16:
                print(f"[INFO] AutoInt: capped autoint_bins from {n_bins} to 16.")
                n_bins = 16
            if 'dropout' in autoint_params and 'dnn_dropout' not in autoint_params:
                autoint_params['dnn_dropout'] = autoint_params.pop('dropout')
            if 'att_layer_num' in autoint_params and int(autoint_params['att_layer_num']) > 3:
                print(f"[INFO] AutoInt: capped att_layer_num from {autoint_params['att_layer_num']} to 3.")
                autoint_params['att_layer_num'] = 3
            if 'att_head_num' in autoint_params and int(autoint_params['att_head_num']) > 4:
                print(f"[INFO] AutoInt: capped att_head_num from {autoint_params['att_head_num']} to 4.")
                autoint_params['att_head_num'] = 4
            autoint_params.setdefault('l2_reg_dnn', weight_decay)
            autoint_params.setdefault('l2_reg_embedding', weight_decay)
            autoint_params.pop('att_embedding_dim', None)
            self.feature_columns, train_model_input = self._fit_autoint_input(X, feature_names, n_bins=n_bins)
            autoint_params = _filter_supported_kwargs(AutoInt.__init__, autoint_params)
            effective_model_params = dict(autoint_params)
            self.model = AutoInt(
                self.feature_columns,
                self.feature_columns,
                task='binary',
                device=device,
                **autoint_params,
            )
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.compile(optimizer, "binary_crossentropy", metrics=['binary_crossentropy', 'auc'])
        
        val_data = None
        callbacks = []
        early_stopping = None
        if X_val is not None:
            if self.model_type == 'AutoInt':
                val_model_input = self._transform_autoint_input(X_val, feature_names)
            else:
                val_model_input = self._transform_dense_input(X_val)
            val_data = (val_model_input, y_val)
            if self.patience > 0:
                early_stopping = TorchStateDictEarlyStopping(
                    monitor='val_auc',
                    min_delta=1e-4,
                    patience=self.patience,
                    verbose=1,
                    mode='max',
                    restore_best_weights=True,
                )
                callbacks.append(early_stopping)
        
        history = self.model.fit(
            train_model_input,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=0,
        )
        history_dict = getattr(history, 'history', {}) or {}
        epochs_ran = len(history_dict.get('loss', [])) or self.epochs
        best_epoch = (early_stopping.best_epoch + 1) if (early_stopping and early_stopping.best_epoch is not None) else None
        early_stopped = bool(early_stopping and early_stopping.stopped_epoch > 0)
        attach_training_metadata(
            self,
            optimizer="Adam",
            best_epoch=best_epoch,
            epochs_ran=epochs_ran,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            lr=lr,
            weight_decay=weight_decay,
            early_stopped=early_stopped,
            model_selection_metric="val_auroc",
            epoch_history=history_dict,
            architecture_params=effective_model_params,
        )
        return self

    def predict(self, X):
        feature_names = self.feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        if self.model_type == 'AutoInt':
            test_model_input = self._transform_autoint_input(X, feature_names)
        else:
            test_model_input = self._transform_dense_input(X)
        pred_ans = self.model.predict(test_model_input, batch_size=self.batch_size)
        return np.where(pred_ans > 0.5, 1, 0).astype(int).flatten()

    def predict_proba(self, X):
        feature_names = self.feature_names or [f"feat_{i}" for i in range(X.shape[1])]
        if self.model_type == 'AutoInt':
            test_model_input = self._transform_autoint_input(X, feature_names)
        else:
            test_model_input = self._transform_dense_input(X)
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


def train_torch_model(model, X_train, y_train, X_val, y_val, 
                      epochs=50, batch_size=FIXED_BATCH_SIZE, lr=1e-3, weight_decay=0.0, patience=5,
                      device='cuda' if torch.cuda.is_available() else 'cpu',
                      X_test=None, y_test=None):
    
    model = model.to(device)
    batch_size = int(batch_size or FIXED_BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    pin = torch.cuda.is_available()
    train_drop_last = len(train_dataset) > batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=train_drop_last, pin_memory=pin, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=2, persistent_workers=True)
    
    criterion = nn.CrossEntropyLoss()
    
    epoch_iterator = tqdm(range(epochs), desc="Training Epochs")

    def _evaluate_test():
        if X_test is None or y_test is None:
            return None, None
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        all_probs = []
        all_targets = []
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        acc = correct / max(1, total)
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        try:
            if all_probs.shape[1] == 2:
                auroc = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                auroc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except Exception:
            auroc = 0.5
        return acc, auroc
    
    best_val_score = -float('inf')  # Changed from best_val_loss (inf)
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

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
            
        train_loss /= max(1, len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                # Collect probs for AUROC
                probs = torch.softmax(outputs, dim=1)[:, 1] # Binary classification assumption (pos class)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= max(1, len(val_loader))
        try:
            val_auroc = roc_auc_score(val_targets, val_probs)
        except:
            val_auroc = 0.5
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        # Early stopping (Maximize AUROC)
        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
            # epoch_iterator.write(f"New Best AUROC: {val_auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        test_acc, test_auroc = _evaluate_test()
        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            "epoch": epoch_num,
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "val_auroc": round(float(val_auroc), 6),
            "test_accuracy": round(float(test_acc), 6) if test_acc is not None else None,
            "test_auroc": round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)

    attach_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        model_selection_metric="val_auroc",
        epoch_history=epoch_history,
    )
    return model

def evaluate_model(model, X_test, y_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    is_torch = isinstance(model, nn.Module)

    if is_torch:
        model.eval()
        model.to(device)
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            if hasattr(model, 'predict'):
                # DG/DA models (DGModel, DAModel subclasses) expose predict() not forward()
                logits = model.predict(X_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]  # some models return (class_logits, domain_logits)
            else:
                logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    else:
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    try:
        if probs.shape[1] == 2:
            auroc = roc_auc_score(y_test, probs[:, 1])
        else:
            auroc = roc_auc_score(y_test, probs, multi_class='ovr')
    except Exception:
        auroc = 0.5

    return {"Accuracy": acc, "F1": f1, "AUROC": auroc}
