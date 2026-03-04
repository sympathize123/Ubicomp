import numpy as np
from src.domainbed_algos import DGModel
from src.da_models import DAModel

def get_hparams(algorithm, dataset, backbone='MLP'):
    """
    Return a dictionary of hyperparameters for a given algorithm and dataset.
    This defines the search space for Randomized Search (Optuna).
    """
    
    # Common Hparams
    hparams = {}
    
    # Grid/Distributions
    # tailored based on DomainBed and Tabular DL papers
    
    # 1. ERM / Backbone Baselines (MLP, ResNet)
    if algorithm in ['MLP', 'ResNet', 'ERM_DG', 'DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM']:
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [32, 64, 128])
        hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.5)
        
        # Architecture Search (Backbone-specific)
        hparams['hidden_dim'] = lambda trial: trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        
        # MLP Backbone
        if backbone == 'MLP' or algorithm == 'MLP':
             hparams['num_layers'] = lambda trial: trial.suggest_int('num_layers', 2, 6)
             
        # ResNet Backbone
        elif backbone == 'ResNet' or algorithm == 'ResNet':
             hparams['num_blocks'] = lambda trial: trial.suggest_int('num_blocks', 2, 4)
             
        # Transformer Backbone
        elif backbone == 'Transformer':
             hparams['num_layers'] = lambda trial: trial.suggest_int('num_layers', 2, 6) # Transformer layers
             hparams['nhead'] = lambda trial: trial.suggest_categorical('nhead', [2, 4, 8])
        
        # Algorithm specific
        if algorithm in ['DANN', 'CDAN', 'ADDA']:
            hparams['discriminator_lr'] = lambda trial: trial.suggest_float('discriminator_lr', 1e-5, 1e-2, log=True)
            
        if algorithm == 'DeepCORAL':
            hparams['mmd_gamma'] = lambda trial: trial.suggest_float('mmd_gamma', 0.1, 10.0, log=True)

        if algorithm == 'DAN':
            hparams['dan_trade_off'] = lambda trial: trial.suggest_float('dan_trade_off', 0.1, 10.0, log=True)
            
        if algorithm == 'MCC':
            hparams['mcc_temp'] = lambda trial: trial.suggest_float('mcc_temp', 1.0, 5.0)
            
        if algorithm == 'JAN':
            hparams['jmmd_lambda'] = lambda trial: trial.suggest_float('jmmd_lambda', 0.1, 10.0)

        # New: CBST
        if algorithm == 'CBST':
            # Self-training portion/iterations are fixed in logic usually, but we could tune.
            pass

    # 2. DG Specific
    elif algorithm in ['IRM', 'VREx', 'GroupDRO', 'MixStyle', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet']:
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [32, 64]) # Smaller batch for DG usually
        
        if algorithm == 'IRM':
            hparams['irm_lambda'] = lambda trial: trial.suggest_float('irm_lambda', 1e-1, 1e4, log=True)
            hparams['irm_penalty_anneal_iters'] = lambda trial: trial.suggest_int('irm_penalty_anneal_iters', 0, 50)
            
        if algorithm == 'VREx':
            hparams['vrex_lambda'] = lambda trial: trial.suggest_float('vrex_lambda', 1e-1, 1e4, log=True)
            hparams['vrex_penalty_anneal_iters'] = lambda trial: trial.suggest_int('vrex_penalty_anneal_iters', 0, 50)
            
        if algorithm == 'GroupDRO':
            hparams['groupdro_eta'] = lambda trial: trial.suggest_float('groupdro_eta', 1e-3, 1.0, log=True)
            
        if algorithm == 'MixStyle':
            hparams['mixstyle_alpha'] = lambda trial: trial.suggest_float('mixstyle_alpha', 0.1, 0.5)
            hparams['mixstyle_p'] = lambda trial: trial.suggest_float('mixstyle_p', 0.1, 0.9)
            hparams['mixstyle_mix'] = lambda trial: trial.suggest_categorical('mixstyle_mix', ['random', 'crossdomain'])

        if algorithm == 'MLDG':
            hparams['mldg_beta'] = lambda trial: trial.suggest_float('mldg_beta', 0.1, 2.0)
            hparams['n_meta_test'] = lambda trial: trial.suggest_int('n_meta_test', 1, 2)

        if algorithm == 'MASF':
            hparams['masf_inner_lr'] = lambda trial: trial.suggest_float('masf_inner_lr', 1e-5, 1e-2, log=True)
            hparams['masf_metric_lr'] = lambda trial: trial.suggest_float('masf_metric_lr', 1e-5, 1e-2, log=True)
            hparams['masf_metric_weight'] = lambda trial: trial.suggest_float('masf_metric_weight', 1e-4, 1e-2, log=True)
            hparams['masf_margin'] = lambda trial: trial.suggest_float('masf_margin', 0.2, 2.0)
            hparams['masf_temperature'] = lambda trial: trial.suggest_float('masf_temperature', 1.0, 5.0)
            hparams['masf_metric_dim'] = lambda trial: trial.suggest_categorical('masf_metric_dim', [64, 128, 256])

        if algorithm == 'Fish':
            hparams['meta_lr'] = lambda trial: trial.suggest_float('meta_lr', 0.1, 1.0)
            
        if algorithm == 'CSD':
            hparams['csd_lambda'] = lambda trial: trial.suggest_float('csd_lambda', 0.1, 5.0)
            hparams['csd_k'] = lambda trial: trial.suggest_categorical('csd_k', [2, 3])

        if algorithm == 'SagNet':
            hparams['sag_w_adv'] = lambda trial: trial.suggest_float('sag_w_adv', 0.1, 2.0)

    # 3. Tabular Deep Learning
    elif algorithm == 'TabNet':
        hparams['n_d'] = lambda trial: trial.suggest_int('n_d', 8, 64)
        hparams['n_a'] = lambda trial: trial.suggest_int('n_a', 8, 64)
        hparams['n_steps'] = lambda trial: trial.suggest_int('n_steps', 3, 10)
        hparams['gamma'] = lambda trial: trial.suggest_float('gamma', 1.0, 2.0)
        hparams['lambda_sparse'] = lambda trial: trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])

    elif algorithm == 'TabTransformer':
        hparams['input_dim'] = lambda trial: trial.suggest_categorical('input_dim', [16, 32])
        hparams['n_heads'] = lambda trial: trial.suggest_categorical('n_heads', [2, 4, 8])
        hparams['n_blocks'] = lambda trial: trial.suggest_int('n_blocks', 1, 4)
        hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.3)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        
    elif algorithm == 'SAINT':
        hparams['input_dim'] = lambda trial: trial.suggest_categorical('input_dim', [16, 32])
        hparams['n_heads'] = lambda trial: trial.suggest_categorical('n_heads', [2, 4, 8])
        hparams['n_blocks'] = lambda trial: trial.suggest_int('n_blocks', 1, 4)
        hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.3)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    elif algorithm == 'FTTransformer':
        hparams['input_dim'] = lambda trial: trial.suggest_categorical('input_dim', [16, 32])
        hparams['n_heads'] = lambda trial: trial.suggest_categorical('n_heads', [2, 4, 8])
        hparams['n_blocks'] = lambda trial: trial.suggest_int('n_blocks', 1, 4)
        hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.3)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-4, 1e-3, log=True)

    elif algorithm == 'DCN':
        hparams['dnn_hidden_units'] = lambda trial: trial.suggest_categorical('dnn_hidden_units', [(128, 128), (256, 128), (256, 256)])
        hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.5)

    elif algorithm == 'AutoInt':
        hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.3)
        hparams['att_layer_num'] = lambda trial: trial.suggest_int('att_layer_num', 1, 4)
        hparams['att_head_num'] = lambda trial: trial.suggest_categorical('att_head_num', [2, 4, 8])

    # 4. Tree-based (XGB/LGB)
    elif algorithm == 'XGB':
        hparams['learning_rate'] = lambda trial: trial.suggest_float('learning_rate', 1e-3, 1.0, log=True)
        hparams['max_depth'] = lambda trial: trial.suggest_int('max_depth', 3, 10)
        hparams['n_estimators'] = lambda trial: trial.suggest_int('n_estimators', 50, 500)
        hparams['subsample'] = lambda trial: trial.suggest_float('subsample', 0.5, 1.0)
        hparams['colsample_bytree'] = lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1.0)
        
    elif algorithm == 'LGB':
        hparams['learning_rate'] = lambda trial: trial.suggest_float('learning_rate', 1e-3, 1.0, log=True)
        hparams['num_leaves'] = lambda trial: trial.suggest_int('num_leaves', 8, 128)
        hparams['n_estimators'] = lambda trial: trial.suggest_int('n_estimators', 50, 500)
        hparams['min_child_samples'] = lambda trial: trial.suggest_int('min_child_samples', 5, 100)
        
    return hparams
