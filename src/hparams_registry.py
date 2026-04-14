def _zero_or_loguniform(trial, name, low, high):
    if trial.suggest_categorical(f"{name}_is_zero", [False, True]):
        return 0.0
    return trial.suggest_float(name, low, high, log=True)


def _pow2_uniform_int(trial, name, low_exp, high_exp):
    value = 2 ** trial.suggest_float(f"{name}_log2", low_exp, high_exp)
    return int(round(value))


def _pow10_uniform_int(trial, name, low_exp, high_exp):
    value = 10 ** trial.suggest_float(f"{name}_log10", low_exp, high_exp)
    return int(round(value))


def _zero_or_uniform(trial, name, low, high):
    if trial.suggest_categorical(f"{name}_is_zero", [False, True]):
        return 0.0
    return trial.suggest_float(name, low, high)


def _add_mlp_backbone_hparams(hparams):
    hparams['hidden_dim'] = lambda trial: trial.suggest_int('hidden_dim', 64, 512)
    hparams['num_layers'] = lambda trial: trial.suggest_int('num_layers', 1, 8)
    hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.5)


def _add_resnet_backbone_hparams(hparams):
    hparams['hidden_dim'] = lambda trial: trial.suggest_int('hidden_dim', 64, 512)
    hparams['num_blocks'] = lambda trial: trial.suggest_int('num_blocks', 1, 8)
    hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.5)


def _add_transformer_backbone_hparams(hparams):
    hparams['num_layers'] = lambda trial: trial.suggest_int('num_layers', 1, 6)
    hparams['nhead'] = lambda trial: trial.suggest_categorical('nhead', [2, 4, 8])
    hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.5)


def _add_backbone_hparams(hparams, algorithm, backbone):
    if backbone == 'ResNet' or algorithm == 'ResNet':
        _add_resnet_backbone_hparams(hparams)
    elif backbone == 'Transformer':
        _add_transformer_backbone_hparams(hparams)
    else:
        _add_mlp_backbone_hparams(hparams)


def get_hparams(algorithm, dataset, backbone='MLP'):
    """
    Return a dictionary of Optuna search spaces for a given algorithm/dataset.
    Batch-size candidates are intentionally left unchanged to avoid VRAM regressions.
    """

    hparams = {}

    if algorithm in ['MLP', 'ResNet']:
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
        _add_backbone_hparams(hparams, algorithm, backbone)

    elif algorithm in ['ERM_DG', 'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet']:
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 3e-5, 3e-4, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        if algorithm in ['IRM', 'VREx']:
            hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [32, 64])
        else:
            hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
        hparams['dropout'] = lambda trial: trial.suggest_categorical('dropout', [0.0, 0.1, 0.5])
        _add_backbone_hparams(hparams, algorithm, backbone)

        if algorithm == 'IRM':
            hparams['irm_lambda'] = lambda trial: trial.suggest_float('irm_lambda', 1e-1, 1e5, log=True)
            hparams['irm_penalty_anneal_iters'] = lambda trial: _pow10_uniform_int(trial, 'irm_penalty_anneal_iters', 0, 4)

        if algorithm == 'VREx':
            hparams['vrex_lambda'] = lambda trial: trial.suggest_float('vrex_lambda', 1e-1, 1e5, log=True)
            hparams['vrex_penalty_anneal_iters'] = lambda trial: _pow10_uniform_int(trial, 'vrex_penalty_anneal_iters', 0, 4)

        if algorithm == 'GroupDRO':
            hparams['groupdro_eta'] = lambda trial: trial.suggest_float('groupdro_eta', 1e-3, 1e-1, log=True)

        if algorithm == 'MixStyle':
            hparams['mixstyle_alpha'] = lambda trial: trial.suggest_float('mixstyle_alpha', 0.1, 0.5)
            hparams['mixstyle_p'] = lambda trial: trial.suggest_float('mixstyle_p', 0.1, 0.9)
            hparams['mixstyle_mix'] = lambda trial: trial.suggest_categorical('mixstyle_mix', ['random', 'crossdomain'])

        if algorithm == 'MLDG':
            hparams['mldg_beta'] = lambda trial: trial.suggest_float('mldg_beta', 1e-1, 10.0, log=True)
            hparams['n_meta_test'] = lambda trial: trial.suggest_categorical('n_meta_test', [1, 2])

        if algorithm == 'MASF':
            hparams['masf_inner_lr'] = lambda trial: trial.suggest_float('masf_inner_lr', 1e-5, 1e-2, log=True)
            hparams['masf_metric_lr'] = lambda trial: trial.suggest_float('masf_metric_lr', 1e-5, 1e-2, log=True)
            hparams['masf_metric_weight'] = lambda trial: trial.suggest_float('masf_metric_weight', 1e-4, 1e-2, log=True)
            hparams['masf_margin'] = lambda trial: trial.suggest_float('masf_margin', 0.2, 2.0)
            hparams['masf_temperature'] = lambda trial: trial.suggest_float('masf_temperature', 1.0, 5.0)
            hparams['masf_metric_dim'] = lambda trial: trial.suggest_categorical('masf_metric_dim', [64, 128, 256])

        if algorithm == 'Fish':
            hparams['meta_lr'] = lambda trial: trial.suggest_categorical('meta_lr', [0.05, 0.1, 0.5])

        if algorithm == 'CSD':
            hparams['csd_lambda'] = lambda trial: trial.suggest_float('csd_lambda', 0.1, 5.0)
            hparams['csd_k'] = lambda trial: trial.suggest_categorical('csd_k', [2, 3])

        if algorithm == 'SagNet':
            hparams['sag_w_adv'] = lambda trial: trial.suggest_float('sag_w_adv', 1e-2, 10.0, log=True)

    elif algorithm in ['DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM']:
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
        _add_backbone_hparams(hparams, algorithm, backbone)

        if algorithm in ['DANN', 'CDAN', 'ADDA']:
            hparams['discriminator_lr'] = lambda trial: trial.suggest_float('discriminator_lr', 1e-5, 1e-2, log=True)
            hparams['disc_hidden'] = lambda trial: _pow2_uniform_int(trial, 'disc_hidden', 6, 10)

        if algorithm == 'DANN':
            hparams['trade_off'] = lambda trial: trial.suggest_float('trade_off', 1e-2, 1e2, log=True)

        if algorithm == 'CDAN':
            hparams['trade_off'] = lambda trial: trial.suggest_float('trade_off', 1e-2, 1e2, log=True)
            hparams['cdan_entropy'] = lambda trial: trial.suggest_categorical('cdan_entropy', [True, False])
            hparams['cdan_randomized'] = lambda trial: trial.suggest_categorical('cdan_randomized', [False, True])
            hparams['cdan_randomized_dim'] = lambda trial: trial.suggest_categorical('cdan_randomized_dim', [256, 512, 1024])

        if algorithm == 'DeepCORAL':
            hparams['mmd_gamma'] = lambda trial: trial.suggest_float('mmd_gamma', 0.1, 10.0, log=True)

        if algorithm == 'DAN':
            hparams['dan_trade_off'] = lambda trial: trial.suggest_float('dan_trade_off', 0.1, 10.0, log=True)
            hparams['dan_kernel_num'] = lambda trial: trial.suggest_categorical('dan_kernel_num', [3, 5, 7])
            hparams['dan_linear'] = lambda trial: trial.suggest_categorical('dan_linear', [False, True])

        if algorithm == 'MCC':
            hparams['mcc_temp'] = lambda trial: trial.suggest_float('mcc_temp', 1.0, 5.0)
            hparams['mcc_trade_off'] = lambda trial: trial.suggest_float('mcc_trade_off', 1e-2, 1e1, log=True)

        if algorithm == 'MCD':
            hparams['mcd_k'] = lambda trial: trial.suggest_int('mcd_k', 1, 4)
            hparams['mcd_trade_off'] = lambda trial: trial.suggest_float('mcd_trade_off', 0.5, 2.0)

        if algorithm == 'JAN':
            hparams['jmmd_lambda'] = lambda trial: trial.suggest_float('jmmd_lambda', 0.1, 10.0)
            hparams['jan_kernel_num'] = lambda trial: trial.suggest_categorical('jan_kernel_num', [3, 5, 7])
            hparams['jan_linear'] = lambda trial: trial.suggest_categorical('jan_linear', [False, True])

        if algorithm == 'SHOT':
            hparams['shot_cls_par'] = lambda trial: trial.suggest_float('shot_cls_par', 0.0, 1.0)
            hparams['shot_ent_par'] = lambda trial: trial.suggest_float('shot_ent_par', 0.5, 2.0)
            hparams['shot_interval'] = lambda trial: trial.suggest_int('shot_interval', 5, 20)
            hparams['shot_epochs'] = lambda trial: trial.suggest_int('shot_epochs', 10, 50)

        if algorithm == 'CBST':
            hparams['cbst_pretrain_epochs'] = lambda trial: trial.suggest_int('cbst_pretrain_epochs', 10, 30)
            hparams['cbst_max_iter'] = lambda trial: trial.suggest_categorical('cbst_max_iter', [3, 5, 7])
            hparams['cbst_init_port'] = lambda trial: trial.suggest_float('cbst_init_port', 0.1, 0.3)
            hparams['cbst_port_step'] = lambda trial: trial.suggest_float('cbst_port_step', 0.05, 0.2)
            hparams['cbst_max_port'] = lambda trial: trial.suggest_float('cbst_max_port', 0.6, 0.9)
            hparams['cbst_retrain_epochs'] = lambda trial: trial.suggest_categorical('cbst_retrain_epochs', [3, 5, 10])

        if algorithm == 'CGDM':
            hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            hparams['num_k'] = lambda trial: trial.suggest_categorical('num_k', [2, 4])

    elif algorithm == 'TabNet':
        hparams['n_d'] = lambda trial: trial.suggest_categorical('n_d', [8, 16, 32, 64])
        hparams['n_a'] = lambda trial: trial.suggest_categorical('n_a', [8, 16, 32, 64])
        hparams['n_steps'] = lambda trial: trial.suggest_int('n_steps', 3, 10)
        hparams['gamma'] = lambda trial: trial.suggest_float('gamma', 1.0, 2.0)
        hparams['lambda_sparse'] = lambda trial: trial.suggest_float('lambda_sparse', 1e-6, 1e-1, log=True)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-3, 1e-2)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        hparams['batch_size'] = lambda trial: trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])

    elif algorithm in ['TabTransformer', 'SAINT']:
        hparams['input_dim'] = lambda trial: trial.suggest_categorical('input_dim', [16, 32])
        hparams['n_heads'] = lambda trial: trial.suggest_categorical('n_heads', [2, 4, 8])
        hparams['n_blocks'] = lambda trial: trial.suggest_int('n_blocks', 1, 4)
        hparams['attn_dropout'] = lambda trial: trial.suggest_float('attn_dropout', 0.0, 0.3)
        hparams['ff_dropout'] = lambda trial: trial.suggest_float('ff_dropout', 0.0, 0.3)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    elif algorithm == 'FTTransformer':
        hparams['n_blocks'] = lambda trial: trial.suggest_int('n_blocks', 1, 4)
        hparams['input_dim'] = lambda trial: trial.suggest_int('input_dim', 64, 512)
        hparams['attn_dropout'] = lambda trial: trial.suggest_float('attn_dropout', 0.0, 0.5)
        hparams['ff_dropout'] = lambda trial: trial.suggest_float('ff_dropout', 0.0, 0.5)
        hparams['residual_dropout'] = lambda trial: _zero_or_uniform(trial, 'residual_dropout', 0.0, 0.2)
        hparams['ff_factor'] = lambda trial: trial.suggest_float('ff_factor', 2.0 / 3.0, 8.0 / 3.0)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    elif algorithm == 'DCN':
        def _dcn_hidden_units(trial):
            n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 8)
            layer_size = trial.suggest_int('layer_size', 64, 512)
            return tuple([layer_size] * n_hidden_layers)

        hparams['n_cross_layers'] = lambda trial: trial.suggest_int('n_cross_layers', 1, 8)
        hparams['dnn_hidden_units'] = _dcn_hidden_units
        hparams['hidden_dropout'] = lambda trial: trial.suggest_float('hidden_dropout', 0.0, 0.5)
        hparams['cross_dropout'] = lambda trial: _zero_or_uniform(trial, 'cross_dropout', 0.0, 0.5)
        hparams['lr'] = lambda trial: trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        hparams['weight_decay'] = lambda trial: trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    elif algorithm == 'AutoInt':
        hparams['dropout'] = lambda trial: trial.suggest_float('dropout', 0.0, 0.3)
        hparams['att_layer_num'] = lambda trial: trial.suggest_int('att_layer_num', 1, 4)
        hparams['att_head_num'] = lambda trial: trial.suggest_categorical('att_head_num', [2, 4, 8])

    elif algorithm == 'XGB':
        hparams['learning_rate'] = lambda trial: trial.suggest_float('learning_rate', 1e-5, 1.0, log=True)
        hparams['max_depth'] = lambda trial: trial.suggest_int('max_depth', 3, 10)
        hparams['min_child_weight'] = lambda trial: trial.suggest_float('min_child_weight', 1e-8, 1e5, log=True)
        hparams['subsample'] = lambda trial: trial.suggest_float('subsample', 0.5, 1.0)
        hparams['colsample_bylevel'] = lambda trial: trial.suggest_float('colsample_bylevel', 0.5, 1.0)
        hparams['colsample_bytree'] = lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1.0)
        hparams['gamma'] = lambda trial: _zero_or_loguniform(trial, 'gamma', 1e-8, 1e2)
        hparams['reg_lambda'] = lambda trial: _zero_or_loguniform(trial, 'reg_lambda', 1e-8, 1e2)
        hparams['reg_alpha'] = lambda trial: _zero_or_loguniform(trial, 'reg_alpha', 1e-8, 1e2)
        hparams['n_estimators'] = lambda trial: trial.suggest_int('n_estimators', 50, 500)

    elif algorithm == 'LGB':
        hparams['learning_rate'] = lambda trial: trial.suggest_float('learning_rate', 1e-3, 1.0, log=True)
        hparams['num_leaves'] = lambda trial: trial.suggest_int('num_leaves', 8, 128)
        hparams['n_estimators'] = lambda trial: trial.suggest_int('n_estimators', 50, 500)
        hparams['min_child_samples'] = lambda trial: trial.suggest_int('min_child_samples', 5, 100)
        hparams['subsample'] = lambda trial: trial.suggest_float('subsample', 0.5, 1.0)
        hparams['colsample_bytree'] = lambda trial: trial.suggest_float('colsample_bytree', 0.5, 1.0)

    return hparams
