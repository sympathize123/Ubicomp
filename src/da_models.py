import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import grad
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from src.backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer
from src.models import attach_training_metadata
from scipy.spatial.distance import cdist
from src.da_tllib_losses import (
    WarmStartGradientReverseLayer,
    DomainDiscriminator,
    DomainAdversarialLoss,
    ConditionalDomainAdversarialLoss,
    GaussianKernel,
    MultipleKernelMaximumMeanDiscrepancy,
    JointMultipleKernelMaximumMeanDiscrepancy,
    CorrelationAlignmentLoss,
    MinimumClassConfusionLoss,
    classifier_discrepancy,
    mcd_entropy,
    entropy as tllib_entropy,
)

class DAModel(nn.Module):
    """
    Base class for Domain Adaptation models ensuring backbone-agnostic behavior.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DAModel, self).__init__()
        self.hparams = hparams if hparams else {}
        backbone_name = self.hparams.get('backbone', 'MLP')
        dropout = self.hparams.get('dropout', 0.3)
        hidden_dim = self.hparams.get('hidden_dim', 256)
        num_layers = self.hparams.get('num_layers', 3)
        
        if backbone_name == 'MLP':
            self.feature_extractor = MLPFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, num_layers=num_layers, dropout=dropout)
        elif backbone_name == 'ResNet':
            # ResNet hparams
            num_blocks = self.hparams.get('num_blocks', 2)
            self.feature_extractor = ResNetFeaturizer(input_dim, hidden_dim=hidden_dim, output_dim=128, num_blocks=num_blocks, dropout=dropout)
        elif backbone_name == 'Transformer':
            num_layers_tr = self.hparams.get('num_layers', 2)
            nhead = self.hparams.get('nhead', 4)
            self.feature_extractor = TransformerFeaturizer(input_dim, hidden_dim=128, output_dim=128, num_layers=num_layers_tr, nhead=nhead, dropout=dropout)
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
    """
    DANN backbone. Domain discriminator and adversarial loss are handled in training.
    """
    def __init__(self, input_dim, num_classes=2, num_domains=None, hparams=None):
        super(DANN, self).__init__(input_dim, num_classes, hparams)
        self.num_domains = num_domains

class CDAN(DAModel):
    """
    CDAN backbone. Conditional adversarial loss handled in training.
    """
    def __init__(self, input_dim, num_classes=2, num_domains=None, hparams=None):
        super(CDAN, self).__init__(input_dim, num_classes, hparams)
        self.num_domains = num_domains

class MCC(DAModel):
    """
    Minimum Class Confusion (Jin et al., 2020)
    Non-adversarial. Loss-based optimization.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(MCC, self).__init__(input_dim, num_classes, hparams)
        self.temperature = self.hparams.get('mcc_temp', 2.0)
    
    def forward(self, x):
        return self.predict(x)


def _finalize_training_metadata(
    model,
    *,
    optimizer,
    best_epoch,
    early_stopped,
    early_stop_epoch,
    epochs_ran,
    max_epochs,
    batch_size,
    patience,
    lr=None,
    weight_decay=None,
    model_selection_metric="val_auroc",
    best_metric_value=None,
    epoch_history=None,
    extra=None,
):
    payload = {
        "optimizer": optimizer,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "early_stop_epoch": early_stop_epoch,
        "epochs_ran": epochs_ran,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "model_selection_metric": model_selection_metric,
        "best_metric_value": best_metric_value,
        "epoch_history": epoch_history or [],
    }
    if extra:
        payload.update(extra)
    attach_training_metadata(model, **payload)
    return model


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

def train_deepcoral(model, X_train, y_train, d_train, X_val, y_val, d_val,
                    epochs=50, batch_size=64, lr=1e-3, patience=5,
                    device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
                    X_test=None, y_test=None):
    """
    DeepCORAL (TLL-style): L = L_cls(source) + lambda * CORAL(f_s, f_t).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("DeepCORAL requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()
    coral = CorrelationAlignmentLoss()
    trade_off = model.hparams.get('coral_lambda', model.hparams.get('mmd_gamma', 1.0))

    # Dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    target_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def infinite_iterator(loader):
        while True:
            for batch in loader:
                yield batch

    target_iter = infinite_iterator(target_loader)

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="DeepCORAL Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            logits_s = model.classifier(f_s)

            cls_loss = class_criterion(logits_s, y_s)
            transfer_loss = coral(f_s, f_t)
            loss = cls_loss + trade_off * transfer_loss

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
                logits = model.predict(X_batch)
                val_loss += class_criterion(logits, y_batch).item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        try:
            val_auroc = roc_auc_score(val_targets, val_probs)
        except:
            val_auroc = 0.5

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
    )
    return model

# --- CGDM: Cross-Domain Gradient Discrepancy Minimization (CVPR 2021) ---
# Ported from /home/iclab/minseo/DomainAdaptation/CGDM with minimal changes.

class CGDMBackbone(nn.Module):
    """
    CGDM ResBottle (MLP for tabular features).
    """
    def __init__(self, input_dim=4):
        super(CGDMBackbone, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dim = 128

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(self.bn1(out))
        out = self.fc2(out)
        out = F.relu(self.bn2(out))
        out = self.fc3(out)
        out = F.relu(self.bn3(out))
        return out

    def output_num(self):
        return self.dim


def grad_reverse(x, lambd=1.0):
    return GradientReversalLayer.apply(x, lambd)


class CGDMClassifier(nn.Module):
    """
    CGDM ResClassifier (MLP classifier head).
    """
    def __init__(self, num_classes=3, num_layer=4, num_unit=128, prob=0.5, middle=64):
        super(CGDMClassifier, self).__init__()
        layers = []
        in_dim = num_unit

        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(in_dim, middle))
        layers.append(nn.BatchNorm1d(middle))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layer - 1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.lambd = 1.0

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x


class CGDM(nn.Module):
    """
    CGDM model container: feature extractor + two classifiers.
    """
    def __init__(self, input_dim=8, num_classes=6):
        super(CGDM, self).__init__()
        self.feature_extractor = CGDMBackbone(input_dim=input_dim)
        self.classifier1 = CGDMClassifier(num_classes=num_classes, num_layer=2, num_unit=self.feature_extractor.output_num(), middle=1000)
        self.classifier2 = CGDMClassifier(num_classes=num_classes, num_layer=2, num_unit=self.feature_extractor.output_num(), middle=1000)

    def forward(self, x):
        feat = self.feature_extractor(x)
        out1 = self.classifier1(feat)
        out2 = self.classifier2(feat)
        return out1, out2

    def predict(self, x):
        out1, out2 = self.forward(x)
        return out1 + out2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))


def Entropy_div(input_):
    epsilon = 1e-5
    input_ = torch.mean(input_, 0) + epsilon
    entropy = input_ * torch.log(input_)
    entropy = torch.sum(entropy)
    return entropy


def Entropy_condition(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1).mean()
    return entropy


def Entropy(input_):
    return Entropy_condition(input_) + Entropy_div(input_)


def Weighted_CrossEntropy(input_, labels):
    input_s = F.softmax(input_, dim=1)
    entropy = -input_s * torch.log(input_s + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    weight = 1.0 + torch.exp(-entropy)
    weight = weight / torch.sum(weight).detach().item()
    return torch.mean(weight * nn.CrossEntropyLoss(reduction='none')(input_, labels))


def obtain_label(loader, netE, netC1, netC2, device):
    start_test = True
    netE.eval()
    netC1.eval()
    netC2.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data[0], data[1]
            inputs = inputs.to(device)
            feas = netE(inputs)
            outputs1 = netC1(feas)
            outputs2 = netC2(feas)
            outputs = outputs1 + outputs2
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    label_np = all_label.float().cpu().numpy()
    has_labels = np.unique(label_np).size > 1
    if has_labels:
        accuracy = torch.sum(predict.to(device).float() == all_label.to(device)).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    if has_labels:
        acc = np.sum(pred_label == label_np) / len(all_fea)

    for _ in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        if has_labels:
            acc = np.sum(pred_label == label_np) / len(all_fea)

    return pred_label.astype('int')


def gradient_discrepancy_loss_margin(p_s1, p_s2, s_y, p_t1, p_t2, t_y, netE, netC1, netC2):
    loss_w = Weighted_CrossEntropy
    loss = nn.CrossEntropyLoss()

    src_loss1 = loss(p_s1, s_y)
    tgt_loss1 = loss_w(p_t1, t_y)

    src_loss2 = loss(p_s2, s_y)
    tgt_loss2 = loss_w(p_t2, t_y)

    grad_cossim11 = []
    grad_cossim22 = []

    for _, p in netC1.named_parameters():
        real_grad = grad([src_loss1], [p], create_graph=True, only_inputs=True, allow_unused=False)[0]
        fake_grad = grad([tgt_loss1], [p], create_graph=True, only_inputs=True, allow_unused=False)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        grad_cossim11.append(_cossim)

    grad_cossim1 = torch.stack(grad_cossim11)
    gm_loss1 = (1.0 - grad_cossim1).mean()

    for _, p in netC2.named_parameters():
        real_grad = grad([src_loss2], [p], create_graph=True, only_inputs=True)[0]
        fake_grad = grad([tgt_loss2], [p], create_graph=True, only_inputs=True)[0]

        if len(p.shape) > 1:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=1).mean()
        else:
            _cossim = F.cosine_similarity(fake_grad, real_grad, dim=0)
        grad_cossim22.append(_cossim)

    grad_cossim2 = torch.stack(grad_cossim22)
    gm_loss2 = (1.0 - grad_cossim2).mean()
    gm_loss = (gm_loss1 + gm_loss2) / 2.0

    return gm_loss


def train_cgdm(model, X_source, y_source, X_target, y_target=None,
               X_val=None, y_val=None,
               epochs=20, batch_size=64, lr=1e-4, weight_decay=5e-4, num_k=4, log_interval=100,
               patience=20,
               device='cuda' if torch.cuda.is_available() else 'cpu'):
    import copy
    from tqdm import tqdm

    model = model.to(device)

    if y_target is None:
        y_target = np.zeros(len(X_target), dtype=np.int64)

    X_source_t = torch.tensor(X_source, dtype=torch.float32)
    y_source_t = torch.tensor(y_source, dtype=torch.long)
    X_target_t = torch.tensor(X_target, dtype=torch.float32)
    y_target_t = torch.tensor(y_target, dtype=torch.long)

    class IndexedTensorDataset(torch.utils.data.Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor

        def __len__(self):
            return len(self.x)

        def __getitem__(self, index):
            return self.x[index], self.y[index], index

    train_dataset = torch.utils.data.TensorDataset(X_source_t, y_source_t)
    test_dataset = IndexedTensorDataset(X_target_t, y_target_t)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader_eval = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    val_loader = None
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    G = model.feature_extractor
    F1 = model.classifier1
    F2 = model.classifier2

    F1.apply(weights_init)
    F2.apply(weights_init)

    optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_f = torch.optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=lr, weight_decay=weight_decay)

    start = 0
    criterion = nn.CrossEntropyLoss()

    def _eval_val():
        if val_loader is None:
            return None, None
        model.eval()
        all_probs, all_targets = [], []
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model.predict(xb)
                total_loss += criterion(logits, yb).item()
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                all_targets.append(yb.cpu().numpy())
        total_loss /= max(1, len(val_loader))
        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        try:
            val_auroc = roc_auc_score(all_targets, all_probs[:, 1]) if all_probs.shape[1] == 2 \
                else roc_auc_score(all_targets, all_probs, multi_class='ovr')
        except Exception:
            val_auroc = 0.5
        return total_loss, val_auroc

    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    mem_label = None
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="CGDM Training")
    for ep in epoch_iterator:
        from itertools import cycle
        iter_source = iter(train_loader)
        iter_target = cycle(test_loader)
        steps = len(train_loader)

        # Pseudo labels are required when ep > start.
        # Ensure we bootstrap once before first use, then refresh every 3 epochs.
        if ep > start and (mem_label is None or ep % 3 == 0):
            mem_label = obtain_label(test_loader_eval, G, F1, F2, device)
            mem_label = torch.from_numpy(mem_label).to(device)

        for batch_idx in range(steps - 1):
            G.train(); F1.train(); F2.train()

            try:
                data_s, label_s = next(iter_source)
            except StopIteration:
                iter_source = iter(train_loader)
                data_s, label_s = next(iter_source)

            data_t, label_t, idx_t = next(iter_target)

            if ep > start:
                pseudo_label_t = mem_label[idx_t].to(device)

            data_s, label_s = data_s.to(device), label_s.to(device)
            data_t, label_t = data_t.to(device), label_t.to(device)
            data_all = torch.cat((data_s, data_t), 0)
            bs = len(label_s)

            # Step A: train G, F1, F2 jointly
            optimizer_g.zero_grad(); optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output); output2 = F2(output)
            output_s1, output_s2 = output1[:bs], output2[:bs]
            output_t1, output_t2 = output1[bs:], output2[bs:]
            output_t1_s = F.softmax(output_t1, dim=1)
            output_t2_s = F.softmax(output_t2, dim=1)

            entropy_loss = Entropy(output_t1_s) + Entropy(output_t2_s)
            supervision_loss = (Weighted_CrossEntropy(output_t1, pseudo_label_t) +
                                Weighted_CrossEntropy(output_t2, pseudo_label_t)) if ep > start else 0
            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            all_loss = loss1 + loss2 + 0.01 * entropy_loss + 0.01 * supervision_loss
            all_loss.backward()
            optimizer_g.step(); optimizer_f.step()

            # Step B: train F1, F2 to maximize discrepancy
            optimizer_g.zero_grad(); optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output); output2 = F2(output)
            output_s1, output_s2 = output1[:bs], output2[:bs]
            output_t1, output_t2 = output1[bs:], output2[bs:]
            output_t1_s = F.softmax(output_t1, dim=1)
            output_t2_s = F.softmax(output_t2, dim=1)

            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)
            entropy_loss = Entropy(output_t1_s) + Entropy(output_t2_s)
            loss_dis = discrepancy(output_t1, output_t2)
            all_loss = loss1 + loss2 - 1.0 * loss_dis + 0.01 * entropy_loss
            all_loss.backward()
            optimizer_f.step()

            # Step C: train G to minimize discrepancy (num_k steps)
            for _ in range(num_k):
                optimizer_g.zero_grad(); optimizer_f.zero_grad()
                output = G(data_all)
                output1 = F1(output); output2 = F2(output)
                output_s1, output_s2 = output1[:bs], output2[:bs]
                output_t1, output_t2 = output1[bs:], output2[bs:]
                output_t1_s = F.softmax(output_t1, dim=1)
                output_t2_s = F.softmax(output_t2, dim=1)

                entropy_loss = Entropy(output_t1_s) + Entropy(output_t2_s)
                loss_dis = discrepancy(output_t1, output_t2)
                gmn_loss = gradient_discrepancy_loss_margin(
                    output_s1, output_s2, label_s, output_t1, output_t2, pseudo_label_t,
                    G, F1, F2) if ep > start else 0

                all_loss = 1.0 * loss_dis + 0.01 * entropy_loss + 0.01 * gmn_loss
                all_loss.backward()
                optimizer_g.step()

        val_loss, val_auroc = _eval_val()
        if val_loss is not None:
            epoch_iterator.set_postfix({'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'})
            epoch_num = ep + 1
            epochs_ran = epoch_num
            epoch_history.append({
                'epoch': epoch_num,
                'val_loss': round(float(val_loss), 6),
                'val_auroc': round(float(val_auroc), 6),
            })
            if val_auroc > best_val_score:
                best_val_score = val_auroc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_epoch = epoch_num
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    epoch_iterator.write(f"Early stopping at epoch {ep}")
                    early_stopped = True
                    early_stop_epoch = epoch_num
                    break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer="Adam",
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran if epochs_ran else epochs,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={"num_k": num_k},
    )
    return model


class DeepCORAL(DAModel):
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DeepCORAL, self).__init__(input_dim, num_classes, hparams)
        
    # Standard forward uses DAModel.predict


class DAN(DAModel):
    """
    Deep Adaptation Network (DAN).
    Uses MK-MMD to align source/target feature distributions.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(DAN, self).__init__(input_dim, num_classes, hparams)


# --- Training Logic ---

def train_standard(model, X_train, y_train, X_val, y_val,
                   epochs=50, batch_size=64, lr=1e-3, patience=5,
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   X_test=None, y_test=None):
    """
    Simple supervised training wrapper (used for fallback in DA methods).
    """
    from src.models import train_torch_model
    return train_torch_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        device=device,
        X_test=X_test,
        y_test=y_test,
    )

def _infinite_iterator(loader):
    while True:
        for batch in loader:
            yield batch


def _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size):
    pin = torch.cuda.is_available()
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin, num_workers=4, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=2, persistent_workers=True)

    target_loader = None
    if X_target is not None:
        target_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_target, dtype=torch.float32)
        )
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin, num_workers=2, persistent_workers=True)

    return train_loader, val_loader, target_loader


def _evaluate_val(model, val_loader, device):
    class_criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0.0
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model.predict(X_batch)
            val_loss += class_criterion(logits, y_batch).item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_probs.append(probs)
            val_targets.append(y_batch.cpu().numpy())
    val_loss /= len(val_loader)
    val_probs = np.concatenate(val_probs, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    try:
        if val_probs.shape[1] == 2:
            val_auroc = roc_auc_score(val_targets, val_probs[:, 1])
        else:
            val_auroc = roc_auc_score(val_targets, val_probs, multi_class='ovr', average='macro')
    except:
        val_auroc = 0.5
    return val_loss, val_auroc


def _evaluate_test(model, X_test, y_test, device, batch_size=1024):
    if X_test is None or y_test is None:
        return None, None
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    all_targets = []
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model.predict(X_batch)
            probs = torch.softmax(logits, dim=1)
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


def train_dann(model, X_train, y_train, d_train, X_val, y_val, d_val,
               epochs=50, batch_size=64, lr=1e-3, patience=5,
               device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
               X_test=None, y_test=None):
    """
    DANN (TLL-style): L = L_cls(source) + lambda * DomainAdversarialLoss(f_s, f_t).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("DANN requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    disc_hidden = model.hparams.get('disc_hidden', 1024)
    disc_lr = model.hparams.get('discriminator_lr', lr)
    trade_off = model.hparams.get('trade_off', 1.0)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    feature_dim = model.feature_extractor.output_dim
    domain_discriminator = DomainDiscriminator(feature_dim, disc_hidden, batch_norm=True, sigmoid=True).to(device)
    grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=epochs * len(train_loader), auto_step=True)
    domain_adv = DomainAdversarialLoss(domain_discriminator, grl=grl, sigmoid=True)

    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters()},
            {"params": model.classifier.parameters()},
            {"params": domain_discriminator.parameters(), "lr": disc_lr},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )
    class_criterion = nn.CrossEntropyLoss()

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="DANN Training")
    for epoch in epoch_iterator:
        model.train()
        domain_discriminator.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            logits_s = model.classifier(f_s)

            cls_loss = class_criterion(logits_s, y_s)
            transfer_loss = domain_adv(f_s, f_t)
            loss = cls_loss + trade_off * transfer_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={"discriminator_lr": disc_lr},
    )
    return model


def train_cdan(model, X_train, y_train, d_train, X_val, y_val, d_val,
               epochs=50, batch_size=64, lr=1e-3, patience=5,
               device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
               X_test=None, y_test=None):
    """
    CDAN (TLL-style): L = L_cls(source) + lambda * ConditionalDomainAdversarialLoss(g_s, f_s, g_t, f_t).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("CDAN requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    disc_hidden = model.hparams.get('disc_hidden', 1024)
    disc_lr = model.hparams.get('discriminator_lr', lr)
    trade_off = model.hparams.get('trade_off', 1.0)
    randomized = model.hparams.get('cdan_randomized', False)
    entropy_conditioning = model.hparams.get('cdan_entropy', True)
    randomized_dim = model.hparams.get('cdan_randomized_dim', 1024)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    feature_dim = model.feature_extractor.output_dim
    num_classes = model.classifier[-1].out_features
    disc_in_dim = randomized_dim if randomized else feature_dim * num_classes
    domain_discriminator = DomainDiscriminator(disc_in_dim, disc_hidden, batch_norm=True, sigmoid=True).to(device)
    cdan_loss = ConditionalDomainAdversarialLoss(
        domain_discriminator,
        entropy_conditioning=entropy_conditioning,
        randomized=randomized,
        num_classes=num_classes,
        features_dim=feature_dim,
        randomized_dim=randomized_dim,
        sigmoid=True,
    )
    cdan_loss.grl.max_iters = epochs * len(train_loader)

    optimizer = torch.optim.Adam(
        [
            {"params": model.feature_extractor.parameters()},
            {"params": model.classifier.parameters()},
            {"params": domain_discriminator.parameters(), "lr": disc_lr},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )
    class_criterion = nn.CrossEntropyLoss()

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="CDAN Training")
    for epoch in epoch_iterator:
        model.train()
        domain_discriminator.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            g_s = model.classifier(f_s)
            g_t = model.classifier(f_t)

            cls_loss = class_criterion(g_s, y_s)
            transfer_loss = cdan_loss(g_s, f_s, g_t, f_t)
            loss = cls_loss + trade_off * transfer_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={"discriminator_lr": disc_lr},
    )
    return model


def train_adversarial_da(model, X_train, y_train, d_train, X_val, y_val, d_val,
                         epochs=50, batch_size=64, lr=1e-3, patience=5,
                         device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
                         X_test=None, y_test=None):
    """
    Backward-compatible wrapper: dispatch to DANN/CDAN depending on model type.
    """
    if isinstance(model, CDAN):
        return train_cdan(model, X_train, y_train, d_train, X_val, y_val, d_val,
                          epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device,
                          X_target=X_target, X_test=X_test, y_test=y_test)
    if isinstance(model, DANN):
        return train_dann(model, X_train, y_train, d_train, X_val, y_val, d_val,
                          epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device,
                          X_target=X_target, X_test=X_test, y_test=y_test)
    raise ValueError("train_adversarial_da supports DANN or CDAN models only.")


def train_mcc(model, X_train, y_train, d_train, X_val, y_val, d_val,
              epochs=50, batch_size=64, lr=1e-3, patience=5,
              device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
              X_test=None, y_test=None):
    """
    MCC (TLL-style): L = L_cls(source) + mu * MCC(target_logits).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("MCC requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    trade_off = model.hparams.get('mcc_trade_off', 1.0)
    mcc_loss_fn = MinimumClassConfusionLoss(model.temperature)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="MCC Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            logits_s = model.predict(X_s)
            logits_t = model.predict(X_t)

            cls_loss = class_criterion(logits_s, y_s)
            transfer_loss = mcc_loss_fn(logits_t)
            loss = cls_loss + trade_off * transfer_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)
    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
    )
    return model


def train_dan(model, X_train, y_train, d_train, X_val, y_val, d_val,
              epochs=50, batch_size=64, lr=1e-3, patience=5,
              device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
              X_test=None, y_test=None):
    """
    DAN (TLL-style): L = L_cls(source) + lambda * MK-MMD(f_s, f_t).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("DAN requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    trade_off = model.hparams.get('dan_trade_off', 1.0)
    kernel_num = model.hparams.get('dan_kernel_num', 5)
    linear = model.hparams.get('dan_linear', False)

    kernels = [GaussianKernel(alpha=2 ** k) for k in range(kernel_num)]
    mkmmd = MultipleKernelMaximumMeanDiscrepancy(kernels, linear=linear)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="DAN Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()

            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            logits_s = model.classifier(f_s)

            cls_loss = class_criterion(logits_s, y_s)
            transfer_loss = mkmmd(f_s, f_t)
            loss = cls_loss + trade_off * transfer_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)
    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
    )
    return model

class ADDA(nn.Module):
    """
    Adversarial Discriminative Domain Adaptation (Tzeng et al., 2017).
    Separate source/target encoders with adversarial discriminator.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(ADDA, self).__init__()
        self.hparams = hparams if hparams else {}

        # Source model (encoder + classifier)
        self.source_model = DAModel(input_dim, num_classes, hparams)

        # Target encoder initialized from source encoder
        import copy
        self.target_encoder = copy.deepcopy(self.source_model.feature_extractor)

        # Domain discriminator (binary, sigmoid output)
        feature_dim = self.source_model.feature_extractor.output_dim
        disc_hidden = self.hparams.get('disc_hidden', 1024)
        self.discriminator = DomainDiscriminator(feature_dim, disc_hidden, batch_norm=True, sigmoid=True)

    def predict(self, x):
        feat = self.target_encoder(x)
        return self.source_model.classifier(feat)

    def forward(self, x):
        return self.predict(x)


def train_adda(model, X_train, y_train, d_train, X_val, y_val, d_val,
               epochs=50, batch_size=64, lr=1e-3, patience=5,
               device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
               X_test=None, y_test=None):
    """
    ADDA training:
      1) Pretrain source encoder+classifier on labeled source.
      2) Adversarially train target encoder to fool discriminator.
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("ADDA requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    model.discriminator = model.discriminator.to(device)

    from src.models import train_torch_model

    # Phase 1: source pretraining
    model.source_model = train_torch_model(
        model.source_model, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device,
        X_test=X_test, y_test=y_test
    )

    # Init target encoder from source
    model.target_encoder.load_state_dict(model.source_model.feature_extractor.state_dict())

    # Freeze source encoder and classifier
    for p in model.source_model.parameters():
        p.requires_grad = False

    weight_decay = model.hparams.get('weight_decay', 0.0)
    disc_lr = model.hparams.get('discriminator_lr', lr)
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=disc_lr, weight_decay=weight_decay)
    opt_t = torch.optim.Adam(model.target_encoder.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()

    # Dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    target_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    target_iter = _infinite_iterator(target_loader)

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    source_pretrain_info = dict(getattr(model.source_model, "_training_info", {}))
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="ADDA Training")
    for epoch in epoch_iterator:
        model.target_encoder.train()
        model.discriminator.train()
        train_loss = 0.0

        for X_s, _ in train_loader:
            X_s = X_s.to(device, non_blocking=True)
            X_t, = next(target_iter)
            X_t = X_t.to(device, non_blocking=True)

            # 1) Train discriminator
            with torch.no_grad():
                f_s = model.source_model.feature_extractor(X_s)
            f_t = model.target_encoder(X_t)

            d_in = torch.cat([f_s, f_t], dim=0)
            d_out = model.discriminator(d_in)
            d_labels = torch.cat([
                torch.ones((f_s.size(0), 1), device=device),
                torch.zeros((f_t.size(0), 1), device=device)
            ], dim=0)

            opt_d.zero_grad()
            loss_d = bce(d_out, d_labels)
            loss_d.backward()
            opt_d.step()

            # 2) Train target encoder to fool discriminator
            f_t = model.target_encoder(X_t)
            d_out_t = model.discriminator(f_t)
            fool_labels = torch.ones((f_t.size(0), 1), device=device)
            opt_t.zero_grad()
            loss_t = bce(d_out_t, fool_labels)
            loss_t.backward()
            opt_t.step()

            train_loss += (loss_d.item() + loss_t.item())

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer=f"{opt_d.__class__.__name__}+{opt_t.__class__.__name__}",
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={
            "discriminator_lr": disc_lr,
            "phases": {
                "source_pretrain": source_pretrain_info,
                "target_adaptation": {"epochs_ran": epochs_ran},
            },
        },
    )
    return model



class MCD(DAModel):
    """
    Maximum Classifier Discrepancy (MCD).
    Two classifiers over a shared feature extractor.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(MCD, self).__init__(input_dim, num_classes, hparams)
        feat_dim = self.feature_extractor.output_dim
        self.classifier1 = nn.Linear(feat_dim, num_classes)
        self.classifier2 = nn.Linear(feat_dim, num_classes)
        self.classifier = None  # override

    def predict(self, x):
        feat = self.feature_extractor(x)
        o1 = self.classifier1(feat)
        o2 = self.classifier2(feat)
        return (o1 + o2) / 2.0

    def forward(self, x):
        feat = self.feature_extractor(x)
        o1 = self.classifier1(feat)
        o2 = self.classifier2(feat)
        return o1, o2


class MCDInferenceWrapper(nn.Module):
    def __init__(self, mcd_model):
        super().__init__()
        self.model = mcd_model

    def forward(self, x):
        o1, o2 = self.model(x)
        return (o1 + o2) / 2.0


def train_mcd(model, X_train, y_train, d_train, X_val, y_val, d_val,
              epochs=50, batch_size=64, lr=1e-3, patience=5,
              device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
              X_test=None, y_test=None):
    """
    MCD (TLL-style):
      A) Min CE on source (G, C1, C2)
      B) Max discrepancy on target (C1, C2) while fitting source
      C) Min discrepancy on target (G)
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("MCD requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    trade_off = model.hparams.get('mcd_trade_off', 1.0)
    k_steps = model.hparams.get('mcd_k', 4)

    optimizer_g = torch.optim.Adam(model.feature_extractor.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_c = torch.optim.Adam(
        list(model.classifier1.parameters()) + list(model.classifier2.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="MCD Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            # Step A: train G, C1, C2 on source
            optimizer_g.zero_grad()
            optimizer_c.zero_grad()
            f_s = model.feature_extractor(X_s)
            o1_s = model.classifier1(f_s)
            o2_s = model.classifier2(f_s)
            loss_s = criterion(o1_s, y_s) + criterion(o2_s, y_s)
            loss_s.backward()
            optimizer_g.step()
            optimizer_c.step()

            # Step B: train C1, C2 to maximize discrepancy on target
            optimizer_c.zero_grad()
            f_s_det = model.feature_extractor(X_s).detach()
            f_t_det = model.feature_extractor(X_t).detach()
            o1_s = model.classifier1(f_s_det)
            o2_s = model.classifier2(f_s_det)
            o1_t = model.classifier1(f_t_det)
            o2_t = model.classifier2(f_t_det)
            loss_s = criterion(o1_s, y_s) + criterion(o2_s, y_s)
            dis = classifier_discrepancy(F.softmax(o1_t, dim=1), F.softmax(o2_t, dim=1))
            loss_c = loss_s - trade_off * dis
            loss_c.backward()
            optimizer_c.step()

            # Step C: train G to minimize discrepancy on target
            for _ in range(k_steps):
                optimizer_g.zero_grad()
                f_t = model.feature_extractor(X_t)
                o1_t = model.classifier1(f_t)
                o2_t = model.classifier2(f_t)
                dis = classifier_discrepancy(F.softmax(o1_t, dim=1), F.softmax(o2_t, dim=1))
                loss_g = trade_off * dis
                loss_g.backward()
                optimizer_g.step()

            train_loss += loss_s.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)
    _finalize_training_metadata(
        model,
        optimizer=f"{optimizer_g.__class__.__name__}+{optimizer_c.__class__.__name__}",
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={"mcd_k": k_steps},
    )
    return model
    
# --- JAN: Joint Adaptation Network ---

class JAN(DAModel):
    """
    Joint Adaptation Network (Long et al., 2017)
    Aligns joint distributions via JMMD.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(JAN, self).__init__(input_dim, num_classes, hparams)
        self.num_classes = num_classes

    def forward(self, x):
        return self.predict(x)


def train_jan(model, X_train, y_train, d_train, X_val, y_val, d_val,
              epochs=50, batch_size=64, lr=1e-3, patience=5,
              device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
              X_test=None, y_test=None):
    """
    JAN (TLL-style): L = L_cls(source) + lambda * JMMD((f_s, p_s), (f_t, p_t)).
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("JAN requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    trade_off = model.hparams.get('jmmd_lambda', 1.0)
    kernel_num = model.hparams.get('jan_kernel_num', 5)
    linear = model.hparams.get('jan_linear', True)

    kernels = [GaussianKernel(alpha=2 ** k) for k in range(kernel_num)]
    jmmd = JointMultipleKernelMaximumMeanDiscrepancy(kernels=(kernels, kernels), linear=linear)

    train_loader, val_loader, target_loader = _build_loaders(X_train, y_train, X_val, y_val, X_target, batch_size)
    target_iter = _infinite_iterator(target_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_criterion = nn.CrossEntropyLoss()

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(epochs), desc="JAN Training")
    for epoch in epoch_iterator:
        model.train()
        train_loss = 0.0

        for X_s, y_s in train_loader:
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, = next(target_iter)
            X_t = X_t.to(device)

            optimizer.zero_grad()
            f_s = model.feature_extractor(X_s)
            f_t = model.feature_extractor(X_t)
            g_s = model.classifier(f_s)
            g_t = model.classifier(f_t)

            cls_loss = class_criterion(g_s, y_s)
            p_s = F.softmax(g_s, dim=1)
            p_t = F.softmax(g_t, dim=1)
            transfer_loss = jmmd((f_s, p_s), (f_t, p_t))
            loss = cls_loss + trade_off * transfer_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)
    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
    )
    return model

# --- SHOT: Source Hypothesis Transfer ---

class SHOT(DAModel):
    """
    SHOT: Source Hypothesis Transfer (Liang et al., 2020)
    Source-free Domain Adaptation.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(SHOT, self).__init__(input_dim, num_classes, hparams)
    
    def forward(self, x):
        return self.predict(x)

def _shot_op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def _shot_lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def _shot_obtain_label(loader, model, num_classes, distance='cosine', threshold=0, epsilon=1e-5, device='cuda'):
    start = True
    with torch.no_grad():
        for X_batch, idx in loader:
            X_batch = X_batch.to(device)
            feat = model.feature_extractor(X_batch)
            outputs = model.classifier(feat)
            if start:
                all_fea = feat.float().cpu()
                all_output = outputs.float().cpu()
                start = False
            else:
                all_fea = torch.cat((all_fea, feat.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = F.softmax(all_output, dim=1)
    ent = torch.sum(-all_output * torch.log(all_output + epsilon), dim=1)
    _unknown_weight = 1 - ent / np.log(num_classes)
    _, predict = torch.max(all_output, 1)

    if distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(num_classes)[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)[0]
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]
        aff = np.eye(num_classes)[predict]

    return predict.astype('int')


def train_shot(model, X_train, y_train, d_train, X_val, y_val, d_val,
               epochs=50, batch_size=64, lr=1e-3, patience=5,
               device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
               X_test=None, y_test=None):
    """
    SHOT (official-style):
      1) Train source model on labeled source.
      2) Freeze classifier, adapt feature extractor with pseudo-labels + InfoMax on target.
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("SHOT requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)

    # Phase 1: Source pretraining
    from src.models import train_torch_model
    model = train_torch_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device,
        X_test=X_test, y_test=y_test
    )
    source_pretrain_info = dict(getattr(model, "_training_info", {}))

    # Freeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = False

    # Hyperparams (from official defaults)
    cls_par = model.hparams.get('shot_cls_par', 0.3)
    ent_par = model.hparams.get('shot_ent_par', 1.0)
    gent = model.hparams.get('shot_gent', True)
    ent = model.hparams.get('shot_ent', True)
    threshold = model.hparams.get('shot_threshold', 0)
    distance = model.hparams.get('shot_distance', 'cosine')
    epsilon = model.hparams.get('shot_epsilon', 1e-5)
    adapt_epochs = model.hparams.get('shot_epochs', max(1, epochs // 2))
    interval = model.hparams.get('shot_interval', 15)
    lr_decay = model.hparams.get('shot_lr_decay', 0.1)

    # Target loaders with indices
    target_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_target, dtype=torch.float32),
        torch.arange(len(X_target), dtype=torch.long)
    )
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    target_loader_eval = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size * 3, shuffle=False, drop_last=False)

    # Validation loader (source)
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer (SGD + official LR schedule)
    optimizer = torch.optim.SGD(model.feature_extractor.parameters(), lr=lr * lr_decay)
    optimizer = _shot_op_copy(optimizer)

    max_iter = adapt_epochs * len(target_loader)
    interval_iter = max_iter // interval if interval > 0 else max_iter
    iter_num = 0

    import copy
    from tqdm import tqdm
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    epoch_iterator = tqdm(range(adapt_epochs), desc="SHOT Adaptation")
    target_iter = iter(target_loader)
    mem_label = None

    for epoch in epoch_iterator:
        model.train()
        model.feature_extractor.train()
        model.classifier.eval()

        train_loss = 0.0
        for _ in range(len(target_loader)):
            try:
                X_t, idx = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                X_t, idx = next(target_iter)

            if X_t.size(0) == 1:
                continue

            if iter_num % interval_iter == 0 and cls_par > 0:
                model.eval()
                mem_label = _shot_obtain_label(
                    target_loader_eval, model, model.classifier[-1].out_features,
                    distance=distance, threshold=threshold, epsilon=epsilon, device=device
                )
                mem_label = torch.from_numpy(mem_label).to(device)
                model.train()
                model.classifier.eval()

            X_t = X_t.to(device)
            idx = idx.to(device)

            iter_num += 1
            _shot_lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            features_t = model.feature_extractor(X_t)
            outputs_t = model.classifier(features_t)

            loss = torch.tensor(0.0, device=device)
            if cls_par > 0 and mem_label is not None:
                pseudo = mem_label[idx]
                loss = loss + cls_par * nn.CrossEntropyLoss()(outputs_t, pseudo)

            if ent:
                softmax_out = F.softmax(outputs_t, dim=1)
                entropy_loss = torch.mean(tllib_entropy(softmax_out))
                if gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + epsilon))
                    entropy_loss -= gentropy_loss
                loss = loss + ent_par * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(target_loader))
        val_loss, val_auroc = _evaluate_val(model, val_loader, device)

        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        epoch_num = epoch + 1
        epochs_ran = epoch_num

        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch} (Best AUROC: {best_val_score:.4f})")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

        postfix = {'Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val AUC': f'{val_auroc:.4f}'}
        if test_acc is not None:
            postfix.update({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})
        epoch_iterator.set_postfix(postfix)
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(train_loss), 6),
            'val_loss': round(float(val_loss), 6),
            'val_auroc': round(float(val_auroc), 6),
            'test_accuracy': round(float(test_acc), 6) if test_acc is not None else None,
            'test_auroc': round(float(test_auroc), 6) if test_auroc is not None else None,
        })

    if best_model_state:
        model.load_state_dict(best_model_state)

    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=adapt_epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr * lr_decay,
        weight_decay=1e-3,
        best_metric_value=round(float(best_val_score), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        extra={
            "phases": {
                "source_pretrain": source_pretrain_info,
                "target_adaptation": {"epochs_ran": epochs_ran},
            },
            "shot_interval": interval,
        },
    )
    return model

# CBST (Class-Balanced Self-Training)
class CBST(DAModel):
    """
    CBST: Class-Balanced Self-Training (Zou et al., 2018)
    Iterative self-training with class-balanced pseudo-label selection.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super(CBST, self).__init__(input_dim, num_classes, hparams)


def train_cbst(model, X_train, y_train, d_train, X_val, y_val, d_val,
               epochs=50, batch_size=64, lr=1e-3, patience=5,
               device='cuda' if torch.cuda.is_available() else 'cpu', X_target=None,
               X_test=None, y_test=None):
    """
    CBST (official-style):
      1) Train on source.
      2) Iteratively select class-balanced pseudo-labels on target and retrain.
    Requires unlabeled target samples (X_target).
    """
    if X_target is None:
        raise ValueError("CBST requires X_target for UDA. Use --uda to provide target samples.")

    model = model.to(device)
    weight_decay = model.hparams.get('weight_decay', 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Source data
    X_s = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_s = torch.tensor(y_train, dtype=torch.long).to(device)
    ds_s = torch.utils.data.TensorDataset(X_s, y_s)
    loader_s = torch.utils.data.DataLoader(ds_s, batch_size=batch_size, shuffle=True, drop_last=True)

    # Target data (unlabeled)
    X_t = torch.tensor(X_target, dtype=torch.float32).to(device)

    # 1) Pretrain on source
    pretrain_epochs = model.hparams.get('cbst_pretrain_epochs', max(1, epochs // 2))
    from tqdm import tqdm
    epoch_iterator = tqdm(range(pretrain_epochs), desc="CBST Pretrain")
    for epoch in epoch_iterator:
        model.train()
        for x, y in loader_s:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        test_acc, test_auroc = _evaluate_test(model, X_test, y_test, device)
        if test_acc is not None:
            epoch_iterator.set_postfix({'Test Acc': f'{test_acc:.4f}', 'Test AUC': f'{test_auroc:.4f}'})

    # 2) Iterative self-training with class-balanced thresholds
    max_iter = model.hparams.get('cbst_max_iter', 5)
    init_port = model.hparams.get('cbst_init_port', 0.2)
    port_step = model.hparams.get('cbst_port_step', 0.1)
    max_port = model.hparams.get('cbst_max_port', 0.8)
    retrain_epochs = model.hparams.get('cbst_retrain_epochs', 5)

    num_classes = model.classifier[-1].out_features

    for round_idx in range(max_iter):
        model.eval()
        with torch.no_grad():
            logits_t = model(X_t)
            probs_t = F.softmax(logits_t, dim=1)
            max_probs, preds = torch.max(probs_t, dim=1)

        current_port = min(init_port + round_idx * port_step, max_port)
        pseudo_idx = []
        pseudo_labels = []

        for c in range(num_classes):
            c_idx = (preds == c).nonzero(as_tuple=True)[0]
            if len(c_idx) == 0:
                continue
            c_probs = max_probs[c_idx]
            k = int(len(c_idx) * current_port)
            if k == 0:
                continue
            topk_vals, topk_indices = torch.topk(c_probs, k)
            global_indices = c_idx[topk_indices]
            pseudo_idx.append(global_indices)
            pseudo_labels.append(torch.full((k,), c, dtype=torch.long).to(device))

        if not pseudo_idx:
            print("CBST: No pseudo-labels selected, skipping round.")
            continue

        pseudo_idx_cat = torch.cat(pseudo_idx)
        pseudo_labels_cat = torch.cat(pseudo_labels)
        X_pseudo = X_t[pseudo_idx_cat]

        # Retrain on source + pseudo-target
        X_aug = torch.cat([X_s, X_pseudo])
        y_aug = torch.cat([y_s, pseudo_labels_cat])
        ds_aug = torch.utils.data.TensorDataset(X_aug, y_aug)
        loader_aug = torch.utils.data.DataLoader(ds_aug, batch_size=batch_size, shuffle=True, drop_last=True)

        print(f"CBST: Round {round_idx+1}/{max_iter} - Retraining with {len(X_pseudo)} pseudo-labels")
        model.train()
        for _ in range(retrain_epochs):
            for x, y in loader_aug:
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

    total_epochs = pretrain_epochs + max_iter * retrain_epochs
    _finalize_training_metadata(
        model,
        optimizer=optimizer.__class__.__name__,
        best_epoch=None,
        early_stopped=False,
        early_stop_epoch=None,
        epochs_ran=total_epochs,
        max_epochs=total_epochs,
        batch_size=batch_size,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        model_selection_metric="target_self_training",
        best_metric_value=None,
        epoch_history=[],
        extra={
            "cbst_pretrain_epochs": pretrain_epochs,
            "cbst_max_iter": max_iter,
            "cbst_retrain_epochs": retrain_epochs,
        },
    )
    return model
