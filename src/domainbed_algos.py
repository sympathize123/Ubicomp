import copy
import operator
from collections import OrderedDict
from itertools import cycle
from numbers import Number

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.backbones import MLPFeaturizer, ResNetFeaturizer, TransformerFeaturizer
from src.models import attach_training_metadata




class FeatureClassifier(nn.Module):
    def __init__(self, featurizer, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, x):
        feats = self.featurizer(x)
        return self.classifier(feats)

    def forward_features(self, x):
        return self.featurizer(x)


def _build_featurizer(input_dim, hparams):
    backbone_name = hparams.get('backbone', 'MLP')
    dropout = hparams.get('dropout', 0.3)
    hidden_dim = hparams.get('hidden_dim', 256)

    if backbone_name == 'MLP':
        return MLPFeaturizer(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=128,
            num_layers=hparams.get('num_layers', 3),
            dropout=dropout
        )
    if backbone_name == 'ResNet':
        return ResNetFeaturizer(
            input_dim,
            hidden_dim=hidden_dim,
            output_dim=128,
            num_blocks=hparams.get('num_blocks', 2),
            dropout=dropout
        )
    if backbone_name == 'Transformer':
        return TransformerFeaturizer(
            input_dim,
            hidden_dim=128,
            output_dim=128,
            num_layers=hparams.get('num_layers', 2),
            nhead=hparams.get('nhead', 4),
            dropout=dropout
        )

    raise ValueError(f"Unknown backbone: {backbone_name}")


def _build_classifier(in_dim, num_classes, hparams):
    nonlinear = hparams.get('nonlinear_classifier', False)
    dropout = hparams.get('classifier_dropout', 0.0)
    hidden_dim = hparams.get('classifier_hidden_dim', in_dim)
    if not nonlinear:
        return nn.Linear(in_dim, num_classes)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes)
    )


class DGModel(nn.Module):
    """
    Base class for Domain Generalization models.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hparams = hparams if hparams else {}
        self.num_domains = self.hparams.get('num_domains')

        self.featurizer = _build_featurizer(input_dim, self.hparams)
        self.classifier = _build_classifier(self.featurizer.output_dim, num_classes, self.hparams)
        self.network = FeatureClassifier(self.featurizer, self.classifier)

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)

    def update(self, minibatches, unlabeled=None, **kwargs):
        """
        Input: minibatches is a list of (x, y) pairs, one per domain.
        """
        raise NotImplementedError


class ERM(DGModel):
    """
    Empirical Risk Minimization (Standard Training)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get('lr', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )

    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class IRM(ERM):
    """
    Invariant Risk Minimization (Arjovsky et al., 2019)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = logits.device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        return torch.sum(grad_1 * grad_2)

    def update(self, minibatches, unlabeled=None, **kwargs):
        penalty_weight = (
            self.hparams.get('irm_lambda', 1.0)
            if self.update_count >= self.hparams.get('irm_penalty_anneal_iters', 0)
            else 1.0
        )
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0

        for x, y in minibatches:
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)

        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + penalty_weight * penalty

        if self.update_count == self.hparams.get('irm_penalty_anneal_iters', 0):
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams.get('lr', 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(), 'penalty': penalty.item()}


class VREx(ERM):
    """
    V-REx (Krueger et al., 2021)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, **kwargs):
        penalty_weight = (
            self.hparams.get('vrex_lambda', 1.0)
            if self.update_count >= self.hparams.get('vrex_penalty_anneal_iters', 0)
            else 1.0
        )

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches), device=all_x.device)

        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            losses[i] = F.cross_entropy(logits, y)

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = mean + penalty_weight * penalty

        if self.update_count == self.hparams.get('vrex_penalty_anneal_iters', 0):
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams.get('lr', 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': mean.item(), 'penalty': penalty.item()}


class GroupDRO(ERM):
    """
    Group DRO (Sagawa et al., 2020)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches, unlabeled=None, **kwargs):
        device = minibatches[0][0].device

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams.get("groupdro_eta", 0.01) * losses[m].data).exp()

        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class MixStyleLayer(nn.Module):
    """
    MixStyle layer (Zhou et al., 2021). Adapted to support 2D features.
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix="random"):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0)

        if x.dim() == 2:
            mu = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((B, 1)).to(x.device)
        else:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)

        if self.mix == "random":
            perm = torch.randperm(B)
        elif self.mix == "crossdomain":
            perm = torch.arange(B - 1, -1, -1)
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(perm_b.shape[0])]
            perm_a = perm_a[torch.randperm(perm_a.shape[0])]
            perm = torch.cat([perm_b, perm_a], 0)
        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


class MixStyle(DGModel):
    """
    MixStyle (Zhou et al., 2021). Training is ERM with MixStyle active.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.mixstyle = MixStyleLayer(
            p=self.hparams.get('mixstyle_p', 0.5),
            alpha=self.hparams.get('mixstyle_alpha', 0.1),
            eps=self.hparams.get('mixstyle_eps', 1e-6),
            mix=self.hparams.get('mixstyle_mix', 'random')
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get('lr', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )

    def forward(self, x):
        feats = self.featurizer(x)
        feats = self.mixstyle(feats)
        return self.classifier(feats)

    def predict(self, x):
        return self.forward(x)

    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}


class MLDG(ERM):
    """
    Meta-Learning for Domain Generalization (Li et al., 2018)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__(input_dim, num_classes, hparams)
        self.num_meta_test = self.hparams.get('n_meta_test', 1)

    def update(self, minibatches, unlabeled=None, **kwargs):
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in split_meta_train_test(minibatches, self.num_meta_test):
            inner_net = copy.deepcopy(self.network)
            inner_opt = torch.optim.Adam(
                inner_net.parameters(),
                lr=self.hparams.get("lr", 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )

            inner_obj = F.cross_entropy(inner_net(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            for p_tgt, p_src in zip(self.network.parameters(), inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            objective += inner_obj.item()

            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True)

            objective += (self.hparams.get('mldg_beta', 1.0) * loss_inner_j).item()

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(self.hparams.get('mldg_beta', 1.0) * g_j.data / num_mb)

        objective /= len(minibatches)

        self.optimizer.step()
        return {'loss': objective}


class ParamDict(OrderedDict):
    """ParamDict from DomainBed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        if isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


class WholeFish(nn.Module):
    def __init__(self, input_dim, num_classes, hparams, weights=None):
        super().__init__()
        featurizer = _build_featurizer(input_dim, hparams)
        classifier = _build_classifier(featurizer.output_dim, num_classes, hparams)
        self.net = FeatureClassifier(featurizer, classifier)
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class Fish(nn.Module):
    """
    Fish: Gradient Matching for Domain Generalization (Shi et al., 2021)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.network = WholeFish(input_dim, num_classes, self.hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = WholeFish(
            self.input_dim,
            self.num_classes,
            self.hparams,
            weights=self.network.state_dict()
        ).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams.get("lr", 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None, **kwargs):
        self.create_clone(minibatches[0][0].device)

        loss = None
        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams.get("meta_lr", self.hparams.get("fish_meta_lr", 0.5))
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item() if loss is not None else 0.0}

    def predict(self, x):
        return self.network(x)


class SagNet(nn.Module):
    """
    SagNet: Style Agnostic Networks (Nam et al., 2021)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.num_classes = num_classes

        self.network_f = _build_featurizer(input_dim, self.hparams)
        self.network_c = _build_classifier(self.network_f.output_dim, num_classes, self.hparams)
        self.network_s = _build_classifier(self.network_f.output_dim, num_classes, self.hparams)

        def opt(p):
            return torch.optim.Adam(
                p,
                lr=self.hparams.get("lr", 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )

        self.optimizer_f = opt(self.network_f.parameters())
        self.optimizer_c = opt(self.network_c.parameters())
        self.optimizer_s = opt(self.network_s.parameters())
        self.weight_adv = self.hparams.get("sag_w_adv", 1.0)

    def forward_c(self, x):
        return self.network_c(self.randomize(self.network_f(x), "style"))

    def forward_s(self, x):
        return self.network_s(self.randomize(self.network_f(x), "content"))

    def randomize(self, x, what="style", eps=1e-5):
        sizes = x.size()
        alpha = torch.rand(sizes[0], 1, device=x.device)

        if len(sizes) == 4:
            x = x.view(sizes[0], sizes[1], -1)
            alpha = alpha.unsqueeze(-1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        x = (x - mean) / (var + eps).sqrt()

        idx_swap = torch.randperm(sizes[0])
        if what == "style":
            mean = alpha * mean + (1 - alpha) * mean[idx_swap]
            var = alpha * var + (1 - alpha) * var[idx_swap]
        else:
            x = x[idx_swap].detach()

        x = x * (var + eps).sqrt() + mean
        return x.view(*sizes)

    def update(self, minibatches, unlabeled=None, **kwargs):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        self.optimizer_f.zero_grad()
        self.optimizer_c.zero_grad()
        loss_c = F.cross_entropy(self.forward_c(all_x), all_y)
        loss_c.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        self.optimizer_s.zero_grad()
        loss_s = F.cross_entropy(self.forward_s(all_x), all_y)
        loss_s.backward()
        self.optimizer_s.step()

        self.optimizer_f.zero_grad()
        loss_adv = -F.log_softmax(self.forward_s(all_x), dim=1).mean(1).mean()
        loss_adv = loss_adv * self.weight_adv
        loss_adv.backward()
        self.optimizer_f.step()

        return {'loss_c': loss_c.item(), 'loss_s': loss_s.item(), 'loss_adv': loss_adv.item()}

    def predict(self, x):
        return self.network_c(self.network_f(x))


class CSDHead(nn.Module):
    def __init__(self, feature_dim, num_classes, num_domains, k=2):
        super().__init__()
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.k = k

        self.sms = nn.Parameter(torch.normal(0, 1e-1, size=[k, feature_dim, num_classes]))
        self.sm_biases = nn.Parameter(torch.normal(0, 1e-1, size=[k, num_classes]))
        self.embs = nn.Parameter(torch.normal(mean=0., std=1e-4, size=[num_domains, k - 1]))
        self.cs_wt = nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[]))

    def forward(self, features, domain_onehot):
        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        logits_common = torch.matmul(features, w_c) + b_c

        c_wts = torch.matmul(domain_onehot, self.embs)
        batch_size = domain_onehot.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), device=features.device) * self.cs_wt, c_wts), 1)
        c_wts = torch.tanh(c_wts)

        w_d = torch.einsum("bk,kdc->bdc", c_wts, self.sms)
        b_d = torch.einsum("bk,kc->bc", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("bdc,bd->bc", w_d, features) + b_d

        return logits_specialized, logits_common


class CSD(nn.Module):
    """
    Common-Specific Decomposition (Piratla et al., 2020)
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.num_classes = num_classes
        self.num_domains = self.hparams.get('num_domains')

        self.featurizer = _build_featurizer(input_dim, self.hparams)
        num_domains = self.num_domains if self.num_domains is not None else 1
        self.csd_head = CSDHead(self.featurizer.output_dim, num_classes, num_domains, k=self.hparams.get('csd_k', 2))

        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.csd_head.parameters()),
            lr=self.hparams.get('lr', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )

    def update(self, minibatches, unlabeled=None, domain_indices=None, **kwargs):
        if domain_indices is None:
            domain_indices = list(range(len(minibatches)))

        num_domains = self.num_domains if self.num_domains is not None else (max(domain_indices) + 1)
        if self.csd_head.num_domains != num_domains:
            self.csd_head = CSDHead(self.featurizer.output_dim, self.num_classes, num_domains, k=self.hparams.get('csd_k', 2)).to(minibatches[0][0].device)
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.csd_head.parameters()),
                lr=self.hparams.get('lr', 1e-3),
                weight_decay=self.hparams.get('weight_decay', 0.0)
            )
        self.csd_head.num_domains = num_domains

        all_features = []
        all_y = []
        all_domain = []

        for (x, y), d_idx in zip(minibatches, domain_indices):
            feats = self.featurizer(x)
            all_features.append(feats)
            all_y.append(y)
            all_domain.append(torch.full((x.size(0),), int(d_idx), device=x.device, dtype=torch.long))

        features = torch.cat(all_features)
        y = torch.cat(all_y)
        domain_ids = torch.cat(all_domain)
        domain_onehot = F.one_hot(domain_ids, num_classes=num_domains).float()

        logits_specialized, logits_common = self.csd_head(features, domain_onehot)

        specific_loss = F.cross_entropy(logits_specialized, y)
        class_loss = F.cross_entropy(logits_common, y)

        sms = self.csd_head.sms
        k = sms.shape[0]
        diag = torch.eye(k, device=sms.device).unsqueeze(0).repeat(self.num_classes, 1, 1)
        cps = torch.stack([torch.matmul(sms[:, :, c], sms[:, :, c].t()) for c in range(self.num_classes)], dim=0)
        orth_loss = torch.mean((cps - diag) ** 2)

        loss = class_loss + specific_loss + self.hparams.get('csd_lambda', 1.0) * orth_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'class_loss': class_loss.item(),
            'specific_loss': specific_loss.item(),
            'orth_loss': orth_loss.item()
        }

    def predict(self, x):
        feats = self.featurizer(x)
        w_c, b_c = self.csd_head.sms[0, :, :], self.csd_head.sm_biases[0, :]
        return torch.matmul(feats, w_c) + b_c


class MASF(nn.Module):
    """
    MASF (Dou et al., 2019). PyTorch port of the official TensorFlow logic.
    """
    def __init__(self, input_dim, num_classes=2, hparams=None):
        super().__init__()
        self.hparams = hparams if hparams else {}
        self.num_classes = num_classes
        self.num_domains = self.hparams.get('num_domains')

        self.featurizer = _build_featurizer(input_dim, self.hparams)
        self.classifier = _build_classifier(self.featurizer.output_dim, num_classes, self.hparams)
        self.network = FeatureClassifier(self.featurizer, self.classifier)

        metric_dim = self.hparams.get('masf_metric_dim', 128)
        self.metric_net = nn.Sequential(
            nn.Linear(self.featurizer.output_dim, metric_dim),
            nn.ReLU(),
            nn.Linear(metric_dim, metric_dim)
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get('lr', 1e-3),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )
        self.metric_optimizer = torch.optim.Adam(
            self.metric_net.parameters(),
            lr=self.hparams.get('masf_metric_lr', self.hparams.get('lr', 1e-3)),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )

    def _kd_loss(self, logits1, y1, logits2, y2, bool_indicator, temperature=2.0):
        kd_loss = 0.0
        eps = 1e-16
        for cls in range(self.num_classes):
            if bool_indicator[cls] < 0.5:
                continue
            mask1 = (y1 == cls)
            mask2 = (y2 == cls)
            if mask1.sum() == 0 or mask2.sum() == 0:
                continue
            act1 = logits1[mask1].mean(0)
            act2 = logits2[mask2].mean(0)
            prob1 = F.softmax(act1 / temperature, dim=0).clamp_min(1e-8)
            prob2 = F.softmax(act2 / temperature, dim=0).clamp_min(1e-8)
            kl_div = 0.5 * (
                torch.sum(prob1 * torch.log(prob1 / (prob2 + eps))) +
                torch.sum(prob2 * torch.log(prob2 / (prob1 + eps)))
            )
            kd_loss += kl_div
        return kd_loss / self.num_classes

    def _triplet_semihard_loss(self, embeddings, labels, margin):
        if embeddings.size(0) < 2:
            # Keep autograd graph connected even when no valid triplets exist.
            return embeddings.sum() * 0.0

        dist = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1)
        loss_vals = []
        for i in range(embeddings.size(0)):
            anchor_label = labels[i]
            pos_mask = labels == anchor_label
            neg_mask = labels != anchor_label

            pos_mask[i] = False
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_dist = dist[i][pos_mask]
            neg_dist = dist[i][neg_mask]
            pos_dist_val = pos_dist.min()

            semihard_neg = neg_dist[neg_dist > pos_dist_val]
            if semihard_neg.numel() > 0:
                neg_dist_val = semihard_neg.min()
            else:
                neg_dist_val = neg_dist.min()

            loss_vals.append(F.relu(pos_dist_val - neg_dist_val + margin))

        if not loss_vals:
            # Keep autograd graph connected even when no valid triplets exist.
            return embeddings.sum() * 0.0

        return torch.stack(loss_vals).mean()

    def update(self, minibatches, unlabeled=None, **kwargs):
        if len(minibatches) < 3:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            loss = F.cross_entropy(self.network(all_x), all_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item()}

        perm = torch.randperm(len(minibatches)).tolist()
        meta_train_idx = perm[:2]
        meta_test_idx = perm[2:3]

        xa, ya = minibatches[meta_train_idx[0]]
        xa1, ya1 = minibatches[meta_train_idx[1]]
        xb, yb = minibatches[meta_test_idx[0]]

        # Step 1: source loss update
        source_loss = 0.5 * (
            F.cross_entropy(self.network(xa), ya) + F.cross_entropy(self.network(xa1), ya1)
        )
        self.optimizer.zero_grad()
        source_loss.backward()
        self.optimizer.step()

        # Inner update
        inner_net = copy.deepcopy(self.network)
        inner_opt = torch.optim.Adam(
            inner_net.parameters(),
            lr=self.hparams.get('masf_inner_lr', self.hparams.get('lr', 1e-3)),
            weight_decay=self.hparams.get('weight_decay', 0.0)
        )
        inner_loss = 0.5 * (
            F.cross_entropy(inner_net(xa), ya) + F.cross_entropy(inner_net(xa1), ya1)
        )
        inner_opt.zero_grad()
        inner_loss.backward()
        inner_opt.step()

        logits_a = inner_net(xa)
        logits_a1 = inner_net(xa1)
        logits_b = inner_net(xb)

        classes_b = torch.unique(yb).tolist()
        classes_a = torch.unique(ya).tolist()
        classes_a1 = torch.unique(ya1).tolist()
        bool_indicator_b_a = torch.zeros(self.num_classes, device=xa.device)
        bool_indicator_b_a1 = torch.zeros(self.num_classes, device=xa.device)
        for cls in range(self.num_classes):
            if (cls in classes_b) and (cls in classes_a):
                bool_indicator_b_a[cls] = 1.0
            if (cls in classes_b) and (cls in classes_a1):
                bool_indicator_b_a1[cls] = 1.0

        kd_temp = self.hparams.get('masf_temperature', 2.0)
        global_loss = 0.5 * (
            self._kd_loss(logits_b, yb, logits_a, ya, bool_indicator_b_a, temperature=kd_temp) +
            self._kd_loss(logits_b, yb, logits_a1, ya1, bool_indicator_b_a1, temperature=kd_temp)
        )

        part = min(xa.size(0), xa1.size(0), xb.size(0))
        input_group = torch.cat([xa[:part], xa1[:part], xb[:part]], dim=0)
        label_group = torch.cat([ya[:part], ya1[:part], yb[:part]], dim=0)

        embeddings = self.metric_net(inner_net.forward_features(input_group))
        metric_loss = self._triplet_semihard_loss(
            embeddings,
            label_group,
            margin=self.hparams.get('masf_margin', 1.0)
        )

        meta_loss = global_loss + self.hparams.get('masf_metric_weight', 0.005) * metric_loss

        # Meta update using gradients from inner_net
        meta_grads = autograd.grad(meta_loss, inner_net.parameters(), allow_unused=True)
        self.optimizer.zero_grad()
        for p, g in zip(self.network.parameters(), meta_grads):
            if g is not None:
                p.grad = g.detach()
        self.optimizer.step()

        # Metric network update (recompute to avoid graph reuse)
        self.metric_optimizer.zero_grad()
        with torch.no_grad():
            group_feats = inner_net.forward_features(input_group)
        metric_embeddings = self.metric_net(group_feats.detach())
        metric_loss_metric = self._triplet_semihard_loss(
            metric_embeddings,
            label_group,
            margin=self.hparams.get('masf_margin', 1.0)
        )
        if metric_loss_metric.requires_grad:
            metric_loss_metric.backward()
            self.metric_optimizer.step()

        return {
            'loss': (source_loss + meta_loss).item(),
            'source_loss': source_loss.item(),
            'global_loss': global_loss.item(),
            'metric_loss': metric_loss.item()
        }

    def predict(self, x):
        return self.network(x)


# --- Training Helper ---

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains - num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i, j in zip(meta_train, cycle(meta_test)):
        xi, yi = minibatches[i][0], minibatches[i][1]
        xj, yj = minibatches[j][0], minibatches[j][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


def train_dg_model(model, X_train, y_train, d_train, X_val, y_val, d_val,
                   epochs=20, batch_size=32, domains_per_batch=8, patience=20, device='cuda'):
    """
    Training loop for Domain Generalization models.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    unique_domains = np.unique(d_train)
    num_domains = len(unique_domains)

    if hasattr(model, 'num_domains'):
        model.num_domains = num_domains
    if hasattr(model, 'hparams'):
        model.hparams['num_domains'] = num_domains

    print(f"DG Training: {num_domains} domains, sampling {domains_per_batch} per batch.")

    domain_loaders = []
    domain_datasets = []
    for domain in unique_domains:
        mask = (d_train == domain)
        X_d = torch.tensor(X_train[mask], dtype=torch.float32)
        y_d = torch.tensor(y_train[mask], dtype=torch.long)
        dataset = TensorDataset(X_d, y_d)
        domain_datasets.append(dataset)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        domain_loaders.append(iter(loader))

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    steps_per_epoch = max(10, int(len(X_train) / (batch_size * domains_per_batch)))

    import copy as _copy
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    best_epoch = None
    early_stopped = False
    early_stop_epoch = None
    epochs_ran = 0
    epoch_history = []

    from tqdm import tqdm
    epoch_iterator = tqdm(range(epochs), desc="DG Training")

    for epoch in epoch_iterator:
        model.train()
        epoch_loss = 0.0

        for _ in range(steps_per_epoch):
            start_domain_idx = np.random.choice(
                num_domains,
                domains_per_batch,
                replace=(num_domains < domains_per_batch)
            )

            minibatches = []
            for d_idx in start_domain_idx:
                loader = domain_loaders[d_idx]
                try:
                    batch = next(loader)
                except StopIteration:
                    domain_loaders[d_idx] = iter(
                        DataLoader(domain_datasets[d_idx], batch_size=batch_size, shuffle=True, drop_last=False)
                    )
                    batch = next(domain_loaders[d_idx])

                x, y = batch
                minibatches.append((x.to(device), y.to(device)))

            metrics = model.update(minibatches, domain_indices=list(start_domain_idx))
            epoch_loss += metrics.get('loss', 0.0)

        model.eval()
        with torch.no_grad():
            logits = model.predict(X_val_t)
            val_loss = F.cross_entropy(logits, y_val_t).item()
            preds = logits.argmax(dim=1)
            val_acc = (preds == y_val_t).float().mean().item()

        epoch_iterator.set_postfix({
            'Loss': f'{epoch_loss/steps_per_epoch:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}'
        })
        epoch_num = epoch + 1
        epochs_ran = epoch_num
        epoch_history.append({
            'epoch': epoch_num,
            'train_loss': round(float(epoch_loss / steps_per_epoch), 6),
            'val_loss': round(float(val_loss), 6),
            'val_accuracy': round(float(val_acc), 6),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = _copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch_num
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_iterator.write(f"Early stopping at epoch {epoch}")
                early_stopped = True
                early_stop_epoch = epoch_num
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    attach_training_metadata(
        model,
        optimizer=getattr(getattr(model, 'optimizer', None), '__class__', type('obj', (), {})).__name__
        if getattr(model, 'optimizer', None) is not None else 'domainbed_optimizer',
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        early_stop_epoch=early_stop_epoch,
        epochs_ran=epochs_ran,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        model_selection_metric='val_loss',
        best_metric_value=round(float(best_val_loss), 6) if best_epoch is not None else None,
        epoch_history=epoch_history,
        domains_per_batch=domains_per_batch,
    )
    return model
