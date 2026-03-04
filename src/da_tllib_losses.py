"""TLL-equivalent loss implementations (ported from Transfer-Learning-Library)."""
from typing import Optional, Sequence, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# ---- GRL ----
class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0) -> torch.Tensor:
        ctx.coeff = coeff
        return input * 1.0

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    def __init__(self, alpha: float = 1.0, lo: float = 0.0, hi: float = 1.0, max_iters: int = 1000, auto_step: bool = False):
        super().__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1


# ---- Entropy ----

def entropy(predictions: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    return H


# ---- Domain Discriminator ----
class DomainDiscriminator(nn.Sequential):
    def __init__(self, in_feature: int, hidden_size: int, batch_norm: bool = True, sigmoid: bool = True):
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)
        if batch_norm:
            super().__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer,
            )
        else:
            super().__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                final_layer,
            )


# ---- CDAN Maps ----
class RandomizedMultiLinearMap(nn.Module):
    def __init__(self, features_dim: int, num_classes: int, output_dim: int = 1024):
        super().__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


# ---- Losses ----
class DomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, reduction: str = 'mean', grl: Optional[nn.Module] = None, sigmoid: bool = True):
        super().__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor, w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1), device=f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1), device=f_t.device)
            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),), device=f_s.device),
                torch.zeros((f_t.size(0),), device=f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),), device=f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),), device=f_t.device)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == 'mean':
                return loss.mean()
            if self.reduction == 'sum':
                return loss.sum()
            if self.reduction == 'none':
                return loss
            raise NotImplementedError(self.reduction)


class ConditionalDomainAdversarialLoss(nn.Module):
    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: bool = False,
                 randomized: bool = False, num_classes: int = -1, features_dim: int = -1, randomized_dim: int = 1024,
                 reduction: str = 'mean', sigmoid: bool = True):
        super().__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True)
        self.entropy_conditioning = entropy_conditioning
        self.sigmoid = sigmoid
        self.reduction = reduction
        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size

        if self.sigmoid:
            d_label = torch.cat((
                torch.ones((g_s.size(0), 1), device=g_s.device),
                torch.zeros((g_t.size(0), 1), device=g_t.device),
            ))
            if self.entropy_conditioning:
                return F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction=self.reduction)
            return F.binary_cross_entropy(d, d_label, reduction=self.reduction)
        else:
            d_label = torch.cat((
                torch.ones((g_s.size(0),), device=g_s.device),
                torch.zeros((g_t.size(0),), device=g_t.device),
            )).long()
            if self.entropy_conditioning:
                raise NotImplementedError('entropy_conditioning with softmax discriminator')
            return F.cross_entropy(d, d_label, reduction=self.reduction)


class GaussianKernel(nn.Module):
    def __init__(self, sigma: Optional[float] = None, track_running_stats: bool = True, alpha: float = 1.0):
        super().__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)
        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())
        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, kernels: Sequence[nn.Module], linear: bool = False):
        super().__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        loss = (kernel_matrix * self.index_matrix).sum() + 2.0 / float(batch_size - 1)
        return loss


def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None, linear: bool = True) -> torch.Tensor:
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1.0 / float(batch_size)
                index_matrix[t1, t2] = 1.0 / float(batch_size)
                index_matrix[s1, t2] = -1.0 / float(batch_size)
                index_matrix[s2, t1] = -1.0 / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1.0 / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1.0 / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1.0 / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1.0 / float(batch_size * batch_size)
    return index_matrix


class JointMultipleKernelMaximumMeanDiscrepancy(nn.Module):
    def __init__(self, kernels: Sequence[Sequence[nn.Module]], linear: bool = True, thetas: Sequence[nn.Module] = None):
        super().__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear
        if thetas:
            self.thetas = thetas
        else:
            self.thetas = [nn.Identity() for _ in kernels]

    def forward(self, z_s: Tuple[torch.Tensor, ...], z_t: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        batch_size = int(z_s[0].size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s[0].device)
        kernel_matrix = torch.ones_like(self.index_matrix)
        for layer_z_s, layer_z_t, layer_kernels, theta in zip(z_s, z_t, self.kernels, self.thetas):
            layer_features = torch.cat([layer_z_s, layer_z_t], dim=0)
            layer_features = theta(layer_features)
            kernel_matrix *= sum([kernel(layer_features) for kernel in layer_kernels])
        loss = (kernel_matrix * self.index_matrix).sum() + 2.0 / float(batch_size - 1)
        return loss


class CorrelationAlignmentLoss(nn.Module):
    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        mean_s = f_s.mean(0, keepdim=True)
        mean_t = f_t.mean(0, keepdim=True)
        cent_s = f_s - mean_s
        cent_t = f_t - mean_t
        cov_s = torch.mm(cent_s.t(), cent_s) / (len(f_s) - 1)
        cov_t = torch.mm(cent_t.t(), cent_t) / (len(f_t) - 1)
        mean_diff = (mean_s - mean_t).pow(2).mean()
        cov_diff = (cov_s - cov_t).pow(2).mean()
        return mean_diff + cov_diff


class MinimumClassConfusionLoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions)
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


# ---- MCD helpers ----

def classifier_discrepancy(predictions1: torch.Tensor, predictions2: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(predictions1 - predictions2))


def mcd_entropy(predictions: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.log(torch.mean(predictions, 0) + 1e-6))
