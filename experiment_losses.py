"""Metric-learning objectives used by the planned controlled comparisons.

CATML follows the mathematical definition in the companion dissertation,
section 1.6. The aggregate reduction is explicit because the dissertation
deliberately leaves published sample/class weighting to be confirmed from the
historical implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CATMLLoss(nn.Module):
    """Cluster-Aware Triplet-Based Metric Loss.

    ``h`` is the network output. Triplet distances use ``softplus(h)`` while
    the class-radius term uses the original ``h`` space. With
    ``class_weighting='per_sample'``, class weights are m_c / B, giving every
    batch item equal weight in the center penalty. ``equal_class`` instead
    gives each represented class equal weight.
    """

    def __init__(
        self,
        margin: float = 10.0,
        rho: float = 0.1,
        tau: float = 1.0,
        xi: float = 1.0,
        center_mode: str = "current",
        class_weighting: str = "per_sample",
        norm_epsilon: float = 0.0,
    ):
        super().__init__()
        if center_mode not in ("current", "fixed"):
            raise ValueError("center_mode must be 'current' or 'fixed'")
        if class_weighting not in ("per_sample", "equal_class"):
            raise ValueError("class_weighting must be 'per_sample' or 'equal_class'")
        if norm_epsilon < 0:
            raise ValueError("norm_epsilon must be non-negative")
        self.margin = float(margin)
        self.rho = float(rho)
        self.tau = float(tau)
        self.xi = float(xi)
        self.center_mode = center_mode
        self.class_weighting = class_weighting
        self.norm_epsilon = float(norm_epsilon)

    def _norm(self, values: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(values, ord=2, dim=-1)
        if self.norm_epsilon:
            # This is an explicitly smoothed variant of the Euclidean norm.
            norm = torch.sqrt(norm.square() + self.norm_epsilon**2) - self.norm_epsilon
        return norm

    def forward(
        self,
        h: torch.Tensor,
        labels: torch.Tensor,
        anchor_idx: torch.Tensor,
        positive_idx: torch.Tensor,
        negative_idx: torch.Tensor,
        fixed_centers: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = h.float()
        labels = labels.to(device=h.device, dtype=torch.long)
        anchor_idx = anchor_idx.to(device=h.device, dtype=torch.long)
        positive_idx = positive_idx.to(device=h.device, dtype=torch.long)
        negative_idx = negative_idx.to(device=h.device, dtype=torch.long)
        if anchor_idx.numel() == 0:
            raise ValueError("CATML requires at least one valid triplet")

        z = F.softplus(h)
        d_ap = self._norm(z[anchor_idx] - z[positive_idx])
        d_an = self._norm(z[anchor_idx] - z[negative_idx])
        g1 = d_ap
        g2 = F.softplus(d_ap - d_an + self.margin)
        triplet = (self.rho * g1 + self.tau * g2).mean()

        represented, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        counts = torch.bincount(inverse, minlength=represented.numel()).to(dtype=h.dtype)
        center_sums = h.new_zeros((represented.numel(), h.shape[1]))
        center_sums.index_add_(0, inverse, h)
        centers = center_sums / counts.clamp_min(1).unsqueeze(1)
        if self.center_mode == "fixed":
            if fixed_centers is not None:
                # The fixed-center path is a deliberate experimental variant;
                # the current-center path below remains fully vectorized.
                centers = torch.stack([
                    fixed_centers.get(int(class_id), centers[index]).to(device=h.device, dtype=h.dtype)
                    for index, class_id in enumerate(represented.detach().cpu().tolist())
                ]).detach()
            else:
                centers = centers.detach()
        per_sample_radius = self._norm(h - centers[inverse])
        if self.class_weighting == "per_sample":
            # With w_c=m_c/B, sum_c w_c G_c equals this sample mean.
            class_term = per_sample_radius.mean()
        else:
            class_radius_sum = h.new_zeros(represented.numel())
            class_radius_sum.index_add_(0, inverse, per_sample_radius)
            class_term = (class_radius_sum / counts.clamp_min(1)).mean()
        total = triplet + self.xi * class_term
        return total, {
            "loss": total.detach(),
            "triplet": triplet.detach(),
            "cluster": class_term.detach(),
            "positive_distance": d_ap.detach().mean(),
            "negative_distance": d_an.detach().mean(),
        }


class ProxyAnchorLoss(nn.Module):
    """Proxy Anchor objective with one learnable proxy per training class."""

    def __init__(self, num_classes: int, embedding_dim: int, margin: float = 0.1, alpha: float = 32.0):
        super().__init__()
        self.proxies = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.proxies)
        self.margin = float(margin)
        self.alpha = float(alpha)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = F.normalize(embeddings.float(), dim=1)
        p = F.normalize(self.proxies.float(), dim=1)
        similarity = x @ p.T
        labels = labels.long()
        positive_mask = F.one_hot(labels, num_classes=p.shape[0]).bool()
        positive = torch.log1p(torch.exp(-self.alpha * (similarity - self.margin)).masked_fill(~positive_mask, -torch.inf).sum(dim=0))
        negative = torch.log1p(torch.exp(self.alpha * (similarity + self.margin)).masked_fill(positive_mask, -torch.inf).sum(dim=0))
        active_positive = positive_mask.any(dim=0)
        active_negative = (~positive_mask).any(dim=0)
        positive_loss = positive[active_positive].sum() / max(1, int(active_positive.sum()))
        negative_loss = negative[active_negative].sum() / max(1, int(active_negative.sum()))
        return positive_loss + negative_loss


def triplet_hinge(embeddings, anchor_idx, positive_idx, negative_idx, margin=0.2):
    x = F.normalize(embeddings.float(), dim=1)
    d_ap = torch.linalg.vector_norm(x[anchor_idx] - x[positive_idx], dim=1)
    d_an = torch.linalg.vector_norm(x[anchor_idx] - x[negative_idx], dim=1)
    return F.relu(d_ap - d_an + margin).mean()


def pair_contrastive(embeddings, anchor_idx, positive_idx, negative_idx, margin=1.0):
    x = embeddings.float()
    d_ap = torch.linalg.vector_norm(x[anchor_idx] - x[positive_idx], dim=1)
    d_an = torch.linalg.vector_norm(x[anchor_idx] - x[negative_idx], dim=1)
    return d_ap.square().mean() + F.relu(margin - d_an).square().mean()


def supervised_contrastive(embeddings, labels, temperature=0.07):
    x = F.normalize(embeddings.float(), dim=1)
    labels = labels.reshape(-1, 1)
    logits = (x @ x.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(x), device=x.device, dtype=torch.bool)
    positive = (labels == labels.T) & ~identity
    exp_logits = torch.exp(logits).masked_fill(identity, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    return -((log_prob * positive).sum(dim=1)[valid] / positive_count[valid]).mean()


def multiple_negatives_ranking(embeddings, labels, temperature=0.05):
    """In-batch ranking with same-label examples masked as false negatives."""
    x = F.normalize(embeddings.float(), dim=1)
    labels = labels.reshape(-1)
    similarity = (x @ x.T) / temperature
    identity = torch.eye(len(x), device=x.device, dtype=torch.bool)
    same_class = labels[:, None] == labels[None, :]
    valid_positive = same_class & ~identity
    row_valid = valid_positive.any(dim=1)
    if not row_valid.any():
        return embeddings.sum() * 0.0
    sampling_mask = valid_positive.clone()
    if not row_valid.all():
        sampling_mask[~row_valid, 0] = True
    positive_indices = torch.multinomial(sampling_mask.float(), num_samples=1).squeeze(1)
    # Each anchor has one target positive. Other same-class samples are masked
    # instead of being treated as false negatives.
    valid = ~same_class & ~identity
    valid[torch.arange(len(x), device=x.device), positive_indices] = True
    logits = similarity.masked_fill(~valid, -torch.inf)
    rows = torch.where(row_valid)[0]
    return F.cross_entropy(logits[rows], positive_indices[rows])


def multi_similarity(embeddings, labels, alpha=2.0, beta=50.0, base=0.5, epsilon=0.1):
    x = F.normalize(embeddings.float(), dim=1)
    similarity = x @ x.T
    labels = labels.reshape(-1)
    identity = torch.eye(len(x), device=x.device, dtype=torch.bool)
    same = labels[:, None] == labels[None, :]
    positive = same & ~identity
    negative = ~same
    has_positive = positive.any(dim=1)
    has_negative = negative.any(dim=1)
    if not (has_positive & has_negative).any():
        return embeddings.sum() * 0.0
    max_negative = similarity.masked_fill(~negative, -torch.inf).max(dim=1).values
    min_positive = similarity.masked_fill(~positive, torch.inf).min(dim=1).values
    selected_positive = positive & (similarity < (max_negative[:, None] + epsilon))
    selected_negative = negative & (similarity > (min_positive[:, None] - epsilon))
    positive_logits = (-alpha * (similarity - base)).masked_fill(~selected_positive, -torch.inf)
    negative_logits = (beta * (similarity - base)).masked_fill(~selected_negative, -torch.inf)
    positive_loss = F.softplus(torch.logsumexp(positive_logits, dim=1)) / alpha
    negative_loss = F.softplus(torch.logsumexp(negative_logits, dim=1)) / beta
    valid = has_positive & has_negative
    return (positive_loss[valid] + negative_loss[valid]).mean()

