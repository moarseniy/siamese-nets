"""Class-balanced batches, APM, class clustering, and in-batch mining."""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np
import torch


class APMState:
    """Online radius-based class probabilities.

    Radii are updated only from training embeddings. With ``smoothing=0`` and
    ``gamma=1`` this follows the published APM probability rule; nonzero
    smoothing is an explicit stabilization variant.
    """

    def __init__(self, labels: list[str], gamma: float = 1.0, smoothing: float = 0.0, ema: float = 0.9, unobserved_floor: float = 0.01):
        self.labels = labels
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        self.ema = float(ema)
        self.unobserved_floor = float(unobserved_floor)
        self.radii = np.zeros(len(labels), dtype=np.float64)
        self.observed = np.zeros(len(labels), dtype=bool)
        self.previous = np.full(len(labels), 1 / max(1, len(labels)), dtype=np.float64)

    def update(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = embeddings.detach().float()
        labels = labels.detach().long()
        sums = embeddings.new_zeros((len(self.labels), embeddings.shape[1]))
        counts = embeddings.new_zeros(len(self.labels))
        sums.index_add_(0, labels, embeddings)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=embeddings.dtype))
        centers = sums / counts.clamp_min(1).unsqueeze(1)
        radii_sum = embeddings.new_zeros(len(self.labels))
        residuals = torch.linalg.vector_norm(embeddings - centers[labels], dim=1)
        radii_sum.index_add_(0, labels, residuals)
        batch_radii = (radii_sum / counts.clamp_min(1)).cpu().numpy()
        active = (counts > 0).cpu().numpy()
        first_observation = active & ~self.observed
        subsequent = active & self.observed
        self.radii[first_observation] = batch_radii[first_observation]
        self.radii[subsequent] = self.ema * self.radii[subsequent] + (1 - self.ema) * batch_radii[subsequent]
        self.observed[active] = True

    def probabilities(self) -> np.ndarray:
        weights = np.where(self.observed, np.maximum(self.radii, 0.0), 0.0) ** self.gamma
        if self.observed.any() and (~self.observed).any():
            observed_mean = float(weights[self.observed].mean())
            weights[~self.observed] = observed_mean * self.unobserved_floor
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            return np.full(len(self.labels), 1 / len(self.labels), dtype=np.float64)
        weights /= weights.sum()
        probs = (1 - self.smoothing) * weights + self.smoothing * self.previous
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.full(len(self.labels), 1 / len(self.labels), dtype=np.float64)
        else:
            probs /= probs.sum()
        self.previous = probs.copy()
        return probs

    def state_dict(self) -> dict:
        return {"radii": self.radii.tolist(), "observed": self.observed.tolist(), "previous": self.previous.tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.radii = np.asarray(state["radii"], dtype=np.float64)
        self.observed = np.asarray(state["observed"], dtype=bool)
        self.previous = np.asarray(state["previous"], dtype=np.float64)


class ClassGraph:
    """Bounded connected components over nearest class-centroid pairs."""

    def __init__(self, class_count: int, max_cluster_size: int = 200):
        self.class_count = class_count
        self.max_cluster_size = int(max_cluster_size)
        self.neighbors: dict[int, set[int]] = defaultdict(set)
        self.clusters: list[list[int]] = []

    @torch.no_grad()
    def update(self, centers: torch.Tensor, active_ids: torch.Tensor, fraction: float = 0.002, max_edges: int = 10000) -> None:
        ids = active_ids.detach().long().unique(sorted=True)
        if ids.numel() < 2:
            self.neighbors.clear()
            self.clusters.clear()
            return
        points = centers[ids].detach().float()
        pair_count = ids.numel() * (ids.numel() - 1) // 2
        keep = min(max(1, int(pair_count * fraction)), int(max_edges), pair_count)
        candidate_d = points.new_empty((0,))
        candidate_a = torch.empty(0, dtype=torch.long, device=points.device)
        candidate_b = torch.empty(0, dtype=torch.long, device=points.device)
        # Chunking bounds peak distance-matrix memory for the 11k-class Hangul
        # alphabet; it also avoids materializing every class pair at once.
        chunk_size = max(64, min(512, int(2_000_000 / max(1, points.shape[0]))))
        for start in range(0, points.shape[0], chunk_size):
            stop = min(points.shape[0], start + chunk_size)
            distances = torch.cdist(points[start:stop], points)
            row_ids = torch.arange(start, stop, device=points.device)
            col_ids = torch.arange(points.shape[0], device=points.device)
            distances[row_ids[:, None] >= col_ids[None, :]] = float("inf")
            values, flat = torch.topk(distances.flatten(), k=min(keep, distances.numel()), largest=False)
            rows = flat // points.shape[0] + start
            cols = flat % points.shape[0]
            finite = torch.isfinite(values)
            values, rows, cols = values[finite], rows[finite], cols[finite]
            candidate_d = torch.cat((candidate_d, values))
            candidate_a = torch.cat((candidate_a, ids[rows]))
            candidate_b = torch.cat((candidate_b, ids[cols]))
            if candidate_d.numel() > keep:
                best = torch.topk(candidate_d, k=keep, largest=False).indices
                candidate_d, candidate_a, candidate_b = candidate_d[best], candidate_a[best], candidate_b[best]

        order = torch.argsort(candidate_d)
        edges = list(zip(candidate_a[order].cpu().tolist(), candidate_b[order].cpu().tolist()))
        parent = {int(class_id): int(class_id) for class_id in ids.cpu().tolist()}
        size = {int(class_id): 1 for class_id in ids.cpu().tolist()}

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for left, right in edges:
            root_left, root_right = find(left), find(right)
            if root_left == root_right:
                continue
            combined = size[root_left] + size[root_right]
            if self.max_cluster_size > 0 and combined > self.max_cluster_size:
                continue
            parent[root_right] = root_left
            size[root_left] = combined

        components: dict[int, list[int]] = defaultdict(list)
        for class_id in parent:
            components[find(class_id)].append(class_id)
        self.clusters = [sorted(group) for group in components.values() if len(group) > 1]
        self.neighbors = defaultdict(set)
        for cluster in self.clusters:
            for left in cluster:
                self.neighbors[left].update(right for right in cluster if right != left)

    def candidates(self, class_id: int, available: set[int]) -> list[int]:
        return sorted(self.neighbors.get(int(class_id), set()) & available)

    def state_dict(self) -> dict:
        return {"clusters": self.clusters}

    def load_state_dict(self, state: dict) -> None:
        self.clusters = [list(map(int, cluster)) for cluster in state.get("clusters", [])]
        self.neighbors = defaultdict(set)
        for cluster in self.clusters:
            for left in cluster:
                self.neighbors[left].update(right for right in cluster if right != left)


class PKBatchSampler:
    """Samples P classes and K distinct examples per class for metric learning."""

    def __init__(
        self,
        catalog,
        labels: list[str],
        split: str,
        protocol: str,
        classes_per_batch: int,
        samples_per_class: int,
        steps_per_epoch: int,
        seed: int,
        data_split_seed: int | None = None,
        apm: APMState | None = None,
        examples_per_class: int | None = None,
    ):
        self.catalog = catalog
        self.labels = labels
        self.split = split
        self.protocol = protocol
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.data_split_seed = int(seed if data_split_seed is None else data_split_seed)
        self.epoch = 0
        self.apm = apm
        self.examples_per_class = int(examples_per_class) if examples_per_class not in (None, "all") else None
        if self.classes_per_batch < 2:
            raise ValueError("Metric learning needs at least two classes per batch")
        if self.classes_per_batch > len(labels):
            raise ValueError(f"classes_per_batch={classes_per_batch} exceeds the {len(labels)} train classes")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        probs = self.apm.probabilities() if self.apm is not None else None
        for _ in range(self.steps_per_epoch):
            if probs is None:
                selected = rng.sample(range(len(self.labels)), self.classes_per_batch)
            else:
                selected = np.random.default_rng(rng.randrange(2**32)).choice(
                    len(self.labels), size=self.classes_per_batch, replace=False, p=probs
                ).tolist()
            batch = []
            for class_index in selected:
                label = self.labels[int(class_index)]
                batch.extend(
                    self.catalog.sample_class(
                        label, self.split, self.data_split_seed, self.protocol, self.samples_per_class, rng,
                        self.examples_per_class,
                    )
                )
            yield batch


def make_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: str = "random",
    margin: float = 0.2,
    class_graph: ClassGraph | None = None,
    directed_probability: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one positive/negative pair per anchor from a P x K batch.

    Mining is vectorized over the batch so large Blackwell batches do not incur
    one host/device synchronization per anchor.
    """
    labels = labels.long()
    device = embeddings.device
    count = len(labels)
    if count < 2:
        raise ValueError("Metric mining requires at least two examples")
    row_ids = torch.arange(count, device=device)
    same_class = labels[:, None] == labels[None, :]
    positive_mask = same_class & (row_ids[:, None] != row_ids[None, :])
    negative_mask = ~same_class
    if not positive_mask.any(dim=1).all() or not negative_mask.any(dim=1).all():
        raise ValueError("Each sampled class needs two examples and each batch needs two classes")
    positive_idx = torch.multinomial(positive_mask.float(), num_samples=1).squeeze(1)

    if mode in ("batch_hard", "hard"):
        distances = torch.cdist(embeddings.detach().float(), embeddings.detach().float())
        negative_idx = distances.masked_fill(~negative_mask, torch.inf).argmin(dim=1)
    elif mode == "semi_hard":
        distances = torch.cdist(embeddings.detach().float(), embeddings.detach().float())
        positive_distance = distances[row_ids, positive_idx]
        semi_mask = negative_mask & (distances > positive_distance[:, None]) & (distances < (positive_distance + margin)[:, None])
        hard_idx = distances.masked_fill(~negative_mask, torch.inf).argmin(dim=1)
        if semi_mask.any(dim=1).any():
            rows = torch.where(semi_mask.any(dim=1))[0]
            semi_idx = distances[rows].masked_fill(~semi_mask[rows], torch.inf).argmin(dim=1)
            negative_idx = hard_idx
            negative_idx[rows] = semi_idx
        else:
            negative_idx = hard_idx
    else:
        negative_idx = torch.multinomial(negative_mask.float(), num_samples=1).squeeze(1)
        if mode == "auto_clusters" and class_graph is not None and directed_probability > 0:
            unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
            unique_values = unique_labels.detach().cpu().tolist()
            index_by_label = {int(value): index for index, value in enumerate(unique_values)}
            pair_rows, pair_cols = [], []
            for class_id in unique_values:
                left = index_by_label[int(class_id)]
                for neighbor in class_graph.neighbors.get(int(class_id), ()):
                    right = index_by_label.get(int(neighbor))
                    if right is not None and right != left:
                        pair_rows.append(left)
                        pair_cols.append(right)
            if pair_rows:
                neighbor_mask = torch.zeros((len(unique_values), len(unique_values)), device=device, dtype=torch.bool)
                neighbor_mask[torch.tensor(pair_rows, device=device), torch.tensor(pair_cols, device=device)] = True
                allowed = negative_mask & neighbor_mask[inverse][:, inverse]
                has_neighbor = allowed.any(dim=1)
                rows = torch.where(has_neighbor)[0]
                if rows.numel():
                    clustered_idx = torch.multinomial(allowed[rows].float(), num_samples=1).squeeze(1)
                    choose_cluster = torch.rand(count, device=device) < directed_probability
                    use = rows[choose_cluster[rows]]
                    negative_idx[use] = clustered_idx[choose_cluster[rows]]

    return row_ids, positive_idx, negative_idx

