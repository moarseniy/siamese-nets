import json

import torch


def generate_sym_probs(dists, ideals, counter, probs_vec, w, gamma):
    """Return a normalized class-sampling distribution from current class distances."""
    class_count = ideals.size(0)
    if class_count == 0:
        return torch.empty(0, dtype=torch.float32)

    weights = torch.as_tensor(dists, dtype=torch.float32).detach().cpu().clamp_min(0)
    if weights.numel() != class_count or not torch.isfinite(weights).all() or weights.sum() <= 0:
        return torch.full((class_count,), 1.0 / class_count)

    weights = weights.pow(gamma)
    weights = weights / weights.sum()
    previous = torch.as_tensor(probs_vec, dtype=torch.float32).detach().cpu()
    if previous.numel() != class_count or not torch.isfinite(previous).all() or previous.sum() <= 0:
        previous = torch.full((class_count,), 1.0 / class_count)
    else:
        previous = previous / previous.sum()

    probabilities = ((1 - w) * weights + w * previous).clamp_min(0)
    probabilities = probabilities.pow(gamma)
    total = probabilities.sum()
    return probabilities / total if total > 0 else torch.full((class_count,), 1.0 / class_count)


def save_sym_probs(out_path, probs_vec, alphabet=None):
    probs = torch.as_tensor(probs_vec).detach().cpu().flatten()
    symbols = alphabet if alphabet is not None and len(alphabet) == len(probs) else list(range(len(probs)))
    sym_probs = {str(symbols[i]): float(probs[i]) for i in range(len(probs))}
    with open(out_path, 'w', encoding='utf8') as json_out:
        json.dump(sym_probs, json_out, indent=2, ensure_ascii=False)


def generate_clusters(ideals, raw_clusters, classes_count, active_ids=None):
    """Return closest inter-class centroid pairs as [distance, class_a, class_b]."""
    classes_count = min(int(classes_count), int(ideals.size(0)))
    if active_ids is None:
        class_ids = torch.arange(classes_count, device=ideals.device)
    else:
        class_ids = torch.as_tensor(active_ids, device=ideals.device, dtype=torch.long)[:classes_count]
        classes_count = min(classes_count, class_ids.numel())
        class_ids = class_ids[:classes_count]
        ideals = ideals[class_ids]
    if classes_count < 2 or raw_clusters <= 0:
        return ideals.new_empty((0, 3))

    pair_count = classes_count * (classes_count - 1) // 2
    keep_count = min(int(raw_clusters), pair_count)
    best = ideals.new_empty((0, 3))

    # Generate one diagonal of the pairwise distance matrix at a time to avoid
    # allocating an O(classes_count^2) matrix.
    for offset in range(1, classes_count):
        left = ideals[:classes_count - offset]
        right = ideals[offset:classes_count]
        distances = torch.linalg.vector_norm(left - right, dim=1)
        left_ids = class_ids[:classes_count - offset]
        right_ids = class_ids[offset:classes_count]
        candidates = torch.stack((distances, left_ids.to(distances.dtype),
                                  right_ids.to(distances.dtype)), dim=1)
        best = torch.cat((best, candidates), dim=0)
        if best.size(0) > keep_count:
            best = best[torch.topk(best[:, 0], k=keep_count, largest=False).indices]

    if best.size(0) > keep_count:
        best = best[torch.topk(best[:, 0], k=keep_count, largest=False).indices]
    if best.size(0):
        best = best[torch.argsort(best[:, 0])]
    return best


def merge_clusters(norms_res, merged_clusters=None, cluster_max_size=None):
    """Build bounded components from nearest-class edges, returning class ids."""
    clusters = [] if merged_clusters is None else merged_clusters
    clusters.clear()
    if norms_res.numel() == 0:
        return clusters

    parent = {}
    sizes = {}

    def find(value):
        parent.setdefault(value, value)
        sizes.setdefault(value, 1)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    max_size = int(cluster_max_size) if cluster_max_size is not None else 0
    for row in norms_res.detach().cpu().tolist():
        class_a, class_b = int(row[1]), int(row[2])
        root_a, root_b = find(class_a), find(class_b)
        if root_a == root_b:
            continue
        merged_size = sizes[root_a] + sizes[root_b]
        if max_size > 0 and merged_size > max_size:
            continue
        parent[root_b] = root_a
        sizes[root_a] = merged_size

    components = {}
    for class_id in parent:
        components.setdefault(find(class_id), []).append(class_id)
    clusters.extend(sorted(component) for component in components.values() if len(component) > 1)
    return clusters


def save_clusters(out_path, clusters, alphabet=None):
    visible_clusters = []
    for cluster in clusters:
        visible_cluster = []
        for class_id in cluster:
            if alphabet is not None and class_id < len(alphabet):
                visible_cluster.append(alphabet[class_id])
            else:
                visible_cluster.append(int(class_id))
        visible_clusters.append(visible_cluster)

    with open(out_path, 'w', encoding='utf8') as json_out:
        json.dump(visible_clusters, json_out, indent=2, ensure_ascii=False)


# Shared batch sampler and online class-selection state for the experiment
# protocols; legacy dataset sampler functions above are kept unchanged.
from experiment_sampling import (  # noqa: E402
    APMState,
    ClassGraph,
    PKBatchSampler,
    make_triplets,
)
