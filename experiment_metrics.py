"""Task metrics kept separate from the legacy pair-verification EER."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def _metric_vectors(values: torch.Tensor, embedding_transform: str, distance: str) -> torch.Tensor:
    if embedding_transform == "softplus":
        values = F.softplus(values.float())
    elif embedding_transform == "identity":
        values = values.float()
    else:
        raise ValueError("embedding_transform must be 'identity' or 'softplus'")
    if distance == "cosine":
        return F.normalize(values, dim=1)
    if distance == "euclidean":
        return values
    raise ValueError("distance must be 'cosine' or 'euclidean'")


def _similarity_scores(queries: torch.Tensor, references: torch.Tensor, distance: str) -> torch.Tensor:
    if distance == "cosine":
        return queries @ references.T
    query_norm = queries.square().sum(dim=1, keepdim=True)
    reference_norm = references.square().sum(dim=1).unsqueeze(0)
    return -(query_norm + reference_norm - 2.0 * (queries @ references.T)).clamp_min_(0.0)


def _classification_summary(true_count: torch.Tensor, predicted_count: torch.Tensor, true_positive: torch.Tensor) -> tuple[float, float]:
    total = true_count.sum()
    if total.item() == 0:
        raise ValueError("Embedding classification produced no queries")
    denominator = true_count + predicted_count
    included = denominator > 0
    class_f1 = 2.0 * true_positive[included].float() / denominator[included].float()
    accuracy = float((true_positive.sum().float() / total).item())
    return accuracy, float(class_f1.mean().item())


@torch.no_grad()
def embedding_classification_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    *,
    num_classes: int,
    distance: str = "cosine",
    embedding_transform: str = "identity",
    query_chunk_size: int = 128,
) -> dict[str, float]:
    """Closed-set accuracy using a fixed gallery: nearest exemplar and centroid."""
    q = _metric_vectors(query_embeddings, embedding_transform, distance)
    g = _metric_vectors(gallery_embeddings, embedding_transform, distance).to(q.device)
    q_labels = query_labels.to(q.device).long()
    g_labels = gallery_labels.to(q.device).long()
    if not len(q) or not len(g):
        raise ValueError("Embedding classification needs nonempty query and gallery sets")
    if (g_labels.min() < 0 or g_labels.max() >= num_classes or
            q_labels.min() < 0 or q_labels.max() >= num_classes):
        raise ValueError("Embedding classification labels are outside the configured class range")

    counts = torch.bincount(g_labels, minlength=num_classes)
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].cpu().tolist()
        raise ValueError(f"Gallery is missing training classes; first missing IDs: {missing[:5]}")
    center_sums = g.new_zeros((num_classes, g.shape[1]))
    center_sums.index_add_(0, g_labels, g)
    centers = center_sums / counts.to(g.dtype).unsqueeze(1)
    if distance == "cosine":
        centers = F.normalize(centers, dim=1)

    true_count = torch.bincount(q_labels, minlength=num_classes)
    exemplar_predicted = torch.zeros(num_classes, device=q.device, dtype=torch.long)
    exemplar_true_positive = torch.zeros_like(exemplar_predicted)
    center_predicted = torch.zeros_like(exemplar_predicted)
    center_true_positive = torch.zeros_like(exemplar_predicted)
    for start in range(0, len(q), query_chunk_size):
        stop = min(len(q), start + query_chunk_size)
        query_chunk = q[start:stop]
        exemplar_scores = _similarity_scores(query_chunk, g, distance)
        exemplar_predictions = g_labels[exemplar_scores.argmax(dim=1)]
        center_scores = _similarity_scores(query_chunk, centers, distance)
        center_predictions = center_scores.argmax(dim=1)
        truth = q_labels[start:stop]
        exemplar_predicted += torch.bincount(exemplar_predictions, minlength=num_classes)
        exemplar_true_positive += torch.bincount(exemplar_predictions[exemplar_predictions == truth], minlength=num_classes)
        center_predicted += torch.bincount(center_predictions, minlength=num_classes)
        center_true_positive += torch.bincount(center_predictions[center_predictions == truth], minlength=num_classes)

    exemplar_accuracy, exemplar_f1 = _classification_summary(true_count, exemplar_predicted, exemplar_true_positive)
    center_accuracy, center_f1 = _classification_summary(true_count, center_predicted, center_true_positive)
    return {
        "nearest_exemplar_accuracy": exemplar_accuracy,
        "nearest_exemplar_macro_f1": exemplar_f1,
        "nearest_centroid_accuracy": center_accuracy,
        "nearest_centroid_macro_f1": center_f1,
        "queries": float(len(q)),
        "gallery": float(len(g)),
    }


@torch.no_grad()
def retrieval_metrics(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: torch.Tensor,
    *,
    query_ids: list[str] | None = None,
    gallery_ids: list[str] | None = None,
    ks: tuple[int, ...] = (1, 2, 4, 8),
    query_chunk_size: int = 128,
    distance: str = "cosine",
    embedding_transform: str = "identity",
) -> dict[str, float]:
    """Compute Recall@K and MAP@R without materializing the full Q x G matrix."""
    q = _metric_vectors(query_embeddings, embedding_transform, distance)
    g = _metric_vectors(gallery_embeddings, embedding_transform, distance).to(q.device)
    q_labels = query_labels.to(q.device).long()
    g_labels = gallery_labels.to(q.device).long()
    if not len(q) or not len(g):
        raise ValueError("Retrieval needs nonempty query and gallery sets")
    if len(q_labels) != len(q) or len(g_labels) != len(g):
        raise ValueError("Embedding and label counts do not match")
    if (query_ids is None) != (gallery_ids is None):
        raise ValueError("Provide both query_ids and gallery_ids, or neither")

    same_ids = None
    if query_ids is not None:
        if len(query_ids) != len(q) or len(gallery_ids) != len(g):
            raise ValueError("ID counts do not match embeddings")
        # Use a compact CPU map so the same-image exclusion does not allocate
        # a large Python object matrix.
        gallery_positions: dict[str, list[int]] = {}
        for index, uid in enumerate(gallery_ids):
            gallery_positions.setdefault(uid, []).append(index)
        same_ids = gallery_positions

    hits = {k: 0 for k in ks}
    ap_at_r = []
    valid_queries = 0
    for start in range(0, len(q), query_chunk_size):
        stop = min(len(q), start + query_chunk_size)
        scores = _similarity_scores(q[start:stop], g, distance)
        if same_ids is not None:
            for local_index, uid in enumerate(query_ids[start:stop]):
                for gallery_index in same_ids.get(uid, ()):
                    scores[local_index, gallery_index] = -torch.inf
        relevant_total = (q_labels[start:stop, None] == g_labels[None, :]).sum(dim=1)
        if same_ids is not None:
            for local_index, uid in enumerate(query_ids[start:stop]):
                excluded = same_ids.get(uid, ())
                if excluded:
                    excluded_tensor = torch.as_tensor(excluded, device=q.device, dtype=torch.long)
                    relevant_total[local_index] -= (g_labels[excluded_tensor] == q_labels[start + local_index]).sum()
        max_r = int(relevant_total.max().item()) if relevant_total.numel() else 0
        top_count = min(len(g), max(max(ks), max_r))
        top_scores, top_indices = torch.topk(scores, k=top_count, dim=1, largest=True, sorted=True)
        del top_scores
        for local_index in range(stop - start):
            r = int(relevant_total[local_index].item())
            if r <= 0:
                continue
            valid_queries += 1
            retrieved_labels = g_labels[top_indices[local_index]]
            correct = retrieved_labels == q_labels[start + local_index]
            for k in ks:
                hits[k] += int(correct[:min(k, len(correct))].any().item())
            window = correct[:min(r, len(correct))].float()
            if window.any():
                ranks = torch.arange(1, len(window) + 1, device=q.device, dtype=torch.float32)
                precision_at_rank = window.cumsum(0) / ranks
                ap_at_r.append(float((precision_at_rank * window).sum().item() / r))
            else:
                ap_at_r.append(0.0)
    if valid_queries == 0:
        raise ValueError("No query has a relevant gallery item")
    output = {f"recall@{k}": hits[k] / valid_queries for k in ks}
    output["map@r"] = float(np.mean(ap_at_r))
    output["queries"] = float(valid_queries)
    return output


def equal_error_rate(labels, similarity_scores) -> float:
    fpr, tpr, _ = roc_curve(labels, similarity_scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def open_set_metrics(known_mask, confidence_scores, threshold: float | None = None) -> dict[str, float]:
    """Known/unknown metrics; higher confidence means more likely known.

    When ``threshold`` is omitted, choose the validation operating point that
    maximizes balanced accuracy. Pass that returned threshold unchanged to
    the final test evaluation.
    """
    known = np.asarray(known_mask, dtype=np.int32)
    scores = np.asarray(confidence_scores, dtype=np.float64)
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(known, scores, pos_label=1)
        balanced = (tpr + (1.0 - fpr)) / 2.0
        index = int(np.nanargmax(balanced))
        threshold = float(thresholds[index])
    predicted_known = scores >= threshold
    known_recall = float(predicted_known[known == 1].mean()) if np.any(known == 1) else float("nan")
    unknown_recall = float((~predicted_known[known == 0]).mean()) if np.any(known == 0) else float("nan")
    return {
        "known_auroc": float(roc_auc_score(known, scores)),
        "confidence_threshold": float(threshold),
        "known_recall": known_recall,
        "unknown_recall": unknown_recall,
        "balanced_accuracy": float(np.nanmean([known_recall, unknown_recall])),
    }

