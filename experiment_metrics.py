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
) -> dict[str, float]:
    """Compute Recall@K and MAP@R without materializing the full Q x G matrix."""
    q = F.normalize(query_embeddings.float(), dim=1)
    g = F.normalize(gallery_embeddings.float(), dim=1).to(q.device)
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
        scores = q[start:stop] @ g.T
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

