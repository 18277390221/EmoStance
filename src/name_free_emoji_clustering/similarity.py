from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .embeddings import cosine_matrix
from .soft_labels import CanonicalData


def build_soft_label_matrix(data: CanonicalData) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
    utterance_index = {utterance.utterance_id: idx for idx, utterance in enumerate(data.utterances)}
    emoji_index = {emoji: idx for idx, emoji in enumerate(data.emojis)}
    q = np.zeros((len(data.utterances), len(data.emojis)), dtype=np.float32)
    for entry in data.entries:
        q[utterance_index[entry.utterance_id], emoji_index[entry.emoji]] = entry.soft_prob
    return q, utterance_index, emoji_index


def build_centroids(
    data: CanonicalData,
    q: np.ndarray,
    embeddings: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    confidence = np.ones((len(data.utterances), 1), dtype=np.float32)
    available = [
        utterance.utterance_mean_confidence
        for utterance in data.utterances
        if utterance.utterance_mean_confidence is not None
    ]
    if available:
        for idx, utterance in enumerate(data.utterances):
            if utterance.utterance_mean_confidence is not None:
                confidence[idx, 0] = float(utterance.utterance_mean_confidence) / 5.0

    weights = q * confidence
    effective_counts = weights.sum(axis=0)
    raw_centroids = np.zeros((len(data.emojis), embeddings.shape[1]), dtype=np.float32)
    nonzero = effective_counts > 0
    raw_centroids[nonzero] = (weights[:, nonzero].T @ embeddings) / effective_counts[nonzero, None]

    global_mean = embeddings.mean(axis=0).astype(np.float32)
    observed = np.array([float(data.observed_counts[emoji]) for emoji in data.emojis], dtype=np.float32)
    alpha = observed / (observed + float(tau))
    shrunk_centroids = alpha[:, None] * raw_centroids + (1.0 - alpha[:, None]) * global_mean[None, :]
    return raw_centroids, shrunk_centroids.astype(np.float32), effective_counts, alpha


def build_confusion_similarity(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = q.T @ q
    diag = np.diag(raw)
    denom = np.sqrt(np.outer(diag, diag))
    sim = np.divide(raw, denom, out=np.zeros_like(raw), where=denom > 0)
    np.fill_diagonal(sim, 1.0)
    return raw.astype(np.float32), sim.astype(np.float32)


def normalize_similarity_for_fusion(matrix: np.ndarray) -> np.ndarray:
    result = matrix.astype(np.float32, copy=True)
    n = result.shape[0]
    if n <= 1:
        return np.ones_like(result, dtype=np.float32)
    mask = ~np.eye(n, dtype=bool)
    values = result[mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        normalized = np.zeros_like(result, dtype=np.float32)
        np.fill_diagonal(normalized, 1.0)
        return normalized
    min_value = float(finite.min())
    max_value = float(finite.max())
    if abs(max_value - min_value) < 1e-12:
        normalized = np.zeros_like(result, dtype=np.float32)
    else:
        normalized = (result - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(normalized, 1.0)
    return normalized


def fuse_similarities(
    context_similarity: np.ndarray,
    confusion_similarity: np.ndarray,
    lambda_ctx: float,
    lambda_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx_norm = normalize_similarity_for_fusion(context_similarity)
    conf_norm = normalize_similarity_for_fusion(confusion_similarity)
    total = lambda_ctx + lambda_conf
    if total <= 0:
        raise ValueError("Fusion weights must have positive sum.")
    fused = (lambda_ctx / total) * ctx_norm + (lambda_conf / total) * conf_norm
    np.fill_diagonal(fused, 1.0)
    return fused.astype(np.float32), ctx_norm, conf_norm


def write_square_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["emoji", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[f"{float(value):.10f}" for value in row]])


def write_centroids(
    path: Path,
    npz_path: Path,
    emojis: list[str],
    observed_counts: dict[str, int],
    effective_counts: np.ndarray,
    alpha: np.ndarray,
    raw_centroids: np.ndarray,
    shrunk_centroids: np.ndarray,
) -> None:
    np.savez_compressed(
        npz_path,
        emojis=np.array(emojis),
        raw_centroids=raw_centroids,
        shrunk_centroids=shrunk_centroids,
        effective_counts=effective_counts,
        observed_counts=np.array([observed_counts[emoji] for emoji in emojis], dtype=np.float32),
        shrinkage_alpha=alpha,
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "emoji",
            "observed_count",
            "effective_count",
            "shrinkage_alpha",
            "raw_centroid_json",
            "shrunk_centroid_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, emoji in enumerate(emojis):
            writer.writerow(
                {
                    "emoji": emoji,
                    "observed_count": observed_counts[emoji],
                    "effective_count": f"{float(effective_counts[idx]):.8f}",
                    "shrinkage_alpha": f"{float(alpha[idx]):.8f}",
                    "raw_centroid_json": json.dumps(
                        [round(float(value), 8) for value in raw_centroids[idx]],
                        ensure_ascii=False,
                    ),
                    "shrunk_centroid_json": json.dumps(
                        [round(float(value), 8) for value in shrunk_centroids[idx]],
                        ensure_ascii=False,
                    ),
                }
            )


def top_neighbors(
    labels: list[str],
    matrix: np.ndarray,
    observed_counts: dict[str, int],
    source_count: int = 20,
    neighbor_count: int = 10,
) -> list[dict[str, object]]:
    frequent_labels = sorted(labels, key=lambda emoji: (-observed_counts[emoji], emoji))[:source_count]
    label_index = {label: idx for idx, label in enumerate(labels)}
    rows: list[dict[str, object]] = []
    for source in frequent_labels:
        i = label_index[source]
        order = sorted(
            (j for j in range(len(labels)) if j != i),
            key=lambda j: (-float(matrix[i, j]), labels[j]),
        )[:neighbor_count]
        for rank, j in enumerate(order, start=1):
            rows.append(
                {
                    "source_emoji": source,
                    "source_observed_count": observed_counts[source],
                    "rank": rank,
                    "neighbor_emoji": labels[j],
                    "neighbor_observed_count": observed_counts[labels[j]],
                    "similarity": f"{float(matrix[i, j]):.10f}",
                }
            )
    return rows


def write_neighbor_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source_emoji",
        "source_observed_count",
        "rank",
        "neighbor_emoji",
        "neighbor_observed_count",
        "similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def context_similarity(shrunk_centroids: np.ndarray) -> np.ndarray:
    return cosine_matrix(shrunk_centroids).astype(np.float32)
