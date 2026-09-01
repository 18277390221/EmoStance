from __future__ import annotations

from typing import Dict, List

import numpy as np

ROLE_PAIRS = ("A->B", "B->A", "A->A", "B->B")


def build_transition_matrices(rows: List[dict], num_clusters: int, alpha: float = 0.05, entropy_weight: bool = True) -> Dict[str, np.ndarray]:
    counts = {role: np.zeros((num_clusters, num_clusters), dtype=np.float64) for role in ROLE_PAIRS}
    for row in rows:
        transition = row.get("transition", "A->B")
        counts.setdefault(transition, np.zeros((num_clusters, num_clusters), dtype=np.float64))
        src = np.asarray(row["source_cluster"], dtype=np.float64)
        tgt = np.asarray(row["target_cluster"], dtype=np.float64)
        weight = 1.0
        if entropy_weight:
            max_entropy = np.log(max(num_clusters, 2))
            src_conf = 1.0 - float(row.get("source_entropy", 0.0)) / max_entropy
            tgt_conf = 1.0 - float(row.get("target_entropy", 0.0)) / max_entropy
            weight = max(0.05, src_conf * tgt_conf)
        counts[transition] += weight * np.outer(src, tgt)
    matrices: Dict[str, np.ndarray] = {}
    for role, mat in counts.items():
        mat = mat + alpha
        matrices[role] = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1e-12)
    return matrices


def graph_prior(source_cluster: np.ndarray, transition: str, matrices: Dict[str, np.ndarray]) -> np.ndarray:
    matrix = matrices.get(transition, matrices.get("A->B"))
    prior = np.asarray(source_cluster, dtype=np.float64) @ np.asarray(matrix, dtype=np.float64)
    prior /= max(float(prior.sum()), 1e-12)
    return prior.astype(np.float32)
