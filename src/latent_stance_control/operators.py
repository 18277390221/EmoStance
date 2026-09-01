from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn


class LowRankRoleOperator(nn.Module):
    """Differentiable low-rank residual operator for later neural training."""

    def __init__(self, num_clusters: int, vector_dim: int, rank: int = 8, role_pairs: Tuple[str, ...] = ("A->B", "B->A", "A->A", "B->B")) -> None:
        super().__init__()
        self.role_to_idx = {role: idx for idx, role in enumerate(role_pairs)}
        shape = (len(role_pairs), num_clusters, num_clusters)
        self.u = nn.Parameter(torch.zeros(*shape, vector_dim, rank))
        self.v = nn.Parameter(torch.zeros(*shape, rank, vector_dim))
        self.bias = nn.Parameter(torch.zeros(*shape, vector_dim))
        nn.init.normal_(self.u, std=0.01)
        nn.init.normal_(self.v, std=0.01)

    def forward(self, z_src: torch.Tensor, source_cluster: torch.Tensor, target_cluster: torch.Tensor, transition_ids: Iterable[str]) -> torch.Tensor:
        outputs = []
        for idx, transition in enumerate(transition_ids):
            role_idx = self.role_to_idx.get(str(transition), self.role_to_idx.get("A->B", 0))
            pair_weight = source_cluster[idx, :, None] * target_cluster[idx, None, :]
            residual = torch.einsum("ij,ijdr,ijrh,h->d", pair_weight, self.u[role_idx], self.v[role_idx], z_src[idx])
            bias = torch.einsum("ij,ijd->d", pair_weight, self.bias[role_idx])
            outputs.append(z_src[idx] + residual + bias)
        return torch.stack(outputs, dim=0)


def ridge_affine_operator(source_vectors: np.ndarray, target_vectors: np.ndarray, l2: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source_vectors, dtype=np.float64)
    y = np.asarray(target_vectors, dtype=np.float64)
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    design = np.concatenate([x, ones], axis=1)
    reg = l2 * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return coef[:-1].T.astype(np.float32), coef[-1].astype(np.float32)
