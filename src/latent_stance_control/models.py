from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class StanceOutput:
    source_logits: torch.Tensor
    target_logits: torch.Tensor
    source_vector: torch.Tensor
    target_vector: torch.Tensor


class DebertaStancePredictor(nn.Module):
    def __init__(self, model_name: str, num_clusters: int, vector_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use DebertaStancePredictor.") from exc
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.source_cluster = nn.Linear(hidden, num_clusters)
        self.target_cluster = nn.Linear(hidden, num_clusters)
        self.source_vector = nn.Linear(hidden, vector_dim)
        self.target_vector = nn.Linear(hidden, vector_dim)
        self.float()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> StanceOutput:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(output.last_hidden_state[:, 0])

        # Some local/HF checkpoints may emit fp16/bf16 hidden states while the
        # freshly initialized stance heads stay in fp32. Keep the heads in their
        # own dtype and cast only the pooled representation before projection.
        head_dtype = self.source_cluster.weight.dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(dtype=head_dtype)

        return StanceOutput(
            source_logits=self.source_cluster(pooled),
            target_logits=self.target_cluster(pooled),
            source_vector=self.source_vector(pooled),
            target_vector=self.target_vector(pooled),
        )


def soft_ce_from_logits(logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    return -(target_prob * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def stance_loss(output: StanceOutput, batch: Dict[str, torch.Tensor], cluster_weight: float = 1.0, vector_weight: float = 0.2) -> torch.Tensor:
    loss = cluster_weight * (
        soft_ce_from_logits(output.source_logits, batch["source_cluster"])
        + soft_ce_from_logits(output.target_logits, batch["target_cluster"])
    )
    loss = loss + vector_weight * (
        F.mse_loss(output.source_vector, batch["source_vector"])
        + F.mse_loss(output.target_vector, batch["target_vector"])
    )
    return loss


class StancePrefixProjector(nn.Module):
    """Projects a stance vector into internal virtual-token embeddings."""

    def __init__(self, stance_dim: int, hidden_size: int, prefix_tokens: int = 8) -> None:
        super().__init__()
        self.prefix_tokens = prefix_tokens
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(stance_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, prefix_tokens * hidden_size),
        )

    def forward(self, stance: torch.Tensor) -> torch.Tensor:
        batch = stance.shape[0]
        return self.net(stance).view(batch, self.prefix_tokens, self.hidden_size)
