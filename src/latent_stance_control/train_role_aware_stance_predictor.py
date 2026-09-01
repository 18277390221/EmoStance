from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import build_model_text, load_prepared_split, read_json, write_json
from .metrics import evaluate_cluster_predictions, evaluate_vectors
from .transitions import ROLE_PAIRS, build_transition_matrices
from .run_ablations import cluster_prototypes


ROLE_TO_ID = {"A": 0, "B": 1}
TRANSITION_TO_ID = {role: idx for idx, role in enumerate(ROLE_PAIRS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_rows(rows: List[dict], limit: int | None, seed: int) -> List[dict]:
    if not limit or limit <= 0 or len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(rows), size=limit, replace=False))
    return [rows[int(i)] for i in indices]


def make_loader(
    rows: List[dict],
    tokenizer: Any,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        RoleAwareStanceDataset(rows, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: collate(b, tokenizer),
    )


@dataclass
class RoleAwareOutput:
    source_logits: torch.Tensor
    target_logits: torch.Tensor
    target_text_logits: torch.Tensor
    source_vector: torch.Tensor
    target_vector: torch.Tensor
    graph_gate: torch.Tensor
    graph_prior: torch.Tensor


class RoleAwareStanceDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer: Any, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            build_model_text(row),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        role = str(row.get("role", "A"))
        next_role = str(row.get("next_role", "B"))
        transition = str(row.get("transition", f"{role}->{next_role}"))
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "role_id": ROLE_TO_ID.get(role, 0),
            "next_role_id": ROLE_TO_ID.get(next_role, 1),
            "transition_idx": TRANSITION_TO_ID.get(transition, 0),
            "source_cluster": np.asarray(row["source_cluster"], dtype=np.float32),
            "target_cluster": np.asarray(row["target_cluster"], dtype=np.float32),
            "source_vector": np.asarray(row["source_vector"], dtype=np.float32),
            "target_vector": np.asarray(row["target_vector"], dtype=np.float32),
        }


def collate(batch: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, torch.Tensor]:
    encoded = tokenizer.pad(
        {"input_ids": [x["input_ids"] for x in batch], "attention_mask": [x["attention_mask"] for x in batch]},
        return_tensors="pt",
    )
    encoded["role_id"] = torch.tensor([x["role_id"] for x in batch], dtype=torch.long)
    encoded["next_role_id"] = torch.tensor([x["next_role_id"] for x in batch], dtype=torch.long)
    encoded["transition_idx"] = torch.tensor([x["transition_idx"] for x in batch], dtype=torch.long)
    for key in ("source_cluster", "target_cluster", "source_vector", "target_vector"):
        encoded[key] = torch.tensor(np.stack([x[key] for x in batch]), dtype=torch.float32)
    return encoded


class RoleAwareDebertaStancePredictor(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_clusters: int,
        vector_dim: int,
        transition_matrices: np.ndarray,
        cluster_prototypes_array: np.ndarray,
        source_prototypes_array: np.ndarray,
        role_dim: int = 32,
        hidden_dropout: float = 0.1,
        graph_prior_weight: float = 1.0,
        entropy_gated_graph: bool = True,
        detach_source_features: bool = True,
        source_vector_feature_mode: str = "direct",
        use_role_features: bool = True,
    ) -> None:
        super().__init__()
        if source_vector_feature_mode not in {"direct", "none", "prototype"}:
            raise ValueError(
                "--source-vector-feature-mode must be one of: direct, none, prototype"
            )
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to use RoleAwareDebertaStancePredictor.") from exc
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden = int(self.encoder.config.hidden_size)
        self.num_clusters = num_clusters
        self.vector_dim = vector_dim
        self.graph_prior_weight = graph_prior_weight
        self.entropy_gated_graph = entropy_gated_graph
        self.detach_source_features = detach_source_features
        self.source_vector_feature_mode = source_vector_feature_mode
        self.use_role_features = use_role_features

        self.role_embedding = nn.Embedding(2, role_dim)
        self.next_role_embedding = nn.Embedding(2, role_dim)
        self.transition_embedding = nn.Embedding(len(ROLE_PAIRS), role_dim)
        self.dropout = nn.Dropout(hidden_dropout)

        self.source_cluster = nn.Linear(hidden, num_clusters)
        self.source_vector = nn.Linear(hidden, vector_dim)

        target_in = hidden + role_dim * 3 + num_clusters
        if source_vector_feature_mode != "none":
            target_in += vector_dim
        self.target_mlp = nn.Sequential(
            nn.Linear(target_in, hidden),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
        )
        self.target_cluster = nn.Linear(hidden // 2, num_clusters)
        self.graph_gate = nn.Linear(hidden // 2, 1)

        transition_tensor = torch.tensor(transition_matrices, dtype=torch.float32)
        target_prototypes_tensor = torch.tensor(cluster_prototypes_array, dtype=torch.float32)
        source_prototypes_tensor = torch.tensor(source_prototypes_array, dtype=torch.float32)
        self.register_buffer("transition_matrices", transition_tensor)
        self.register_buffer("cluster_prototypes", target_prototypes_tensor)
        self.register_buffer("source_cluster_prototypes", source_prototypes_tensor)
        self.float()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_id: torch.Tensor,
        next_role_id: torch.Tensor,
        transition_idx: torch.Tensor,
    ) -> RoleAwareOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(encoded.last_hidden_state[:, 0])
        head_dtype = self.source_cluster.weight.dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(dtype=head_dtype)

        source_logits = self.source_cluster(pooled)
        source_prob = torch.softmax(source_logits, dim=-1)
        direct_source_vector = self.source_vector(pooled)
        prototype_source_vector = source_prob @ self.source_cluster_prototypes.to(source_prob.device)
        source_vector = prototype_source_vector if self.source_vector_feature_mode == "prototype" else direct_source_vector

        source_for_target = source_prob.detach() if self.detach_source_features else source_prob
        if self.use_role_features:
            role_feature = self.role_embedding(role_id)
            next_role_feature = self.next_role_embedding(next_role_id)
            transition_feature = self.transition_embedding(transition_idx)
        else:
            role_feature = torch.zeros(role_id.shape[0], self.role_embedding.embedding_dim, device=pooled.device, dtype=pooled.dtype)
            next_role_feature = torch.zeros_like(role_feature)
            transition_feature = torch.zeros_like(role_feature)
        feature_parts = [
            pooled,
            role_feature,
            next_role_feature,
            transition_feature,
            source_for_target,
        ]
        if self.source_vector_feature_mode != "none":
            vector_for_target = source_vector.detach() if self.detach_source_features else source_vector
            feature_parts.append(vector_for_target)
        features = torch.cat(feature_parts, dim=-1)
        target_hidden = self.target_mlp(features)
        target_text_logits = self.target_cluster(target_hidden)

        matrices = self.transition_matrices.to(target_text_logits.device)[transition_idx]
        graph_prior = torch.bmm(source_prob.unsqueeze(1), matrices).squeeze(1)
        graph_prior = graph_prior / torch.clamp(graph_prior.sum(dim=-1, keepdim=True), min=1e-12)
        log_graph = torch.log(torch.clamp(graph_prior, min=1e-12))

        gate = torch.sigmoid(self.graph_gate(target_hidden))
        if self.entropy_gated_graph:
            text_prob = torch.softmax(target_text_logits, dim=-1)
            entropy = -(text_prob * torch.log(torch.clamp(text_prob, min=1e-12))).sum(dim=-1, keepdim=True)
            gate = gate * torch.clamp(entropy / np.log(self.num_clusters), min=0.0, max=1.0)
        target_logits = target_text_logits + self.graph_prior_weight * gate * log_graph
        target_prob = torch.softmax(target_logits, dim=-1)
        target_vector = target_prob @ self.cluster_prototypes.to(target_prob.device)
        return RoleAwareOutput(
            source_logits=source_logits,
            target_logits=target_logits,
            target_text_logits=target_text_logits,
            source_vector=source_vector,
            target_vector=target_vector,
            graph_gate=gate.squeeze(-1),
            graph_prior=graph_prior,
        )


def harden_distribution(target_prob: torch.Tensor) -> torch.Tensor:
    hard = torch.zeros_like(target_prob)
    hard.scatter_(1, torch.argmax(target_prob, dim=-1, keepdim=True), 1.0)
    return hard


def soft_focal_ce_from_logits(
    logits: torch.Tensor,
    target_prob: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    prob = torch.exp(log_prob)
    loss_terms = -target_prob * log_prob
    if gamma > 0:
        loss_terms = loss_terms * torch.pow(torch.clamp(1.0 - prob, min=0.0), gamma)
    if class_weights is not None:
        loss_terms = loss_terms * class_weights.view(1, -1)
    return loss_terms.sum(dim=-1).mean()


def role_aware_loss(
    output: RoleAwareOutput,
    batch: Dict[str, torch.Tensor],
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    source_loss_weight: float,
    target_loss_weight: float,
    vector_loss_weight: float,
    focal_gamma: float,
    text_target_aux_weight: float,
    hard_label_training: bool = False,
) -> torch.Tensor:
    source_target = harden_distribution(batch["source_cluster"]) if hard_label_training else batch["source_cluster"]
    target_target = harden_distribution(batch["target_cluster"]) if hard_label_training else batch["target_cluster"]
    source_ce = soft_focal_ce_from_logits(output.source_logits, source_target, source_weights, gamma=0.0)
    target_ce = soft_focal_ce_from_logits(output.target_logits, target_target, target_weights, gamma=focal_gamma)
    text_target_ce = soft_focal_ce_from_logits(output.target_text_logits, target_target, target_weights, gamma=focal_gamma)
    vector_loss = F.mse_loss(output.source_vector, batch["source_vector"]) + F.mse_loss(output.target_vector, batch["target_vector"])
    return (
        source_loss_weight * source_ce
        + target_loss_weight * target_ce
        + text_target_aux_weight * text_target_ce
        + vector_loss_weight * vector_loss
    )


def row_array(rows: List[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def vector_prototypes(rows: List[dict], cluster_key: str, vector_key: str, num_clusters: int, vector_dim: int) -> np.ndarray:
    sums = np.zeros((num_clusters, vector_dim), dtype=np.float64)
    weights = np.zeros(num_clusters, dtype=np.float64)
    for row in rows:
        q = np.asarray(row[cluster_key], dtype=np.float64)
        z = np.asarray(row[vector_key], dtype=np.float64)
        sums += q[:, None] * z[None, :]
        weights += q
    global_mean = row_array(rows, vector_key).mean(axis=0)
    prototypes = sums / np.maximum(weights[:, None], 1e-12)
    prototypes[weights <= 1e-12] = global_mean
    return prototypes.astype(np.float32)


def class_weights_from_rows(rows: List[dict], key: str, num_clusters: int, power: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    dist = row_array(rows, key).sum(axis=0).astype(np.float64)
    dist = dist / max(float(dist.sum()), eps)
    weights = np.power(1.0 / np.maximum(dist, eps), power)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def transition_tensor_from_rows(rows: List[dict], num_clusters: int, alpha: float) -> np.ndarray:
    matrices = build_transition_matrices(rows, num_clusters, alpha=alpha)
    return np.stack([matrices[role] for role in ROLE_PAIRS]).astype(np.float32)


@torch.no_grad()
def predict(model, loader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    preds = {"source_cluster": [], "target_cluster": [], "source_vector": [], "target_vector": [], "graph_gate": [], "graph_prior": []}
    gold = {"source_cluster": [], "target_cluster": [], "source_vector": [], "target_vector": []}
    for batch in loader:
        tensor_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        output = model(
            tensor_batch["input_ids"],
            tensor_batch["attention_mask"],
            tensor_batch["role_id"],
            tensor_batch["next_role_id"],
            tensor_batch["transition_idx"],
        )
        for name, tensor in {
            "source_logits": output.source_logits,
            "target_logits": output.target_logits,
            "source_vector": output.source_vector,
            "target_vector": output.target_vector,
        }.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite role-aware output during prediction: {name}")
        preds["source_cluster"].append(torch.softmax(output.source_logits, dim=-1).cpu().numpy())
        preds["target_cluster"].append(torch.softmax(output.target_logits, dim=-1).cpu().numpy())
        preds["source_vector"].append(output.source_vector.cpu().numpy())
        preds["target_vector"].append(output.target_vector.cpu().numpy())
        preds["graph_gate"].append(output.graph_gate.cpu().numpy())
        preds["graph_prior"].append(output.graph_prior.cpu().numpy())
        for key in gold:
            gold[key].append(batch[key].numpy())
    result = {f"pred_{k}": np.concatenate(v, axis=0) for k, v in preds.items()}
    result.update({f"gold_{k}": np.concatenate(v, axis=0) for k, v in gold.items()})
    return result


def evaluate_prediction_result(result: Dict[str, np.ndarray]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics |= {f"source_{k}": v for k, v in evaluate_cluster_predictions(result["gold_source_cluster"], result["pred_source_cluster"]).items()}
    metrics |= {f"target_{k}": v for k, v in evaluate_cluster_predictions(result["gold_target_cluster"], result["pred_target_cluster"]).items()}
    metrics |= {f"source_{k}": v for k, v in evaluate_vectors(result["gold_source_vector"], result["pred_source_vector"]).items()}
    metrics |= {f"target_{k}": v for k, v in evaluate_vectors(result["gold_target_vector"], result["pred_target_vector"]).items()}
    metrics["mean_graph_gate"] = float(np.mean(result["pred_graph_gate"]))
    metrics["median_graph_gate"] = float(np.median(result["pred_graph_gate"]))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a role-aware stronger DeBERTa stance predictor.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--role-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--transition-alpha", type=float, default=0.05)
    parser.add_argument("--graph-prior-weight", type=float, default=0.5)
    parser.add_argument("--disable-entropy-gated-graph", action="store_true")
    parser.add_argument("--disable-role-features", action="store_true", help="Ablation: zero current-role, next-role, and transition embedding features.")
    parser.add_argument("--hard-label-training", action="store_true", help="Ablation: train cluster CE losses on argmax hard labels instead of soft distributions.")
    parser.add_argument("--allow-target-grad-to-source", action="store_true")
    parser.add_argument(
        "--source-vector-feature-mode",
        choices=["direct", "none", "prototype"],
        default="direct",
        help=(
            "Ablate how source vector enters the target MLP: direct uses the predicted source vector head; "
            "none removes source vector from target features; prototype uses pred_source_cluster @ source prototypes."
        ),
    )
    parser.add_argument("--source-loss-weight", type=float, default=0.4)
    parser.add_argument("--target-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-target-aux-weight", type=float, default=0.2)
    parser.add_argument("--vector-loss-weight", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional random subset for smoke/debug runs; 0 uses all train rows.")
    parser.add_argument("--max-eval-examples", type=int, default=0, help="Optional random subset for dev/test smoke runs; 0 uses all rows.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)

    try:
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("Install transformers to train the role-aware stance predictor.") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_json(Path(args.prepared) / "meta.json")
    train_rows = load_prepared_split(args.prepared, "train")
    dev_rows = load_prepared_split(args.prepared, "dev") or load_prepared_split(args.prepared, "validation")
    test_rows = load_prepared_split(args.prepared, "test")
    train_rows = maybe_limit_rows(train_rows, args.max_train_examples, args.seed)
    dev_rows = maybe_limit_rows(dev_rows, args.max_eval_examples, args.seed + 1)
    test_rows = maybe_limit_rows(test_rows, args.max_eval_examples, args.seed + 2)

    num_clusters = int(meta["num_clusters"])
    vector_dim = int(meta["vector_dim"])
    transition_matrices = transition_tensor_from_rows(train_rows, num_clusters, alpha=args.transition_alpha)
    prototypes = cluster_prototypes(train_rows, num_clusters, vector_dim)
    source_prototypes = vector_prototypes(train_rows, "source_cluster", "source_vector", num_clusters, vector_dim)
    source_weights = torch.tensor(class_weights_from_rows(train_rows, "source_cluster", num_clusters, args.class_weight_power), dtype=torch.float32, device=args.device)
    target_weights = torch.tensor(class_weights_from_rows(train_rows, "target_cluster", num_clusters, args.class_weight_power), dtype=torch.float32, device=args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = RoleAwareDebertaStancePredictor(
        args.model,
        num_clusters,
        vector_dim,
        transition_matrices=transition_matrices,
        cluster_prototypes_array=prototypes,
        source_prototypes_array=source_prototypes,
        role_dim=args.role_dim,
        hidden_dropout=args.dropout,
        graph_prior_weight=args.graph_prior_weight,
        entropy_gated_graph=not args.disable_entropy_gated_graph,
        detach_source_features=not args.allow_target_grad_to_source,
        source_vector_feature_mode=args.source_vector_feature_mode,
        use_role_features=not args.disable_role_features,
    ).float().to(args.device)

    train_loader = make_loader(train_rows, tokenizer, args.max_length, args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev_loader = make_loader(dev_rows, tokenizer, args.max_length, args.batch_size, shuffle=False, num_workers=args.num_workers) if dev_rows else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    train_history = []
    best_dev_soft_ce = float("inf")
    best_epoch = 0
    best_path = out / "best_role_aware_stance_predictor.pt"
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_gate = 0.0
        batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            output = model(batch["input_ids"], batch["attention_mask"], batch["role_id"], batch["next_role_id"], batch["transition_idx"])
            loss = role_aware_loss(
                output,
                batch,
                source_weights=source_weights,
                target_weights=target_weights,
                source_loss_weight=args.source_loss_weight,
                target_loss_weight=args.target_loss_weight,
                vector_loss_weight=args.vector_loss_weight,
                focal_gamma=args.focal_gamma,
                text_target_aux_weight=args.text_target_aux_weight,
                hard_label_training=args.hard_label_training,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite role-aware training loss detected. Try --lr 1e-5 or lower --graph-prior-weight.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            total_gate += float(output.graph_gate.detach().mean().cpu())
            batches += 1
        epoch_summary = {"epoch": epoch + 1, "train_loss": total_loss / max(batches, 1), "mean_graph_gate": total_gate / max(batches, 1)}
        if dev_loader is not None:
            dev_result = predict(model, dev_loader, args.device)
            dev_metrics = evaluate_prediction_result(dev_result)
            epoch_summary.update(
                {
                    "dev_target_soft_ce": dev_metrics["target_soft_ce"],
                    "dev_target_accuracy": dev_metrics["target_accuracy"],
                    "dev_target_macro_f1": dev_metrics["target_macro_f1"],
                    "dev_target_vector_cosine": dev_metrics["target_vector_cosine"],
                    "dev_mean_graph_gate": dev_metrics["mean_graph_gate"],
                }
            )
            if dev_metrics["target_soft_ce"] < best_dev_soft_ce:
                best_dev_soft_ce = dev_metrics["target_soft_ce"]
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
                write_json(out / "best_epoch.json", {"best_epoch": best_epoch, "dev_target_soft_ce": best_dev_soft_ce})
        train_history.append(epoch_summary)
        write_json(out / "train_history.json", train_history)
        write_json(out / f"epoch_{epoch + 1}.json", epoch_summary)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=args.device))

    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save(model.state_dict(), out / "role_aware_stance_predictor.pt")
    write_json(
        out / "config.json",
        {
            "model": args.model,
            "num_clusters": num_clusters,
            "vector_dim": vector_dim,
            "role_dim": args.role_dim,
            "transition_alpha": args.transition_alpha,
            "graph_prior_weight": args.graph_prior_weight,
            "best_epoch": best_epoch,
            "best_dev_target_soft_ce": best_dev_soft_ce if np.isfinite(best_dev_soft_ce) else None,
            "seed": args.seed,
            "max_train_examples": args.max_train_examples,
            "max_eval_examples": args.max_eval_examples,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "entropy_gated_graph": not args.disable_entropy_gated_graph,
            "use_role_features": not args.disable_role_features,
            "hard_label_training": args.hard_label_training,
            "detach_source_features": not args.allow_target_grad_to_source,
            "source_vector_feature_mode": args.source_vector_feature_mode,
            "source_loss_weight": args.source_loss_weight,
            "target_loss_weight": args.target_loss_weight,
            "text_target_aux_weight": args.text_target_aux_weight,
            "vector_loss_weight": args.vector_loss_weight,
            "focal_gamma": args.focal_gamma,
            "class_weight_power": args.class_weight_power,
            "source_class_weights": source_weights.detach().cpu().numpy().tolist(),
            "target_class_weights": target_weights.detach().cpu().numpy().tolist(),
            "role_pairs": list(ROLE_PAIRS),
        },
    )
    write_json(out / "transition_matrices.json", {role: transition_matrices[idx].tolist() for idx, role in enumerate(ROLE_PAIRS)})
    write_json(out / "cluster_prototypes.json", prototypes.tolist())
    write_json(out / "source_cluster_prototypes.json", source_prototypes.tolist())
    write_json(out / "target_cluster_prototypes.json", prototypes.tolist())

    metrics: Dict[str, Dict[str, float]] = {}
    for split, rows in (("dev", dev_rows), ("test", test_rows)):
        if not rows:
            continue
        loader = make_loader(rows, tokenizer, args.max_length, args.batch_size, shuffle=False, num_workers=args.num_workers)
        result = predict(model, loader, args.device)
        np.savez_compressed(out / f"{split}_predictions.npz", **result)
        metrics[split] = evaluate_prediction_result(result)
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
