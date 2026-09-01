from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import build_model_text, load_prepared_split, read_json, write_json


ROLE_PAIRS = ("A->B", "B->A", "A->A", "B->B")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_transitions(value: str) -> set[str] | None:
    value = value.strip()
    if value.lower() in {"all", "*"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def target_top(row: dict) -> int:
    return int(np.argmax(np.asarray(row["target_cluster"], dtype=np.float64)))


def target_top_prob(row: dict) -> float:
    q = np.asarray(row["target_cluster"], dtype=np.float64)
    return float(q.max())


def make_gate_rows(
    rows: List[dict],
    c7_cluster: int,
    negative_clusters: Sequence[int],
    pos_min_prob: float,
    neg_min_prob: float,
    transitions: set[str] | None,
) -> List[dict]:
    neg_set = set(int(x) for x in negative_clusters)
    out: List[dict] = []
    for row in rows:
        if transitions is not None and row.get("transition", "A->B") not in transitions:
            continue
        top = target_top(row)
        prob = target_top_prob(row)
        label = None
        if top == c7_cluster and prob >= pos_min_prob:
            label = 1.0
        elif top in neg_set and prob >= neg_min_prob:
            label = 0.0
        if label is None:
            continue
        item = dict(row)
        item["c7_gate_label"] = label
        item["c7_gold_top"] = top
        item["c7_gold_top_prob"] = prob
        out.append(item)
    return out


def downsample_negatives(rows: List[dict], neg_ratio: float, seed: int) -> List[dict]:
    if neg_ratio <= 0:
        return rows
    pos = [row for row in rows if float(row["c7_gate_label"]) == 1.0]
    neg = [row for row in rows if float(row["c7_gate_label"]) == 0.0]
    max_neg = min(len(neg), int(round(len(pos) * neg_ratio)))
    rng = random.Random(seed)
    neg = rng.sample(neg, max_neg) if max_neg < len(neg) else neg
    merged = pos + neg
    rng.shuffle(merged)
    return merged


class C7GateDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(build_model_text(row), truncation=True, max_length=self.max_length, padding=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": float(row.get("c7_gate_label", 0.0)),
        }


def collate(batch: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, torch.Tensor]:
    encoded = tokenizer.pad(
        {"input_ids": [x["input_ids"] for x in batch], "attention_mask": [x["attention_mask"] for x in batch]},
        return_tensors="pt",
    )
    encoded["label"] = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    return encoded


class C7GateModel(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("Install transformers to train the c7 gate.") from exc
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)
        self.float()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(output.last_hidden_state[:, 0])
        head_dtype = self.classifier.weight.dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(dtype=head_dtype)
        return self.classifier(pooled).squeeze(-1)


def load_gate_model(gate_dir: str | Path, device: str = "cpu") -> tuple[C7GateModel, Any, Dict[str, Any]]:
    from transformers import AutoTokenizer

    gate_dir = Path(gate_dir)
    config = read_json(gate_dir / "config.json")
    encoder_path = gate_dir / "encoder"
    tokenizer_path = gate_dir / "tokenizer"
    model = C7GateModel(str(encoder_path), dropout=float(config.get("dropout", 0.1))).to(device)
    state = torch.load(gate_dir / "c7_gate.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else encoder_path, use_fast=True)
    return model, tokenizer, config


@torch.no_grad()
def predict_gate(model: C7GateModel, tokenizer: Any, rows: List[dict], max_length: int, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(C7GateDataset(rows, tokenizer, max_length), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer))
    probs: List[np.ndarray] = []
    model.eval()
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    pred = (probs >= threshold).astype(np.int64)
    tp = float(((labels == 1) & (pred == 1)).sum())
    fp = float(((labels == 0) & (pred == 1)).sum())
    fn = float(((labels == 1) & (pred == 0)).sum())
    tn = float(((labels == 0) & (pred == 0)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "pred_positive_rate": float(pred.mean()) if pred.size else 0.0,
        "mean_prob": float(probs.mean()) if probs.size else 0.0,
    }


def best_threshold_by_f1(labels: np.ndarray, probs: np.ndarray) -> tuple[float, Dict[str, float]]:
    candidates = sorted(set([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8] + probs.tolist()))
    best_t = 0.5
    best_m = binary_metrics(labels, probs, best_t)
    for threshold in candidates:
        m = binary_metrics(labels, probs, threshold)
        if (m["f1"], m["precision"], m["recall"]) > (best_m["f1"], best_m["precision"], best_m["recall"]):
            best_t, best_m = float(threshold), m
    return best_t, best_m


def eval_split(model: C7GateModel, tokenizer: Any, rows: List[dict], args, threshold: float | None = None) -> tuple[Dict[str, float], np.ndarray, float]:
    labels = np.asarray([row["c7_gate_label"] for row in rows], dtype=np.float64)
    probs = predict_gate(model, tokenizer, rows, args.max_length, args.batch_size, args.device)
    if threshold is None:
        threshold, metrics = best_threshold_by_f1(labels, probs)
    else:
        metrics = binary_metrics(labels, probs, threshold)
    return metrics, probs, float(threshold)


def write_prediction_jsonl(path: Path, rows: List[dict], probs: np.ndarray, threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row, prob in zip(rows, probs):
            f.write(
                json.dumps(
                    {
                        "dialogue_id": row.get("dialogue_id"),
                        "turn_id": row.get("turn_id"),
                        "split": row.get("split"),
                        "transition": row.get("transition"),
                        "label": float(row["c7_gate_label"]),
                        "gold_target_top1": int(row["c7_gold_top"]),
                        "gold_target_top1_prob": float(row["c7_gold_top_prob"]),
                        "gate_prob": float(prob),
                        "gate_pred": int(float(prob) >= threshold),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary gate for whether cluster 7 playful/teasing control is suitable.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--c7-cluster", type=int, default=7)
    parser.add_argument("--negative-clusters", default="1,2,3,4,5,6")
    parser.add_argument("--transitions", default="all", help="Comma-separated transitions or 'all'. Quote values like 'A->B'.")
    parser.add_argument("--pos-min-prob", type=float, default=0.8)
    parser.add_argument("--neg-min-prob", type=float, default=0.8)
    parser.add_argument("--neg-ratio", type=float, default=3.0, help="Train-time negatives per positive. <=0 keeps all negatives.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_seed(args.seed)

    try:
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("Install transformers to train the c7 gate.") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(args.prepared)
    neg_clusters = parse_int_list(args.negative_clusters)
    transitions = parse_transitions(args.transitions)

    raw_train = load_prepared_split(prepared, "train")
    raw_dev = load_prepared_split(prepared, "dev") or load_prepared_split(prepared, "validation")
    raw_test = load_prepared_split(prepared, "test")
    train_rows = make_gate_rows(raw_train, args.c7_cluster, neg_clusters, args.pos_min_prob, args.neg_min_prob, transitions)
    dev_rows = make_gate_rows(raw_dev, args.c7_cluster, neg_clusters, args.pos_min_prob, args.neg_min_prob, transitions)
    test_rows = make_gate_rows(raw_test, args.c7_cluster, neg_clusters, args.pos_min_prob, args.neg_min_prob, transitions)
    sampled_train = downsample_negatives(train_rows, args.neg_ratio, args.seed)
    if not sampled_train:
        raise RuntimeError("No training rows selected for c7 gate. Relax thresholds or transitions.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = C7GateModel(args.model, dropout=args.dropout).float().to(args.device)
    labels = np.asarray([row["c7_gate_label"] for row in sampled_train], dtype=np.float64)
    pos = max(float(labels.sum()), 1.0)
    neg = max(float((labels == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=args.device)

    train_loader = DataLoader(C7GateDataset(sampled_train, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)

    best_threshold = 0.5
    best_f1 = -1.0
    best_path = out / "best_c7_gate.pt"
    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"].to(args.device), batch["attention_mask"].to(args.device))
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"].to(args.device), pos_weight=pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite c7 gate loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        epoch_summary: Dict[str, Any] = {"epoch": epoch + 1, "train_loss": total_loss / max(len(train_loader), 1)}
        if dev_rows:
            dev_metrics, _, dev_threshold = eval_split(model, tokenizer, dev_rows, args, threshold=None)
            epoch_summary.update({f"dev_{k}": v for k, v in dev_metrics.items()})
            if dev_metrics["f1"] > best_f1:
                best_f1 = dev_metrics["f1"]
                best_threshold = dev_threshold
                torch.save(model.state_dict(), best_path)
                write_json(out / "best_epoch.json", {"best_epoch": epoch + 1, "best_dev_f1": best_f1, "best_threshold": best_threshold})
        history.append(epoch_summary)
        write_json(out / "train_history.json", history)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=args.device))

    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save(model.state_dict(), out / "c7_gate.pt")

    metrics: Dict[str, Dict[str, float]] = {}
    for split, rows in [("train_sampled", sampled_train), ("dev", dev_rows), ("test", test_rows)]:
        if not rows:
            continue
        split_metrics, probs, _ = eval_split(model, tokenizer, rows, args, threshold=best_threshold)
        metrics[split] = split_metrics
        if split in {"dev", "test"}:
            write_prediction_jsonl(out / f"{split}_gate_predictions.jsonl", rows, probs, best_threshold)

    config = {
        "model": args.model,
        "c7_cluster": args.c7_cluster,
        "negative_clusters": neg_clusters,
        "transitions": "all" if transitions is None else sorted(transitions),
        "pos_min_prob": args.pos_min_prob,
        "neg_min_prob": args.neg_min_prob,
        "neg_ratio": args.neg_ratio,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "seed": args.seed,
        "best_threshold": best_threshold,
        "train_rows_full": len(train_rows),
        "train_rows_sampled": len(sampled_train),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "train_positive": int(sum(row["c7_gate_label"] for row in train_rows)),
        "sampled_train_positive": int(sum(row["c7_gate_label"] for row in sampled_train)),
    }
    write_json(out / "config.json", config)
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
