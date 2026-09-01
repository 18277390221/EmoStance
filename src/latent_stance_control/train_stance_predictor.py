from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StanceDataset, load_prepared_split, read_json, write_json
from .metrics import evaluate_cluster_predictions, evaluate_vectors
from .models import DebertaStancePredictor, stance_loss


def collate(batch: List[Dict], tokenizer) -> Dict[str, torch.Tensor]:
    encoded = tokenizer.pad(
        {"input_ids": [x["input_ids"] for x in batch], "attention_mask": [x["attention_mask"] for x in batch]},
        return_tensors="pt",
    )
    for key in ("source_cluster", "target_cluster", "source_vector", "target_vector"):
        encoded[key] = torch.tensor(np.stack([x[key] for x in batch]), dtype=torch.float32)
    encoded["transition_id"] = [x["transition_id"] for x in batch]
    return encoded


@torch.no_grad()
def predict(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    preds = {"source_cluster": [], "target_cluster": [], "source_vector": [], "target_vector": []}
    gold = {"source_cluster": [], "target_cluster": [], "source_vector": [], "target_vector": []}
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for name, tensor in {
            "source_logits": output.source_logits,
            "target_logits": output.target_logits,
            "source_vector": output.source_vector,
            "target_vector": output.target_vector,
        }.items():
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite model output during prediction: {name}")
        preds["source_cluster"].append(torch.softmax(output.source_logits, dim=-1).cpu().numpy())
        preds["target_cluster"].append(torch.softmax(output.target_logits, dim=-1).cpu().numpy())
        preds["source_vector"].append(output.source_vector.cpu().numpy())
        preds["target_vector"].append(output.target_vector.cpu().numpy())
        for key in gold:
            gold[key].append(batch[key].numpy())
    return {f"pred_{k}": np.concatenate(v, axis=0) for k, v in preds.items()} | {f"gold_{k}": np.concatenate(v, axis=0) for k, v in gold.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DeBERTa stance predictor.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--vector-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError("Install transformers to train the stance predictor.") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_json(Path(args.prepared) / "meta.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_rows = load_prepared_split(args.prepared, "train")
    dev_rows = load_prepared_split(args.prepared, "dev") or load_prepared_split(args.prepared, "validation")
    test_rows = load_prepared_split(args.prepared, "test")

    model = DebertaStancePredictor(args.model, meta["num_clusters"], meta["vector_dim"]).float().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(StanceDataset(train_rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer))
    total_steps = max(len(train_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            output = model(batch["input_ids"], batch["attention_mask"])
            loss = stance_loss(output, batch, vector_weight=args.vector_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected. The most common cause is fp16 "
                    "encoder weights or an unstable learning rate. This script now forces "
                    "fp32 weights; if this persists, retry with --lr 1e-5 or a smaller batch."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        write_json(out / f"epoch_{epoch + 1}.json", {"train_loss": total_loss / max(len(train_loader), 1)})

    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save(model.state_dict(), out / "stance_predictor.pt")

    metrics: Dict[str, Dict[str, float]] = {}
    for split, rows in (("dev", dev_rows), ("test", test_rows)):
        if not rows:
            continue
        loader = DataLoader(StanceDataset(rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate(b, tokenizer))
        result = predict(model, loader, args.device)
        np.savez_compressed(out / f"{split}_predictions.npz", **result)
        split_metrics = {}
        split_metrics |= {f"source_{k}": v for k, v in evaluate_cluster_predictions(result["gold_source_cluster"], result["pred_source_cluster"]).items()}
        split_metrics |= {f"target_{k}": v for k, v in evaluate_cluster_predictions(result["gold_target_cluster"], result["pred_target_cluster"]).items()}
        split_metrics |= {f"source_{k}": v for k, v in evaluate_vectors(result["gold_source_vector"], result["pred_source_vector"]).items()}
        split_metrics |= {f"target_{k}": v for k, v in evaluate_vectors(result["gold_target_vector"], result["pred_target_vector"]).items()}
        metrics[split] = split_metrics
    write_json(out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
