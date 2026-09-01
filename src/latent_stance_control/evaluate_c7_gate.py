from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .data import load_prepared_split, write_json
from .metrics import evaluate_cluster_predictions, normalize_prob, soft_cross_entropy


def parse_splits(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def top_distribution(prob: np.ndarray) -> Dict[str, int]:
    prob = normalize_prob(prob)
    labels, counts = np.unique(np.argmax(prob, axis=-1), return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def c7_block(gold: np.ndarray, pred: np.ndarray, c7_cluster: int, mask: np.ndarray) -> Dict[str, float]:
    gold = normalize_prob(gold)
    pred = normalize_prob(pred)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.sum() == 0:
        return {
            "n": 0,
            "soft_ce": 0.0,
            "top1_accuracy": 0.0,
            "c7_top1_rate": 0.0,
            "mean_c7_prob": 0.0,
        }
    g = gold[mask]
    p = pred[mask]
    return {
        "n": int(mask.sum()),
        "soft_ce": soft_cross_entropy(g, p),
        "top1_accuracy": float((np.argmax(g, axis=-1) == np.argmax(p, axis=-1)).mean()),
        "c7_top1_rate": float((np.argmax(p, axis=-1) == c7_cluster).mean()),
        "mean_c7_prob": float(p[:, c7_cluster].mean()),
    }


def gate_block(rows: List[dict], gold: np.ndarray, gated: np.ndarray, c7_cluster: int) -> Dict[str, Any]:
    if not rows or "c7_gate_prob" not in rows[0]:
        return {}
    probs = np.asarray([float(row.get("c7_gate_prob", 0.0)) for row in rows], dtype=np.float64)
    preds = np.asarray([int(row.get("c7_gate_pred", 0)) for row in rows], dtype=np.int64)
    gold_top = np.argmax(normalize_prob(gold), axis=-1)
    gated_top = np.argmax(normalize_prob(gated), axis=-1)
    active = preds == 1
    inactive = ~active
    out: Dict[str, Any] = {
        "mean_gate_prob": float(probs.mean()) if probs.size else 0.0,
        "gate_active_rate": float(active.mean()) if active.size else 0.0,
        "active_n": int(active.sum()),
        "inactive_n": int(inactive.sum()),
        "active_gold_c7_rate": float((gold_top[active] == c7_cluster).mean()) if active.any() else 0.0,
        "inactive_gold_c7_rate": float((gold_top[inactive] == c7_cluster).mean()) if inactive.any() else 0.0,
        "active_gated_c7_top1_rate": float((gated_top[active] == c7_cluster).mean()) if active.any() else 0.0,
        "inactive_gated_c7_top1_rate": float((gated_top[inactive] == c7_cluster).mean()) if inactive.any() else 0.0,
    }
    return out


def transition_metrics(rows: List[dict], gold: np.ndarray, ungated: np.ndarray, gated: np.ndarray, c7_cluster: int) -> Dict[str, Any]:
    transitions = sorted({str(row.get("transition", "")) for row in rows})
    out: Dict[str, Any] = {}
    for transition in transitions:
        mask = np.asarray([str(row.get("transition", "")) == transition for row in rows], dtype=bool)
        if not mask.any():
            continue
        gold_top = np.argmax(normalize_prob(gold[mask]), axis=-1)
        out[transition or "unknown"] = {
            "n": int(mask.sum()),
            "gold_c7_top1_rate": float((gold_top == c7_cluster).mean()),
            "ungated": evaluate_cluster_predictions(gold[mask], ungated[mask]),
            "gated": evaluate_cluster_predictions(gold[mask], gated[mask]),
            "ungated_c7_top1_rate": float((np.argmax(normalize_prob(ungated[mask]), axis=-1) == c7_cluster).mean()),
            "gated_c7_top1_rate": float((np.argmax(normalize_prob(gated[mask]), axis=-1) == c7_cluster).mean()),
        }
    return out


def delta_block(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(before) | set(after))
    return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in keys if isinstance(before.get(key, 0.0), (int, float))}


def evaluate_split(prepared: Path, stance_dir: Path, gated_prepared: Path, split: str, c7_cluster: int) -> Dict[str, Any]:
    gold_rows = load_prepared_split(prepared, split)
    gated_rows = load_prepared_split(gated_prepared, split)
    if not gold_rows or not gated_rows:
        return {"rows": 0}
    if len(gold_rows) != len(gated_rows):
        raise ValueError(f"{split}: gold/gated row mismatch: {len(gold_rows)} vs {len(gated_rows)}")
    for i, (gold_row, gated_row) in enumerate(zip(gold_rows, gated_rows)):
        if gold_row.get("dialogue_id") != gated_row.get("dialogue_id") or int(gold_row.get("turn_id", -1)) != int(gated_row.get("turn_id", -2)):
            raise ValueError(f"{split}: row alignment mismatch at {i}")

    gold = normalize_prob(np.asarray([row["target_cluster"] for row in gold_rows], dtype=np.float64))
    gated = normalize_prob(np.asarray([row["target_cluster"] for row in gated_rows], dtype=np.float64))
    pred_path = stance_dir / f"{split}_predictions.npz"
    if pred_path.exists():
        with np.load(pred_path) as data:
            ungated = normalize_prob(np.asarray(data["pred_target_cluster"], dtype=np.float64))
    elif "ungated_target_cluster" in gated_rows[0]:
        ungated = normalize_prob(np.asarray([row["ungated_target_cluster"] for row in gated_rows], dtype=np.float64))
    else:
        raise FileNotFoundError(f"Missing ungated predictions for {split}: {pred_path}")

    if ungated.shape != gated.shape or gold.shape != gated.shape:
        raise ValueError(f"{split}: shape mismatch gold={gold.shape} ungated={ungated.shape} gated={gated.shape}")

    gold_top = np.argmax(gold, axis=-1)
    c7_gold = gold_top == c7_cluster
    non_c7_gold = ~c7_gold
    ungated_metrics = evaluate_cluster_predictions(gold, ungated)
    gated_metrics = evaluate_cluster_predictions(gold, gated)
    out: Dict[str, Any] = {
        "rows": len(gold_rows),
        "gold_c7_top1_count": int(c7_gold.sum()),
        "gold_c7_top1_rate": float(c7_gold.mean()),
        "ungated": ungated_metrics,
        "gated": gated_metrics,
        "delta_gated_minus_ungated": delta_block(ungated_metrics, gated_metrics),
        "top1_distribution": {
            "gold": top_distribution(gold),
            "ungated": top_distribution(ungated),
            "gated": top_distribution(gated),
        },
        "c7_gold_subset": {
            "ungated": c7_block(gold, ungated, c7_cluster, c7_gold),
            "gated": c7_block(gold, gated, c7_cluster, c7_gold),
        },
        "non_c7_gold_subset": {
            "ungated": c7_block(gold, ungated, c7_cluster, non_c7_gold),
            "gated": c7_block(gold, gated, c7_cluster, non_c7_gold),
        },
        "gate": gate_block(gated_rows, gold, gated, c7_cluster),
        "by_transition": transition_metrics(gold_rows, gold, ungated, gated, c7_cluster),
    }
    return out


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(path: Path, metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# C7 Gate Evaluation",
        "",
        f"Prepared: `{args.prepared}`",
        f"Ungated stance dir: `{args.stance_dir}`",
        f"Gated prepared: `{args.gated_prepared}`",
        f"C7 cluster: `{args.c7_cluster}`",
        "",
        "| split | rows | gold c7 rate | model | CE | acc | macro-F1 | c7 recall/top1 | non-c7 false c7 | mean c7 prob on c7 | mean c7 prob on non-c7 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, m in metrics.items():
        if int(m.get("rows", 0)) == 0:
            continue
        for model_name in ("ungated", "gated"):
            overall = m[model_name]
            c7_subset = m["c7_gold_subset"][model_name]
            non = m["non_c7_gold_subset"][model_name]
            lines.append(
                f"| {split} | {m['rows']} | {m['gold_c7_top1_rate']:.4f} | {model_name} | "
                f"{overall['soft_ce']:.4f} | {overall['accuracy']:.4f} | {overall['macro_f1']:.4f} | "
                f"{c7_subset['c7_top1_rate']:.4f} | {non['c7_top1_rate']:.4f} | "
                f"{c7_subset['mean_c7_prob']:.4f} | {non['mean_c7_prob']:.4f} |"
            )
    lines += ["", "## Gate Activity", ""]
    for split, m in metrics.items():
        gate = m.get("gate", {})
        if not gate:
            continue
        lines += [
            f"### {split}",
            "",
            f"- gate active rate: `{fmt(gate.get('gate_active_rate', 0.0))}`",
            f"- active gold c7 rate: `{fmt(gate.get('active_gold_c7_rate', 0.0))}`",
            f"- inactive gold c7 rate: `{fmt(gate.get('inactive_gold_c7_rate', 0.0))}`",
            f"- active gated c7 top1 rate: `{fmt(gate.get('active_gated_c7_top1_rate', 0.0))}`",
            f"- inactive gated c7 top1 rate: `{fmt(gate.get('inactive_gated_c7_top1_rate', 0.0))}`",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether a c7-gated predicted-control prepared set improves c7 usage without excessive false c7.")
    parser.add_argument("--prepared", required=True, help="Original gold prepared directory.")
    parser.add_argument("--stance-dir", required=True, help="Ungated stance prediction directory with *_predictions.npz.")
    parser.add_argument("--gated-prepared", required=True, help="Prepared directory produced by apply_c7_gate_prepared.")
    parser.add_argument("--out", required=True, help="Output directory for metrics.json and report.md.")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--c7-cluster", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(args.prepared)
    stance_dir = Path(args.stance_dir)
    gated_prepared = Path(args.gated_prepared)

    metrics: Dict[str, Any] = {}
    for split in parse_splits(args.splits):
        metrics[split] = evaluate_split(prepared, stance_dir, gated_prepared, split, args.c7_cluster)

    summary = {
        "prepared": str(prepared),
        "stance_dir": str(stance_dir),
        "gated_prepared": str(gated_prepared),
        "c7_cluster": args.c7_cluster,
        "splits": parse_splits(args.splits),
        "metrics": metrics,
    }
    write_json(out / "metrics.json", summary)
    write_report(out / "report.md", metrics, args)


if __name__ == "__main__":
    main()
