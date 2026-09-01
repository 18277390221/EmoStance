from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import load_prepared_split, read_json, write_json, write_jsonl
from .metrics import normalize_prob
from .train_c7_gate import load_gate_model, predict_gate


def parse_splits(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def adjust_c7_distribution(
    pred_cluster: np.ndarray,
    gate_prob: np.ndarray,
    c7_cluster: int,
    threshold: float,
    boost: float,
    low_cap: float,
    high_floor: float,
) -> np.ndarray:
    q = normalize_prob(pred_cluster).astype(np.float64)
    g = np.asarray(gate_prob, dtype=np.float64).reshape(-1)
    if q.shape[0] != g.shape[0]:
        raise ValueError(f"Prediction/gate length mismatch: {q.shape[0]} vs {g.shape[0]}")
    out = q.copy()
    for i, prob in enumerate(g):
        if prob >= threshold:
            out[i, c7_cluster] = max(out[i, c7_cluster] * (1.0 + boost * prob), high_floor)
        else:
            out[i, c7_cluster] = min(out[i, c7_cluster], low_cap)
    out = normalize_prob(out)
    return out.astype(np.float64)


def mix_active_prototype_vectors(
    vectors: np.ndarray,
    gate_prob: np.ndarray,
    prototype: np.ndarray,
    threshold: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"--active-prototype-mix must be in [0, 1], got {gamma}")
    active = np.asarray(gate_prob, dtype=np.float64).reshape(-1) >= threshold
    out = np.asarray(vectors, dtype=np.float64).copy()
    if gamma <= 0.0 or not active.any():
        return out, active, 0.0
    before = out[active].copy()
    proto = np.asarray(prototype, dtype=np.float64).reshape(1, -1)
    out[active] = (1.0 - gamma) * out[active] + gamma * proto
    mean_shift = float(np.linalg.norm(out[active] - before, axis=1).mean())
    return out, active, mean_shift


def replace_split(
    rows: List[dict],
    predictions_path: Path,
    gate_model,
    gate_tokenizer,
    gate_config: Dict,
    prototypes: np.ndarray,
    args,
) -> tuple[List[dict], Dict]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing stance predictions: {predictions_path}")
    with np.load(predictions_path) as data:
        pred_cluster = np.asarray(data["pred_target_cluster"], dtype=np.float64)
    if pred_cluster.shape[0] != len(rows):
        raise ValueError(f"Prediction/row count mismatch for {predictions_path}: {pred_cluster.shape[0]} vs {len(rows)}")
    if pred_cluster.shape[1] != prototypes.shape[0]:
        raise ValueError(f"Prediction/prototype cluster mismatch: {pred_cluster.shape[1]} vs {prototypes.shape[0]}")
    gate_prob = predict_gate(gate_model, gate_tokenizer, rows, int(gate_config.get("max_length", args.score_max_length)), args.batch_size, args.device)
    threshold = args.threshold if args.threshold is not None else float(gate_config.get("best_threshold", 0.5))
    ungated = normalize_prob(pred_cluster)
    gated = adjust_c7_distribution(
        ungated,
        gate_prob,
        args.c7_cluster,
        threshold=threshold,
        boost=args.boost,
        low_cap=args.low_cap,
        high_floor=args.high_floor,
    )
    base_vectors = gated @ prototypes
    vectors, active_mask, mean_vector_mix_l2 = mix_active_prototype_vectors(
        base_vectors,
        gate_prob,
        prototypes[args.c7_cluster],
        threshold=threshold,
        gamma=args.active_prototype_mix,
    )
    ungated_vectors = ungated @ prototypes
    out_rows: List[dict] = []
    for row, q0, q1, z0, z_base, z1, gp, active in zip(rows, ungated, gated, ungated_vectors, base_vectors, vectors, gate_prob, active_mask):
        new_row = copy.deepcopy(row)
        new_row["gold_target_cluster"] = row.get("target_cluster")
        new_row["gold_target_vector"] = row.get("target_vector")
        new_row["ungated_target_cluster"] = q0.astype(float).tolist()
        new_row["ungated_target_vector"] = z0.astype(float).tolist()
        new_row["target_cluster"] = q1.astype(float).tolist()
        new_row["target_vector"] = z1.astype(float).tolist()
        if args.active_prototype_mix > 0.0:
            new_row["gated_target_vector_before_mix"] = z_base.astype(float).tolist()
        new_row["active_prototype_mix"] = float(args.active_prototype_mix)
        new_row["active_prototype_mix_applied"] = int(bool(active) and args.active_prototype_mix > 0.0)
        new_row["c7_gate_prob"] = float(gp)
        new_row["c7_gate_threshold"] = float(threshold)
        new_row["c7_gate_pred"] = int(float(gp) >= threshold)
        new_row["control_type"] = "c7_gated_predicted_cluster_prototype_mix" if args.active_prototype_mix > 0.0 else "c7_gated_predicted_cluster_prototype"
        out_rows.append(new_row)
    summary = {
        "rows": len(rows),
        "prediction_file": str(predictions_path),
        "threshold": threshold,
        "mean_gate_prob": float(np.mean(gate_prob)) if len(gate_prob) else 0.0,
        "gate_positive_rate": float(np.mean(gate_prob >= threshold)) if len(gate_prob) else 0.0,
        "mean_ungated_c7_prob": float(np.mean(ungated[:, args.c7_cluster])) if len(ungated) else 0.0,
        "mean_gated_c7_prob": float(np.mean(gated[:, args.c7_cluster])) if len(gated) else 0.0,
        "ungated_c7_top1_rate": float(np.mean(np.argmax(ungated, axis=1) == args.c7_cluster)) if len(ungated) else 0.0,
        "gated_c7_top1_rate": float(np.mean(np.argmax(gated, axis=1) == args.c7_cluster)) if len(gated) else 0.0,
        "active_prototype_mix": float(args.active_prototype_mix),
        "active_prototype_mix_rows": int(active_mask.sum()) if len(active_mask) else 0,
        "mean_active_vector_mix_l2": float(mean_vector_mix_l2),
    }
    return out_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a predicted-control prepared directory with a c7 suitability gate applied to target cluster distributions.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--stance-dir", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--cluster-prototypes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--train-mode", choices=["copy_gold", "skip"], default="copy_gold")
    parser.add_argument("--c7-cluster", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=None, help="Override gate threshold. Defaults to gate config best_threshold.")
    parser.add_argument("--boost", type=float, default=1.0)
    parser.add_argument("--low-cap", type=float, default=0.02)
    parser.add_argument("--high-floor", type=float, default=0.0, help="Optional minimum c7 prob when gate is active before renormalization.")
    parser.add_argument("--active-prototype-mix", type=float, default=0.0, help="When gate is active, mix target_vector toward the c7 prototype by this gamma in [0, 1]. target_cluster is unchanged.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-max-length", type=int, default=320)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.active_prototype_mix < 0.0 or args.active_prototype_mix > 1.0:
        raise ValueError(f"--active-prototype-mix must be in [0, 1], got {args.active_prototype_mix}")

    prepared = Path(args.prepared)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stance_dir = Path(args.stance_dir)
    prototypes = np.asarray(read_json(args.cluster_prototypes), dtype=np.float64)
    if args.c7_cluster < 0 or args.c7_cluster >= prototypes.shape[0]:
        raise ValueError(f"c7 cluster {args.c7_cluster} is outside prototype matrix with {prototypes.shape[0]} rows")
    gate_model, gate_tokenizer, gate_config = load_gate_model(args.gate_dir, args.device)
    splits = parse_splits(args.splits)

    control_type = "c7_gated_predicted_cluster_prototype_mix" if args.active_prototype_mix > 0.0 else "c7_gated_predicted_cluster_prototype"
    meta = read_json(prepared / "meta.json")
    meta["control_type"] = control_type
    meta["control_source"] = {
        "stance_dir": str(stance_dir),
        "gate_dir": str(args.gate_dir),
        "cluster_prototypes": str(args.cluster_prototypes),
        "replaced_splits": splits,
        "train_mode": args.train_mode,
        "c7_cluster": args.c7_cluster,
        "boost": args.boost,
        "low_cap": args.low_cap,
        "high_floor": args.high_floor,
        "active_prototype_mix": args.active_prototype_mix,
        "threshold": args.threshold if args.threshold is not None else gate_config.get("best_threshold", 0.5),
    }
    write_json(out / "meta.json", meta)

    summaries: Dict[str, Dict] = {}
    if args.train_mode == "copy_gold":
        train_rows = load_prepared_split(prepared, "train")
        for row in train_rows:
            row["control_type"] = "gold_target_vector_copied_train"
        write_jsonl(out / "train.jsonl", train_rows)
        summaries["train"] = {"rows": len(train_rows), "mode": "copy_gold"}

    for split in splits:
        rows = load_prepared_split(prepared, split)
        if not rows:
            continue
        out_rows, summary = replace_split(rows, stance_dir / f"{split}_predictions.npz", gate_model, gate_tokenizer, gate_config, prototypes, args)
        write_jsonl(out / f"{split}.jsonl", out_rows)
        summaries[split] = summary

    write_json(
        out / "prepare_summary.json",
        {
            "source_prepared": str(prepared),
            "out": str(out),
            "control_type": control_type,
            "splits": summaries,
            "num_clusters": int(prototypes.shape[0]),
            "vector_dim": int(prototypes.shape[1]),
        },
    )


if __name__ == "__main__":
    main()
