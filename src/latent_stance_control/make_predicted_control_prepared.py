from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import load_prepared_split, read_json, write_json, write_jsonl


def parse_splits(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def replace_split_vectors(rows: List[dict], predictions_path: Path, prototypes: np.ndarray) -> tuple[List[dict], Dict]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    with np.load(predictions_path) as data:
        pred_cluster = np.asarray(data["pred_target_cluster"], dtype=np.float64)
    if pred_cluster.shape[0] != len(rows):
        raise ValueError(
            f"Prediction/row count mismatch for {predictions_path}: "
            f"{pred_cluster.shape[0]} predictions vs {len(rows)} rows"
        )
    if pred_cluster.shape[1] != prototypes.shape[0]:
        raise ValueError(
            f"Cluster dim mismatch: predictions have {pred_cluster.shape[1]}, "
            f"prototypes have {prototypes.shape[0]}"
        )
    pred_cluster = np.clip(np.nan_to_num(pred_cluster, nan=0.0), 0.0, None)
    pred_cluster /= np.maximum(pred_cluster.sum(axis=1, keepdims=True), 1e-12)
    pred_vectors = pred_cluster @ prototypes

    output_rows: List[dict] = []
    for row, q, z in zip(rows, pred_cluster, pred_vectors):
        new_row = copy.deepcopy(row)
        new_row["gold_target_cluster"] = row.get("target_cluster")
        new_row["gold_target_vector"] = row.get("target_vector")
        new_row["target_cluster"] = q.astype(float).tolist()
        new_row["target_vector"] = z.astype(float).tolist()
        new_row["control_type"] = "predicted_cluster_prototype"
        output_rows.append(new_row)
    summary = {
        "rows": len(rows),
        "prediction_file": str(predictions_path),
        "mean_pred_entropy": float((-(pred_cluster * np.log(pred_cluster + 1e-12)).sum(axis=1)).mean()),
    }
    return output_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a prepared dataset whose target_vector is predicted target cluster @ cluster prototypes.")
    parser.add_argument("--prepared", required=True, help="Original prepared directory with gold target vectors.")
    parser.add_argument("--stance-dir", required=True, help="Directory containing dev/test *_predictions.npz from a trained stance predictor.")
    parser.add_argument("--cluster-prototypes", required=True, help="cluster_prototypes.json from run_ablations.")
    parser.add_argument("--out", required=True, help="Output prepared directory.")
    parser.add_argument("--splits", default="dev,test", help="Comma-separated splits to replace using predictions.")
    parser.add_argument(
        "--train-mode",
        choices=["copy_gold", "skip"],
        default="copy_gold",
        help="What to do with train split when train predictions are unavailable. copy_gold keeps train usable for smoke runs; comparison should use dev/test.",
    )
    args = parser.parse_args()

    prepared = Path(args.prepared)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stance_dir = Path(args.stance_dir)
    prototypes = np.asarray(read_json(args.cluster_prototypes), dtype=np.float64)
    splits = parse_splits(args.splits)

    meta = read_json(prepared / "meta.json")
    meta["control_type"] = "predicted_cluster_prototype"
    meta["control_source"] = {
        "stance_dir": str(stance_dir),
        "cluster_prototypes": str(args.cluster_prototypes),
        "replaced_splits": splits,
        "train_mode": args.train_mode,
    }
    write_json(out / "meta.json", meta)

    split_summaries: Dict[str, Dict] = {}
    if args.train_mode == "copy_gold":
        train_rows = load_prepared_split(prepared, "train")
        for row in train_rows:
            row["control_type"] = "gold_target_vector_copied_train"
        write_jsonl(out / "train.jsonl", train_rows)
        split_summaries["train"] = {"rows": len(train_rows), "mode": "copy_gold"}

    for split in splits:
        rows = load_prepared_split(prepared, split)
        if not rows:
            continue
        output_rows, summary = replace_split_vectors(rows, stance_dir / f"{split}_predictions.npz", prototypes)
        write_jsonl(out / f"{split}.jsonl", output_rows)
        split_summaries[split] = summary

    write_json(
        out / "prepare_summary.json",
        {
            "source_prepared": str(prepared),
            "out": str(out),
            "control_type": "predicted_cluster_prototype",
            "splits": split_summaries,
            "num_clusters": int(prototypes.shape[0]),
            "vector_dim": int(prototypes.shape[1]),
        },
    )


if __name__ == "__main__":
    main()
