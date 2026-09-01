from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ablation_utils import npz_arrays, sanitize, vector_cosine, vector_mse, write_json


def norm_distribution(gold: np.ndarray, pred: np.ndarray) -> dict:
    gold_norm = np.linalg.norm(np.nan_to_num(gold, nan=0.0), axis=-1)
    pred_norm = np.linalg.norm(np.nan_to_num(pred, nan=0.0), axis=-1)
    return {
        "gold_norm_mean": float(gold_norm.mean()),
        "gold_norm_std": float(gold_norm.std()),
        "pred_norm_mean": float(pred_norm.mean()),
        "pred_norm_std": float(pred_norm.std()),
        "pred_norm_p05": float(np.percentile(pred_norm, 5)),
        "pred_norm_p50": float(np.percentile(pred_norm, 50)),
        "pred_norm_p95": float(np.percentile(pred_norm, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute continuous vector metrics from a predictions NPZ file.")
    parser.add_argument("--predictions", required=True, help="NPZ containing gold/pred vector arrays.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix", default="target", choices=["target", "source"])
    args = parser.parse_args()

    arrays = npz_arrays(args.predictions)
    gold_key = f"gold_{args.prefix}_vector"
    pred_key = f"pred_{args.prefix}_vector"
    if gold_key not in arrays or pred_key not in arrays:
        raise KeyError(f"{args.predictions} must contain {gold_key} and {pred_key}.")
    metrics = {
        "vector_cosine": vector_cosine(arrays[gold_key], arrays[pred_key]),
        "vector_mse": vector_mse(arrays[gold_key], arrays[pred_key]),
        "norm_distribution": norm_distribution(arrays[gold_key], arrays[pred_key]),
    }
    write_json(args.out, sanitize({"predictions": str(Path(args.predictions)), "prefix": args.prefix, "metrics": metrics}))


if __name__ == "__main__":
    main()
