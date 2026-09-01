from __future__ import annotations

import argparse
from pathlib import Path

from ablation_utils import cluster_metrics, npz_arrays, sanitize, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stance prediction metrics from a predictions NPZ file.")
    parser.add_argument("--predictions", required=True, help="NPZ containing gold_target_cluster and pred_target_cluster arrays.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument("--prefix", default="target", choices=["target", "source"])
    args = parser.parse_args()

    arrays = npz_arrays(args.predictions)
    gold_key = f"gold_{args.prefix}_cluster"
    pred_key = f"pred_{args.prefix}_cluster"
    if gold_key not in arrays or pred_key not in arrays:
        raise KeyError(f"{args.predictions} must contain {gold_key} and {pred_key}.")
    metrics = cluster_metrics(arrays[gold_key], arrays[pred_key])
    write_json(args.out, sanitize({"predictions": str(Path(args.predictions)), "prefix": args.prefix, "metrics": metrics}))


if __name__ == "__main__":
    main()
