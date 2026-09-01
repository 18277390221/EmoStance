from __future__ import annotations

import argparse
from pathlib import Path

from ablation_utils import append_command, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect/reuse continuous-vector ablation artifacts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_command(out_dir, f"python scripts/run_vector_ablation.py --config {args.config} --seeds {' '.join(map(str, args.seeds))} --out_dir {args.out_dir}")
    payload = {
        "status": "reused_existing_artifacts",
        "source_artifacts": [
            "runs/main/ablations/metrics.json",
            "runs/main/source_vector_feature_ablation/metrics.json",
        ],
        "metrics": {
            "main_vector": read_json("runs/main/ablations/metrics.json")["test"],
            "source_vector_feature": read_json("runs/main/source_vector_feature_ablation/metrics.json"),
        },
    }
    write_json(out_dir / "vector_ablation_reuse.json", payload)


if __name__ == "__main__":
    main()
