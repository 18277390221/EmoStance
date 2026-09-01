from __future__ import annotations

import argparse
from pathlib import Path

from ablation_utils import append_command, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect/reuse generation-control ablation artifacts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42])
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_command(out_dir, f"python scripts/run_generation_control_ablation.py --config {args.config} --seeds {' '.join(map(str, args.seeds))} --out_dir {args.out_dir}")
    payload = {
        "status": "reused_existing_512_example_artifacts",
        "source_artifacts": [
            "runs/main/main_results_summary/metrics.json",
            "runs/main/generator_control_eval_role_weight025_c7mix050_compare/metrics.json",
            "runs/main/rerank_c7mix050_512_seed13/metrics.json",
            "runs/main/rerank_c7mix050_512_seed21/metrics.json",
            "runs/main/rerank_c7mix050_512_seed42/metrics.json",
        ],
        "metrics": read_json("runs/main/main_results_summary/metrics.json"),
    }
    write_json(out_dir / "generation_ablation_reuse.json", payload)


if __name__ == "__main__":
    main()
