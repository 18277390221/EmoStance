from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List

from ablation_utils import append_command, write_json


def add_common(cmd: List[str], args: argparse.Namespace, seed: int) -> List[str]:
    cmd.extend([
        "--prepared", "runs/main/prepared",
        "--model", args.model,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--focal-gamma", str(args.focal_gamma),
        "--class-weight-power", str(args.class_weight_power),
    ])
    if args.max_train_examples > 0:
        cmd.extend(["--max-train-examples", str(args.max_train_examples)])
    if args.max_eval_examples > 0:
        cmd.extend(["--max-eval-examples", str(args.max_eval_examples)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run missing stance-prediction ablations.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 21, 42])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--execute", action="store_true", help="Actually run the training jobs.")
    parser.add_argument("--model", default="deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_command(out_dir, "python scripts/run_stance_ablation.py " + " ".join(__import__('sys').argv[1:]))

    specs = [
        {
            "ablation_id": "A2",
            "method": "w/o role-aware transition",
            "dir_prefix": "no_role_transition",
            "extra": ["--disable-role-features", "--graph-prior-weight", "0.0"],
            "notes": "Zeroes current-role, next-role, and transition embeddings and disables graph prior.",
        },
        {
            "ablation_id": "A3",
            "method": "w/o gated transition prior",
            "dir_prefix": "no_gated_prior",
            "extra": ["--graph-prior-weight", "0.0"],
            "notes": "Keeps role features but removes the gated log-prior by setting graph_prior_weight=0.",
        },
        {
            "ablation_id": "A4",
            "method": "hard-label training",
            "dir_prefix": "hard_label",
            "extra": ["--hard-label-training"],
            "notes": "Uses argmax hard cluster labels for source/target CE losses.",
        },
    ]
    planned = []
    for spec in specs:
        for seed in args.seeds:
            out = out_dir / f"{spec['dir_prefix']}_seed{seed}"
            cmd = [".venv/bin/python", "-m", "latent_stance_control.train_role_aware_stance_predictor", "--out", str(out)]
            add_common(cmd, args, seed)
            cmd.extend(spec["extra"])
            planned.append({
                "ablation_id": spec["ablation_id"],
                "method": spec["method"],
                "seed": seed,
                "status": "planned" if not args.execute else "running",
                "command": cmd,
                "notes": spec["notes"],
            })

    if args.execute:
        for job in planned:
            append_command(out_dir, " ".join(job["command"]))
            subprocess.run(job["command"], check=True)
            job["status"] = "completed"
    write_json(out_dir / "stance_ablation_plan.json", planned)


if __name__ == "__main__":
    main()
