from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate system metrics and run paired bootstrap significance.")
    parser.add_argument("--metrics_dir", default="system_baseline/metrics")
    parser.add_argument("--output_dir", default="system_baseline/metrics")
    parser.add_argument("--generations_dir", default="system_baseline/generations")
    parser.add_argument("--config", default="system_baseline/configs/system_eval.yaml")
    args = parser.parse_args()

    root = Path(".").resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "system_baseline/scripts/paired_bootstrap.py",
        "--metrics_dir",
        args.metrics_dir,
        "--generations_dir",
        args.generations_dir,
        "--config",
        args.config,
        "--output_csv",
        str(output_dir / "pairwise_significance.csv"),
        "--output_json",
        str(output_dir / "pairwise_significance.json"),
    ]
    subprocess.run(cmd, cwd=root, check=True)
    print(f"Aggregated metrics in {output_dir}")


if __name__ == "__main__":
    main()
