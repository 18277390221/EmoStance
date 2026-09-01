from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audit_lib import ExperimentError, write_analysis_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LLM-vs-human emoji distribution agreement.")
    parser.add_argument("--experiment-dir", default="human_llm_emoji_audit")
    parser.add_argument("--exports-dir", default="human_llm_emoji_audit/exports")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = write_analysis_outputs(
            Path(args.experiment_dir),
            Path(args.exports_dir),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
    except ExperimentError as exc:
        print(f"analyze_agreement: FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    print("analyze_agreement: PASS")
    print(f"emoji_jsd_mean={summary['overall']['emoji_jsd_mean']:.6f}")
    print(f"region_jsd_mean={summary['overall']['region_jsd_mean']:.6f}")


if __name__ == "__main__":
    main()
