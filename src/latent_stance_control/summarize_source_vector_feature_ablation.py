from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .data import read_json, write_json


KEYS = (
    "source_soft_ce",
    "source_accuracy",
    "source_macro_f1",
    "source_vector_cosine",
    "target_soft_ce",
    "target_accuracy",
    "target_macro_f1",
    "target_vector_cosine",
    "mean_graph_gate",
)


def parse_item(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(value)
    return path.name, path


def load_run(name: str, path: Path) -> Dict[str, Any]:
    metrics_path = path / "metrics.json"
    config_path = path / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file for {name}: {metrics_path}")
    metrics = read_json(metrics_path)
    config = read_json(config_path) if config_path.exists() else {}
    return {
        "name": name,
        "path": str(path),
        "source_vector_feature_mode": config.get("source_vector_feature_mode", "unknown"),
        "best_epoch": config.get("best_epoch"),
        "best_dev_target_soft_ce": config.get("best_dev_target_soft_ce"),
        "metrics": metrics,
    }


def metric_value(run: Dict[str, Any], split: str, key: str) -> float | None:
    value = run.get("metrics", {}).get(split, {}).get(key)
    return float(value) if value is not None else None


def fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def write_report(path: Path, runs: List[Dict[str, Any]], splits: List[str]) -> None:
    lines = [
        "# Source Vector Feature Ablation",
        "",
        "This report compares how source-vector information enters the role-aware target predictor.",
        "",
        "| run | mode | best epoch | best dev target CE |",
        "|---|---|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['name']} | {run['source_vector_feature_mode']} | "
            f"{run.get('best_epoch') or 'NA'} | {fmt(run.get('best_dev_target_soft_ce'))} |"
        )

    for split in splits:
        lines += [
            "",
            f"## {split}",
            "",
            "| run | target CE | target acc | target macro-F1 | target vec cos | source CE | source acc | source macro-F1 | source vec cos | graph gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for run in runs:
            lines.append(
                f"| {run['name']} | "
                f"{fmt(metric_value(run, split, 'target_soft_ce'))} | "
                f"{fmt(metric_value(run, split, 'target_accuracy'))} | "
                f"{fmt(metric_value(run, split, 'target_macro_f1'))} | "
                f"{fmt(metric_value(run, split, 'target_vector_cosine'))} | "
                f"{fmt(metric_value(run, split, 'source_soft_ce'))} | "
                f"{fmt(metric_value(run, split, 'source_accuracy'))} | "
                f"{fmt(metric_value(run, split, 'source_macro_f1'))} | "
                f"{fmt(metric_value(run, split, 'source_vector_cosine'))} | "
                f"{fmt(metric_value(run, split, 'mean_graph_gate'))} |"
            )

    lines += [
        "",
        "Interpretation guide:",
        "",
        "- Lower `target CE` is better for soft stance prediction.",
        "- Higher `target macro-F1` is better for small-cluster hard decisions.",
        "- Higher `target vec cos` means the generated control vector is closer to gold stance vector.",
        "- If `none` beats `direct`, the direct source-vector head was likely adding noise to target prediction.",
        "- If `prototype` beats both, source cluster is useful but direct source-vector regression is too noisy.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize source-vector feature ablation runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run specs as name=path or plain path. Each path must contain metrics.json.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", default="dev,test")
    args = parser.parse_args()

    runs = [load_run(name, path) for name, path in (parse_item(item) for item in args.runs)]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", {"runs": runs, "splits": splits})
    write_report(out / "report.md", runs, splits)


if __name__ == "__main__":
    main()
