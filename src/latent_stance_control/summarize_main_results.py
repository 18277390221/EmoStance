from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .data import write_json


DEFAULT_CONTROL_DIRS = [
    "runs/main/generator_control_seed13",
    "runs/main/generator_control_seed21",
    "runs/main/generator_control_seed42",
]
DEFAULT_COMPARISON = "runs/main/generator_control_compare"
DEFAULT_RERANK_DIRS = [
    "runs/main/rerank_seed13",
    "runs/main/rerank_seed21",
    "runs/main/rerank_seed42",
]
DEFAULT_HYBRID = "runs/main/hybrid_pairwise_reranker_eval"


METRICS = ("vs_gold_ce", "vs_gold_acc", "vs_gold_macro_f1", "vs_control_ce", "mean_words", "empty_rate")
SPLITS = ("dev", "test")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_summary(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    clean = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not clean:
        return {"mean": None, "std": None, "values": []}
    arr = np.asarray(clean, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "values": arr.astype(float).tolist()}


def empty_row(method: str, source: str) -> Dict[str, Any]:
    return {"method": method, "source": source, "splits": {split: {} for split in SPLITS}}


def from_metric_block(block: Dict[str, Any]) -> Dict[str, Optional[float]]:
    vs_gold = block.get("vs_gold", {})
    vs_control = block.get("vs_control", {})
    return {
        "vs_gold_ce": vs_gold.get("soft_ce"),
        "vs_gold_acc": vs_gold.get("accuracy"),
        "vs_gold_macro_f1": vs_gold.get("macro_f1"),
        "vs_control_ce": vs_control.get("soft_ce"),
        "mean_words": block.get("mean_generated_words"),
        "empty_rate": block.get("empty_rate"),
    }


def aggregate_control(control_dirs: List[str], control_type: str, method: str) -> Dict[str, Any]:
    row = empty_row(method, source="generator_control_eval")
    for split in SPLITS:
        values: Dict[str, List[Optional[float]]] = {name: [] for name in METRICS}
        for run_dir in control_dirs:
            path = Path(run_dir) / "metrics.json"
            if not path.exists():
                continue
            metrics = read_json(path).get("metrics", {})
            block = metrics.get(split, {}).get(control_type)
            if not block:
                continue
            flat = from_metric_block(block)
            for name in METRICS:
                values[name].append(flat.get(name))
        row["splits"][split] = {name: metric_summary(values[name]) for name in METRICS}
    return row


def aggregate_comparison(comparison_dir: str, key: str, method: str) -> Dict[str, Any]:
    row = empty_row(method, source="role_weight025_c7mix050_compare")
    path = Path(comparison_dir) / "metrics.json"
    data = read_json(path)
    for split in SPLITS:
        split_data = data.get(key, {}).get(split, {})
        row["splits"][split] = {}
        for name in METRICS:
            metric_key = "mean_words" if name == "mean_words" else name
            row["splits"][split][name] = split_data.get(metric_key, {"mean": None, "std": None, "values": []})
    return row


def aggregate_rerank(rerank_dirs: List[str], selection_type: str, method: str) -> Dict[str, Any]:
    row = empty_row(method, source="generate_and_rerank")
    for split in SPLITS:
        values: Dict[str, List[Optional[float]]] = {name: [] for name in METRICS}
        for run_dir in rerank_dirs:
            path = Path(run_dir) / "metrics.json"
            if not path.exists():
                continue
            metrics = read_json(path).get("metrics", {})
            block = metrics.get(split, {}).get("selections", {}).get(selection_type)
            if not block:
                continue
            flat = from_metric_block(block)
            for name in METRICS:
                values[name].append(flat.get(name))
        row["splits"][split] = {name: metric_summary(values[name]) for name in METRICS}
    return row


def maybe_aggregate_hybrid(hybrid_dir: str) -> Optional[Dict[str, Any]]:
    path = Path(hybrid_dir) / "metrics.json"
    if not path.exists():
        return None
    data = read_json(path)
    row = empty_row("hybrid pairwise reranker best-lambda", source="hybrid_pairwise_reranker")
    for split in SPLITS:
        selections = data.get("summary", {}).get(split, {}).get("selections", {})
        best_name = None
        best_ce = None
        for name, block in selections.items():
            if not name.startswith("hybrid_"):
                continue
            ce = block.get("vs_gold_soft_ce_mean")
            if ce is None:
                continue
            if best_ce is None or float(ce) < best_ce:
                best_ce = float(ce)
                best_name = name
        if not best_name:
            continue
        block = selections[best_name]
        row["splits"][split] = {
            "selected_hybrid": best_name,
            "vs_gold_ce": {"mean": block.get("vs_gold_soft_ce_mean"), "std": block.get("vs_gold_soft_ce_std"), "values": []},
            "vs_gold_acc": {"mean": block.get("vs_gold_accuracy_mean"), "std": block.get("vs_gold_accuracy_std"), "values": []},
            "vs_gold_macro_f1": {"mean": block.get("vs_gold_macro_f1_mean"), "std": block.get("vs_gold_macro_f1_std"), "values": []},
            "vs_control_ce": {"mean": block.get("vs_control_soft_ce_mean"), "std": block.get("vs_control_soft_ce_std"), "values": []},
            "mean_words": {"mean": None, "std": None, "values": []},
            "empty_rate": {"mean": None, "std": None, "values": []},
        }
    return row


def fmt(summary: Dict[str, Any]) -> str:
    mean = summary.get("mean")
    std = summary.get("std")
    if mean is None:
        return "NA"
    if std is None:
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f}±{float(std):.4f}"


def write_report(path: Path, rows: List[Dict[str, Any]], args) -> None:
    lines = [
        "# Main Results Summary",
        "",
        "This table aggregates the current experiments.",
        "",
        f"Control dirs: `{', '.join(args.control_dirs)}`",
        f"Comparison dir: `{args.comparison_dir}`",
        f"Rerank dirs: `{', '.join(args.rerank_dirs)}`",
        f"Hybrid dir: `{args.hybrid_dir}`",
        "",
    ]
    for split in SPLITS:
        lines += [
            f"## {split}",
            "",
            "| method | vs_gold CE | acc | macro-F1 | vs_control CE | mean words | empty rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            data = row["splits"].get(split, {})
            lines.append(
                f"| {row['method']} | {fmt(data.get('vs_gold_ce', {}))} | {fmt(data.get('vs_gold_acc', {}))} | "
                f"{fmt(data.get('vs_gold_macro_f1', {}))} | {fmt(data.get('vs_control_ce', {}))} | "
                f"{fmt(data.get('mean_words', {}))} | {fmt(data.get('empty_rate', {}))} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- `gold control` uses the gold latent stance vector and is an upper-reference control, not a deployable setting.",
        "- `oracle_gold` selects the best generated candidate using gold stance CE; it is an upper bound for candidate selection.",
        "- `role+c7mix050 + rerank_control` is the current deployable generation setting.",
        "- `hybrid pairwise reranker best-lambda` is included as a diagnostic; it is not currently the main result because its best lambda is not stable across splits.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize main latent stance control experiments into a single report.")
    parser.add_argument("--control-dirs", nargs="+", default=DEFAULT_CONTROL_DIRS)
    parser.add_argument("--comparison-dir", default=DEFAULT_COMPARISON)
    parser.add_argument("--rerank-dirs", nargs="+", default=DEFAULT_RERANK_DIRS)
    parser.add_argument("--hybrid-dir", default=DEFAULT_HYBRID)
    parser.add_argument("--out", default="runs/main/main_results_summary")
    args = parser.parse_args()

    rows = [
        aggregate_control(args.control_dirs, "zero", "zero control"),
        aggregate_control(args.control_dirs, "shuffled", "shuffled control"),
        aggregate_control(args.control_dirs, "gold", "gold control"),
        aggregate_comparison(args.comparison_dir, "baseline", "predicted control baseline"),
        aggregate_comparison(args.comparison_dir, "role_weight025", "role_weight025"),
        aggregate_comparison(args.comparison_dir, "role_weight025_c7mix050", "role+c7mix050"),
        aggregate_rerank(args.rerank_dirs, "rerank_control", "role+c7mix050 + rerank_control"),
        aggregate_rerank(args.rerank_dirs, "oracle_gold", "oracle_gold candidate selection"),
    ]
    hybrid = maybe_aggregate_hybrid(args.hybrid_dir)
    if hybrid:
        rows.append(hybrid)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "metrics.json",
        {
            "control_dirs": args.control_dirs,
            "comparison_dir": args.comparison_dir,
            "rerank_dirs": args.rerank_dirs,
            "hybrid_dir": args.hybrid_dir,
            "rows": rows,
        },
    )
    write_report(out / "report.md", rows, args)


if __name__ == "__main__":
    main()
