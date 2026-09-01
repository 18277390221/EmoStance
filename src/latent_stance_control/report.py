from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .data import read_json


def maybe_json(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def metrics_table(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    lines: List[str] = []
    for split, split_metrics in metrics.items():
        lines.append(f"## {split}")
        lines.append("")
        lines.append("| ablation | soft_ce | kl | acc | macro_f1 | ece | vector_mse | vector_cosine | beta |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, m in split_metrics.items():
            lines.append("| " + " | ".join([name, fmt(m.get("soft_ce")), fmt(m.get("kl")), fmt(m.get("accuracy")), fmt(m.get("macro_f1")), fmt(m.get("ece")), fmt(m.get("vector_mse")), fmt(m.get("vector_cosine")), fmt(m.get("beta"))]) + " |")
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Markdown report for a latent stance run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    prepare_summary = maybe_json(run_dir / "prepared" / "prepare_summary.json")
    stance_dir = next((run_dir / name for name in ("stance_role_aware", "stance_fp32", "stance") if (run_dir / name).exists()), run_dir / "stance")
    ablations_dir = next((run_dir / name for name in ("ablations_role_aware", "ablations") if (run_dir / name).exists()), run_dir / "ablations")
    generator_dir = next((run_dir / name for name in ("generator_gold", "generator") if (run_dir / name).exists()), run_dir / "generator")
    stance_metrics = maybe_json(stance_dir / "metrics.json")
    ablation_metrics = maybe_json(ablations_dir / "metrics.json")
    generator_history = maybe_json(generator_dir / "loss_history.json")
    lines: List[str] = [
        "# Latent Stance Control Report",
        "",
        "## Summary",
        "",
        "This run evaluates a text-only hierarchical latent stance control model. Emoji annotations are used as weak supervision during training only; inference uses dialogue text.",
        "",
        f"Stance metrics source: `{stance_dir.name}`.",
        "",
    ]
    if prepare_summary:
        lines += [
            "## Prepared Data",
            "",
            f"- examples: {prepare_summary.get('num_examples')}",
            f"- splits: {prepare_summary.get('splits')}",
            f"- emojis: {prepare_summary.get('num_emojis')}",
            f"- clusters: {prepare_summary.get('num_clusters')}",
            f"- vector_dim: {prepare_summary.get('vector_dim')}",
            "",
        ]
    if stance_metrics:
        lines += ["## DeBERTa Stance Predictor", ""]
        for split, metrics in stance_metrics.items():
            lines.append(f"### {split}")
            lines.append("")
            for key, value in metrics.items():
                lines.append(f"- {key}: {fmt(value)}")
            lines.append("")
    if ablation_metrics:
        lines += ["## Ablations", ""] + metrics_table(ablation_metrics)
    if generator_history:
        lines += ["## Mistral Internal Control Generator", ""]
        for item in generator_history:
            lines.append(f"- epoch {item.get('epoch')}: train_lm_loss={fmt(item.get('train_lm_loss'))}")
        lines.append("")
    lines += [
        "## Interpretation Checklist",
        "",
        "- If `text_graph_calibrated` improves soft CE/KL without hurting macro-F1 much, the graph is acting as a useful calibration prior.",
        "- If `graph_only` has weak macro-F1 but reasonable soft CE, the transition graph captures global soft tendencies rather than sample-level decisions.",
        "- If vector cosine improves with operator variants, continuous stance transitions add information beyond cluster id.",
        "- If gold-control generation is much better than predicted-control generation, the generation interface works and the bottleneck is stance prediction.",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
