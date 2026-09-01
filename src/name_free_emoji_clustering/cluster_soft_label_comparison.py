from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .cluster_stability_analysis import (
    AggregateRow,
    build_representation_metrics,
    default_output_dir,
    discover_inputs,
    draw_entropy_chart,
    load_four_model_labels,
    load_membership,
    load_soft_label_table,
)


REPRESENTATION_ORDER = {"emoji": 0, "cluster_raw": 1, "cluster_sharp": 2}


def sorted_rows(rows: list[AggregateRow]) -> list[AggregateRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.group_value,
            REPRESENTATION_ORDER.get(row.representation, 99),
        ),
    )


def aggregate_rows(
    metrics,
    group_type: str,
    group_getter,
) -> list[AggregateRow]:
    from .cluster_stability_analysis import aggregate_rows as base_aggregate_rows

    return base_aggregate_rows(metrics, group_type, group_getter)


def write_comparison_csv(path: Path, rows: list[AggregateRow]) -> None:
    fieldnames = [
        "group_type",
        "group_value",
        "representation",
        "utterance_count",
        "mean_entropy_bits",
        "mean_top1_prob",
        "mean_nonzero_label_count",
        "mean_entropy_delta_vs_emoji",
        "mean_top1_delta_vs_emoji",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows(rows):
            writer.writerow(
                {
                    "group_type": row.group_type,
                    "group_value": row.group_value,
                    "representation": row.representation,
                    "utterance_count": row.utterance_count,
                    "mean_entropy_bits": f"{row.mean_entropy_bits:.12f}",
                    "mean_top1_prob": f"{row.mean_top1_prob:.12f}",
                    "mean_nonzero_label_count": f"{row.mean_nonzero_count:.12f}",
                    "mean_entropy_delta_vs_emoji": f"{row.mean_entropy_delta_vs_emoji:.12f}",
                    "mean_top1_delta_vs_emoji": f"{row.mean_top1_delta_vs_emoji:.12f}",
                }
            )


def row_lookup(rows: list[AggregateRow]) -> dict[tuple[str, str], AggregateRow]:
    return {(row.group_value, row.representation): row for row in rows}


def optional_fields(fields: tuple[str, ...]) -> list[str]:
    candidates = [
        "mean_confidence",
        "utterance_mean_confidence",
        "row_mean_confidence",
        "agreement_type",
        "agreement_pattern",
        "top1_prob",
        "entropy",
    ]
    return [field for field in candidates if field in fields]


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def choose_raw_recommendation(overall_rows: list[AggregateRow]) -> tuple[bool, str]:
    rows = {row.representation: row for row in overall_rows if row.group_value == "all"}
    emoji = rows.get("emoji")
    raw = rows.get("cluster_raw")
    sharp = rows.get("cluster_sharp")
    if emoji is None or raw is None:
        return True, "raw membership is available and is the requested default."

    raw_entropy_drop = emoji.mean_entropy_bits - raw.mean_entropy_bits
    raw_top1_gain = raw.mean_top1_prob - emoji.mean_top1_prob
    if sharp is None:
        return (
            raw_entropy_drop >= 0 and raw_top1_gain >= 0,
            "raw membership is the only available cluster membership version.",
        )

    sharp_entropy_drop = emoji.mean_entropy_bits - sharp.mean_entropy_bits
    sharp_top1_gain = sharp.mean_top1_prob - emoji.mean_top1_prob
    raw_is_conservative = raw.mean_entropy_bits <= sharp.mean_entropy_bits + 1e-9
    sharp_gain_is_tiny = abs(sharp_top1_gain - raw_top1_gain) < 0.002
    use_raw = raw_entropy_drop > 0 and raw_top1_gain > 0 and (raw_is_conservative or sharp_gain_is_tiny)
    reason = (
        "raw lowers entropy and raises top1 probability while preserving the original soft membership mass."
        if use_raw
        else "sharpened materially improves concentration enough to consider it over raw."
    )
    return use_raw, reason


def write_report(
    path: Path,
    soft_path: Path,
    soft_fields: tuple[str, ...],
    membership_path: Path,
    membership_fields: tuple[str, ...],
    overall_rows: list[AggregateRow],
    role_rows: list[AggregateRow],
    split_rows: list[AggregateRow],
    diagnostics: dict[str, float],
    output_paths: dict[str, Path],
) -> bool:
    overall = {row.representation: row for row in overall_rows if row.group_value == "all"}
    roles = row_lookup(role_rows)
    splits = row_lookup(split_rows)
    use_raw, raw_reason = choose_raw_recommendation(overall_rows)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Cluster Soft Label Comparison\n\n")
        handle.write("## Inputs\n\n")
        handle.write(f"- Soft emoji label table: `{soft_path}`\n")
        handle.write(f"- Emoji-to-cluster membership matrix: `{membership_path}`\n")
        handle.write(f"- Optional soft-label fields read: `{', '.join(optional_fields(soft_fields)) or 'none'}`\n")
        versions = ["raw"]
        if "membership_sharp" in membership_fields:
            versions.append("sharpened")
        handle.write(
            "- Membership versions read: "
            + ", ".join(f"`{version}`" for version in versions)
            + "\n"
        )
        handle.write("- Emoji names and transition information were not used.\n\n")

        handle.write("## Overall\n\n")
        handle.write("| representation | mean entropy bits | mean top1 probability | mean nonzero labels | entropy delta vs emoji | top1 delta vs emoji |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for representation in ("emoji", "cluster_raw", "cluster_sharp"):
            row = overall.get(representation)
            if row is None:
                continue
            handle.write(
                f"| `{representation}` | {row.mean_entropy_bits:.6f} | "
                f"{row.mean_top1_prob:.6f} | {row.mean_nonzero_count:.6f} | "
                f"{format_delta(row.mean_entropy_delta_vs_emoji)} | "
                f"{format_delta(row.mean_top1_delta_vs_emoji)} |\n"
            )

        handle.write("\n## Role\n\n")
        for representation in ("emoji", "cluster_raw", "cluster_sharp"):
            a = roles.get(("A", representation))
            b = roles.get(("B", representation))
            if a is None or b is None:
                continue
            handle.write(
                f"- `{representation}`: A entropy `{a.mean_entropy_bits:.6f}`, "
                f"B entropy `{b.mean_entropy_bits:.6f}`; A top1 `{a.mean_top1_prob:.6f}`, "
                f"B top1 `{b.mean_top1_prob:.6f}`.\n"
            )

        handle.write("\n## Split\n\n")
        for split in sorted({row.group_value for row in split_rows}):
            raw = splits.get((split, "cluster_raw"))
            sharp = splits.get((split, "cluster_sharp"))
            emoji = splits.get((split, "emoji"))
            if raw is None or emoji is None:
                continue
            sharp_text = (
                f", sharpened entropy `{sharp.mean_entropy_bits:.6f}`, "
                f"sharpened top1 `{sharp.mean_top1_prob:.6f}`"
                if sharp is not None
                else ""
            )
            handle.write(
                f"- `{split}`: emoji entropy `{emoji.mean_entropy_bits:.6f}`, "
                f"raw entropy `{raw.mean_entropy_bits:.6f}`, raw top1 `{raw.mean_top1_prob:.6f}`"
                f"{sharp_text}.\n"
            )

        emoji = overall.get("emoji")
        raw = overall.get("cluster_raw")
        handle.write("\n## Conclusion\n\n")
        if emoji is not None and raw is not None:
            handle.write(
                f"- Raw cluster projection changes entropy by "
                f"`{format_delta(raw.mean_entropy_bits - emoji.mean_entropy_bits)}` bits "
                f"and top1 probability by `{format_delta(raw.mean_top1_prob - emoji.mean_top1_prob)}`.\n"
            )
        a_raw = roles.get(("A", "cluster_raw"))
        b_raw = roles.get(("B", "cluster_raw"))
        if a_raw is not None and b_raw is not None:
            stable = b_raw.mean_entropy_bits < a_raw.mean_entropy_bits and b_raw.mean_top1_prob > a_raw.mean_top1_prob
            handle.write(
                f"- B is {'still' if stable else 'not'} more stable than A at cluster level "
                f"under raw membership.\n"
            )
        handle.write(f"- Raw membership assessment: {raw_reason}\n")
        handle.write(f"- Missing membership mass: `{diagnostics['missing_membership_mass']:.12f}`\n\n")

        handle.write("## Outputs\n\n")
        for label, output_path in output_paths.items():
            handle.write(f"- {label}: `{output_path}`\n")

        recommendation = "use" if use_raw else "not prioritize"
        handle.write(f"\nRecommendation: {recommendation} raw membership for the main graph.\n")
    return use_raw


def run_comparison(root: Path, output_dir: Path | None = None) -> tuple[dict[str, Path], bool]:
    soft_candidate, membership_candidate, four_model_candidate = discover_inputs(root)
    utterances, max_soft_sum_error = load_soft_label_table(soft_candidate.path)
    matrices = [load_membership(membership_candidate.path, "raw", "membership_raw")]
    if "membership_sharp" in membership_candidate.fields:
        matrices.append(load_membership(membership_candidate.path, "sharp", "membership_sharp"))
    four_model_labels = load_four_model_labels(
        four_model_candidate.path if four_model_candidate is not None else None
    )
    metrics, diagnostics = build_representation_metrics(
        utterances,
        tuple(matrices),
        four_model_labels,
    )
    diagnostics["max_soft_label_sum_error"] = max_soft_sum_error

    overall_rows = aggregate_rows(metrics, "overall", lambda row: "all")
    role_rows = aggregate_rows(metrics, "role", lambda row: row.role)
    split_rows = aggregate_rows(metrics, "split", lambda row: row.split)

    actual_output_dir = output_dir if output_dir is not None else default_output_dir(soft_candidate.path)
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "cluster_soft_labels_summary": actual_output_dir / "cluster_soft_labels_summary.csv",
        "cluster_soft_labels_by_role": actual_output_dir / "cluster_soft_labels_by_role.csv",
        "cluster_soft_labels_by_split": actual_output_dir / "cluster_soft_labels_by_split.csv",
        "cluster_soft_labels_report": actual_output_dir / "cluster_soft_labels_report.md",
        "emoji_vs_cluster_entropy": actual_output_dir / "emoji_vs_cluster_entropy.png",
    }
    write_comparison_csv(output_paths["cluster_soft_labels_summary"], overall_rows)
    write_comparison_csv(output_paths["cluster_soft_labels_by_role"], role_rows)
    write_comparison_csv(output_paths["cluster_soft_labels_by_split"], split_rows)
    draw_entropy_chart(output_paths["emoji_vs_cluster_entropy"], overall_rows)
    use_raw = write_report(
        output_paths["cluster_soft_labels_report"],
        soft_candidate.path,
        soft_candidate.fields,
        membership_candidate.path,
        membership_candidate.fields,
        overall_rows,
        role_rows,
        split_rows,
        diagnostics,
        output_paths,
    )
    return output_paths, use_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utterance-level soft emoji labels with projected soft cluster labels."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths, use_raw = run_comparison(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
    )
    print(f"Wrote summary: {output_paths['cluster_soft_labels_summary']}")
    print(f"Wrote role breakdown: {output_paths['cluster_soft_labels_by_role']}")
    print(f"Wrote split breakdown: {output_paths['cluster_soft_labels_by_split']}")
    print(f"Wrote report: {output_paths['cluster_soft_labels_report']}")
    print(f"Wrote entropy plot: {output_paths['emoji_vs_cluster_entropy']}")
    print(f"Recommendation: {'use' if use_raw else 'do not prioritize'} raw membership for the main graph.")


if __name__ == "__main__":
    main()
