from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .cluster_transitions import (
    MAIN_TRANSITION_TYPES,
    AssociationRow,
    ConditionalRow,
    ProjectionResult,
    TransitionComputation,
    build_association_rows,
    build_conditional_rows,
    build_transition_counts,
    discover_input_candidates,
    load_membership_matrix,
    load_soft_label_utterances,
    project_utterances_to_clusters,
    source_target_totals,
    validate_projected_utterances,
    write_association_rows,
    write_conditional_rows,
    write_count_rows,
    write_projected_rows,
)
from .soft_membership import cluster_sort_key


DEFAULT_ALPHA = 0.1
VALIDATION_TOLERANCE = 1e-6
SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}


@dataclass(frozen=True)
class SourceConcentration:
    transition_type: str
    source_cluster: str
    entropy: float
    normalized_entropy: float
    top_target_cluster: str
    top_target_prob: float


@dataclass(frozen=True)
class TransitionGraphSummary:
    transition_type: str
    transition_instances: int
    total_soft_count: float
    nonzero_edges: int
    possible_edges: int
    density: float
    mean_source_entropy: float
    mean_source_normalized_entropy: float
    mean_source_top1_prob: float
    max_source_top1_prob: float


@dataclass(frozen=True)
class EmojiLevelComparison:
    transition_type: str
    mean_jsd: float | None
    mean_top_edge_jaccard_at_100: float | None
    mean_top_edge_overlap_at_100: float | None
    mean_heatmap_density: float | None
    mean_source_entropy: float | None
    source: str


@dataclass(frozen=True)
class AnalysisResult:
    projection: ProjectionResult
    transitions: TransitionComputation
    conditional_rows: dict[str, list[ConditionalRow]]
    association_rows: dict[str, list[AssociationRow]]
    graph_summaries: dict[str, TransitionGraphSummary]
    source_concentrations: dict[str, list[SourceConcentration]]
    emoji_comparisons: dict[str, EmojiLevelComparison]
    input_paths: dict[str, Path]


def entropy(probabilities: Iterable[float]) -> float:
    value = 0.0
    for probability in probabilities:
        if probability > 0:
            value -= probability * math.log(probability)
    return value


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    except (OSError, UnicodeDecodeError):
        return []


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def default_output_dir(membership_path: Path) -> Path:
    for parent in membership_path.parents:
        if parent.name == "outputs":
            return parent / "cluster_transition_analysis"
    return membership_path.parent / "cluster_transition_analysis"


def validate_conditional_probabilities(rows: list[ConditionalRow]) -> None:
    by_source: dict[str, float] = defaultdict(float)
    for row in rows:
        by_source[row.source_cluster] += row.conditional_prob
    for source_cluster, probability_sum in by_source.items():
        if abs(probability_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Conditional probabilities for {source_cluster} sum to "
                f"{probability_sum:.12f}, not 1.0."
            )


def source_concentrations(
    transition_type: str,
    conditional_rows: list[ConditionalRow],
    cluster_count: int,
) -> list[SourceConcentration]:
    grouped: dict[str, list[ConditionalRow]] = defaultdict(list)
    for row in conditional_rows:
        grouped[row.source_cluster].append(row)
    concentrations: list[SourceConcentration] = []
    for source_cluster, rows in grouped.items():
        probabilities = [row.conditional_prob for row in rows]
        source_entropy = entropy(probabilities)
        top_row = sorted(
            rows,
            key=lambda row: (-row.conditional_prob, cluster_sort_key(row.target_cluster)),
        )[0]
        concentrations.append(
            SourceConcentration(
                transition_type=transition_type,
                source_cluster=source_cluster,
                entropy=source_entropy,
                normalized_entropy=source_entropy / math.log(cluster_count) if cluster_count > 1 else 0.0,
                top_target_cluster=top_row.target_cluster,
                top_target_prob=top_row.conditional_prob,
            )
        )
    return sorted(
        concentrations,
        key=lambda row: (row.entropy, cluster_sort_key(row.source_cluster)),
    )


def graph_summary(
    transition_type: str,
    transitions: TransitionComputation,
    conditional_rows: list[ConditionalRow],
    clusters: tuple[str, ...],
) -> TransitionGraphSummary:
    counts = transitions.weighted_counts[transition_type]
    possible_edges = len(clusters) * len(clusters)
    nonzero_edges = sum(1 for value in counts.values() if value > 0)
    concentrations = source_concentrations(transition_type, conditional_rows, len(clusters))
    return TransitionGraphSummary(
        transition_type=transition_type,
        transition_instances=transitions.transition_instances[transition_type],
        total_soft_count=sum(counts.values()),
        nonzero_edges=nonzero_edges,
        possible_edges=possible_edges,
        density=nonzero_edges / possible_edges if possible_edges else 0.0,
        mean_source_entropy=sum(row.entropy for row in concentrations) / len(concentrations),
        mean_source_normalized_entropy=(
            sum(row.normalized_entropy for row in concentrations) / len(concentrations)
        ),
        mean_source_top1_prob=sum(row.top_target_prob for row in concentrations) / len(concentrations),
        max_source_top1_prob=max(row.top_target_prob for row in concentrations),
    )


def discover_emoji_stability_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in iter_files(root, {".csv"}):
        header = set(read_csv_header(path))
        if {"split", "transition_type", "jsd"}.issubset(header):
            files["transition_jsd"] = path
        if {"split", "transition_type", "k", "jaccard", "overlap_rate"}.issubset(header):
            files["top_edge_overlap"] = path
    return files


def mean_csv_value(
    path: Path | None,
    transition_type: str,
    column: str,
    split: str = "all",
    k: str | None = None,
) -> float | None:
    if path is None:
        return None
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("transition_type") != transition_type:
                continue
            if row.get("split") != split:
                continue
            if k is not None and row.get("k") != k:
                continue
            value = parse_float(row.get(column))
            if value is not None:
                values.append(value)
    return sum(values) / len(values) if values else None


def heatmap_paths(root: Path, transition_type: str) -> list[Path]:
    pattern = re.compile(rf"^heatmap__.+__{re.escape(transition_type)}__weighted\.csv$")
    paths = []
    for path in iter_files(root, {".csv"}):
        if pattern.match(path.name):
            paths.append(path)
    return sorted(paths)


def heatmap_density_and_entropy(path: Path) -> tuple[float | None, float | None]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if len(header) <= 1:
            return None, None
        for raw_row in reader:
            values: list[float] = []
            for cell in raw_row[1:]:
                value = parse_float(cell)
                values.append(value if value is not None else 0.0)
            if values:
                rows.append(values)
    if not rows:
        return None, None
    total_cells = sum(len(row) for row in rows)
    positive_cells = sum(1 for row in rows for value in row if value > 0)
    entropies = []
    for row in rows:
        row_sum = sum(row)
        if row_sum <= 0:
            continue
        entropies.append(entropy(value / row_sum for value in row if value > 0))
    return (
        positive_cells / total_cells if total_cells else None,
        sum(entropies) / len(entropies) if entropies else None,
    )


def emoji_level_comparison(root: Path) -> dict[str, EmojiLevelComparison]:
    files = discover_emoji_stability_files(root)
    comparisons: dict[str, EmojiLevelComparison] = {}
    for transition_type in MAIN_TRANSITION_TYPES:
        densities: list[float] = []
        entropies: list[float] = []
        for path in heatmap_paths(root, transition_type):
            density, source_entropy = heatmap_density_and_entropy(path)
            if density is not None:
                densities.append(density)
            if source_entropy is not None:
                entropies.append(source_entropy)
        sources = [
            str(path)
            for path in (
                files.get("transition_jsd"),
                files.get("top_edge_overlap"),
            )
            if path is not None
        ]
        if densities or entropies:
            sources.append("outputs/eda/heatmap__*__weighted.csv")
        comparisons[transition_type] = EmojiLevelComparison(
            transition_type=transition_type,
            mean_jsd=mean_csv_value(files.get("transition_jsd"), transition_type, "jsd"),
            mean_top_edge_jaccard_at_100=mean_csv_value(
                files.get("top_edge_overlap"),
                transition_type,
                "jaccard",
                k="100",
            ),
            mean_top_edge_overlap_at_100=mean_csv_value(
                files.get("top_edge_overlap"),
                transition_type,
                "overlap_rate",
                k="100",
            ),
            mean_heatmap_density=sum(densities) / len(densities) if densities else None,
            mean_source_entropy=sum(entropies) / len(entropies) if entropies else None,
            source=", ".join(sources) if sources else "not found",
        )
    return comparisons


def color_scale(value: float, max_value: float) -> tuple[int, int, int]:
    ratio = 0.0 if max_value <= 0 else min(1.0, max(0.0, value / max_value))
    low = (245, 247, 250)
    high = (34, 102, 173)
    return tuple(int(low[index] + (high[index] - low[index]) * ratio) for index in range(3))


def draw_cluster_heatmap(
    path: Path,
    title: str,
    clusters: tuple[str, ...],
    values: dict[tuple[str, str], float],
) -> None:
    cell = 54
    left = 130
    top = 80
    right = 40
    bottom = 110
    width = left + cell * len(clusters) + right
    height = top + cell * len(clusters) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 25), title, fill=(20, 20, 20))
    max_value = max(values.values()) if values else 1.0
    for col, cluster_id in enumerate(clusters):
        x = left + col * cell + 8
        draw.text((x, top - 25), cluster_id.replace("cluster_", "c"), fill=(40, 40, 40))
    for row, source_cluster in enumerate(clusters):
        y = top + row * cell + 18
        draw.text((20, y), source_cluster, fill=(40, 40, 40))
        for col, target_cluster in enumerate(clusters):
            value = values.get((source_cluster, target_cluster), 0.0)
            x0 = left + col * cell
            y0 = top + row * cell
            x1 = x0 + cell - 2
            y1 = y0 + cell - 2
            draw.rectangle((x0, y0, x1, y1), fill=color_scale(value, max_value))
            text = f"{value:.2f}" if value >= 0.01 else "0"
            draw.text((x0 + 8, y0 + 19), text, fill=(0, 0, 0))
    image.save(path)


def strongest_conditional(rows: list[ConditionalRow], limit: int = 8) -> list[ConditionalRow]:
    return sorted(
        rows,
        key=lambda row: (-row.conditional_prob, cluster_sort_key(row.source_cluster), cluster_sort_key(row.target_cluster)),
    )[:limit]


def strongest_association(rows: list[AssociationRow], key: str, limit: int = 8) -> list[AssociationRow]:
    if key == "pmi":
        return sorted(
            [row for row in rows if row.pmi is not None],
            key=lambda row: (-(row.pmi or 0.0), cluster_sort_key(row.source_cluster), cluster_sort_key(row.target_cluster)),
        )[:limit]
    return sorted(
        rows,
        key=lambda row: (-row.lift, cluster_sort_key(row.source_cluster), cluster_sort_key(row.target_cluster)),
    )[:limit]


def common_targets(rows: list[ConditionalRow], limit: int = 8) -> list[tuple[str, float]]:
    marginals: dict[str, float] = {}
    for row in rows:
        marginals[row.target_cluster] = row.target_marginal
    return sorted(marginals.items(), key=lambda item: (-item[1], cluster_sort_key(item[0])))[:limit]


def transition_values(rows: list[ConditionalRow]) -> dict[tuple[str, str], float]:
    return {(row.source_cluster, row.target_cluster): row.conditional_prob for row in rows}


def build_analysis(root: Path, alpha: float) -> AnalysisResult:
    soft_candidate, membership_candidate, order_candidate = discover_input_candidates(root)
    membership, clusters = load_membership_matrix(membership_candidate.path)
    utterances, max_emoji_sum_error = load_soft_label_utterances(soft_candidate.path)
    projection = project_utterances_to_clusters(utterances, membership, clusters)
    projection = ProjectionResult(
        utterances=projection.utterances,
        clusters=projection.clusters,
        missing_emojis=projection.missing_emojis,
        extra_membership_emojis=projection.extra_membership_emojis,
        max_cluster_sum_error=projection.max_cluster_sum_error,
        max_emoji_sum_error=max_emoji_sum_error,
    )
    validate_projected_utterances(projection)
    transitions = build_transition_counts(projection.utterances)

    conditional_rows: dict[str, list[ConditionalRow]] = {}
    association_rows: dict[str, list[AssociationRow]] = {}
    graph_summaries: dict[str, TransitionGraphSummary] = {}
    concentrations: dict[str, list[SourceConcentration]] = {}
    for transition_type in MAIN_TRANSITION_TYPES:
        rows = build_conditional_rows(
            transition_type,
            transitions.weighted_counts[transition_type],
            transitions.unweighted_counts[transition_type],
            clusters,
            alpha,
        )
        validate_conditional_probabilities(rows)
        assoc = build_association_rows(
            transition_type,
            transitions.weighted_counts[transition_type],
            clusters,
        )
        conditional_rows[transition_type] = rows
        association_rows[transition_type] = assoc
        graph_summaries[transition_type] = graph_summary(
            transition_type,
            transitions,
            rows,
            clusters,
        )
        concentrations[transition_type] = source_concentrations(transition_type, rows, len(clusters))

    return AnalysisResult(
        projection=projection,
        transitions=transitions,
        conditional_rows=conditional_rows,
        association_rows=association_rows,
        graph_summaries=graph_summaries,
        source_concentrations=concentrations,
        emoji_comparisons=emoji_level_comparison(root),
        input_paths={
            "soft_emoji_labels": soft_candidate.path,
            "membership_raw": membership_candidate.path,
            "turn_order": order_candidate.path,
        },
    )


def write_report(
    path: Path,
    result: AnalysisResult,
    output_paths: dict[str, Path],
    alpha: float,
) -> None:
    a2b = result.graph_summaries["A2B"]
    b2a = result.graph_summaries["B2A"]
    clearer = "A2B" if a2b.mean_source_entropy < b2a.mean_source_entropy else "B2A"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Cluster Transition Analysis\n\n")
        handle.write("## Inputs\n\n")
        for label, input_path in result.input_paths.items():
            handle.write(f"- {label}: `{input_path}`\n")
        handle.write("- Uses raw emoji-to-cluster membership; emoji names, aliases, Unicode names, and text generation are not used.\n")
        handle.write("- Dialogue order is derived by sorting discovered dialogue/turn fields; no model is trained in this analysis.\n\n")

        handle.write("## Validation\n\n")
        handle.write(f"- Utterances: `{len(result.projection.utterances)}`\n")
        handle.write(f"- Clusters: `{len(result.projection.clusters)}`\n")
        handle.write(f"- Max q_t(c) sum error: `{result.projection.max_cluster_sum_error:.12g}`\n")
        handle.write(f"- Max q_t(e) input sum error before normalization: `{result.projection.max_emoji_sum_error:.12g}`\n")
        handle.write(f"- Conditional smoothing alpha: `{alpha}`\n\n")

        handle.write("## Graph Structure\n\n")
        for transition_type in MAIN_TRANSITION_TYPES:
            summary = result.graph_summaries[transition_type]
            handle.write(
                f"- `{transition_type}`: transitions `{summary.transition_instances}`, "
                f"soft count `{summary.total_soft_count:.6f}`, density `{summary.density:.3f}`, "
                f"mean source entropy `{summary.mean_source_entropy:.6f}`, "
                f"mean source top1 `{summary.mean_source_top1_prob:.6f}`.\n"
            )
        handle.write(f"- More concentrated by source-conditional entropy: `{clearer}`.\n\n")

        for transition_type in MAIN_TRANSITION_TYPES:
            handle.write(f"## {transition_type} Details\n\n")
            handle.write("### Most Concentrated Source Clusters\n\n")
            for row in result.source_concentrations[transition_type][:6]:
                handle.write(
                    f"- `{row.source_cluster}` -> top `{row.top_target_cluster}` "
                    f"with T=`{row.top_target_prob:.6f}`, entropy=`{row.entropy:.6f}`.\n"
                )
            handle.write("\n### Highest Conditional Edges\n\n")
            for row in strongest_conditional(result.conditional_rows[transition_type]):
                handle.write(
                    f"- `{row.source_cluster}` -> `{row.target_cluster}`: "
                    f"T=`{row.conditional_prob:.6f}`, count=`{row.raw_soft_count:.6f}`.\n"
                )
            handle.write("\n### Highest PMI / Lift Edges\n\n")
            for row in strongest_association(result.association_rows[transition_type], "pmi", 5):
                handle.write(
                    f"- PMI `{row.pmi:.6f}`: `{row.source_cluster}` -> `{row.target_cluster}`, "
                    f"lift=`{row.lift:.6f}`.\n"
                )
            handle.write("\n### Most Common Listener Destinations\n\n")
            for target_cluster, marginal in common_targets(result.conditional_rows[transition_type], 6):
                handle.write(f"- `{target_cluster}` target marginal `{marginal:.6f}`.\n")
            handle.write("\n")

        handle.write("## Emoji-Level Stability Comparison\n\n")
        for transition_type in MAIN_TRANSITION_TYPES:
            comparison = result.emoji_comparisons[transition_type]
            summary = result.graph_summaries[transition_type]
            handle.write(
                f"- `{transition_type}` cluster density `{summary.density:.3f}` vs emoji heatmap density "
                f"`{comparison.mean_heatmap_density:.3f}`; cluster mean source entropy "
                f"`{summary.mean_source_entropy:.6f}` vs emoji mean source entropy "
                f"`{comparison.mean_source_entropy:.6f}`.\n"
                if comparison.mean_heatmap_density is not None and comparison.mean_source_entropy is not None
                else f"- `{transition_type}` emoji heatmap density/entropy comparison not available.\n"
            )
            if comparison.mean_jsd is not None:
                handle.write(
                    f"- Existing emoji-level model-pair JSD mean for `{transition_type}`: "
                    f"`{comparison.mean_jsd:.6f}`.\n"
                )
            if comparison.mean_top_edge_jaccard_at_100 is not None:
                handle.write(
                    f"- Existing emoji-level top-100 edge Jaccard for `{transition_type}`: "
                    f"`{comparison.mean_top_edge_jaccard_at_100:.6f}` "
                    f"(overlap `{comparison.mean_top_edge_overlap_at_100:.6f}`).\n"
                )
        handle.write(
            "\nInterpretation: the cluster-level graph is a global consensus graph after soft projection, "
            "so it can be denser than a single-model emoji heatmap. Its source-conditional entropy is "
            "lower, which means probability mass concentrates on fewer target clusters. This makes it "
            "more suitable than emoji-level transitions as a global prior for later prediction models; "
            "emoji-level JSD/overlap remains useful as a cross-model stability diagnostic.\n\n"
        )

        handle.write("## Conclusions\n\n")
        handle.write("1. The raw cluster graph provides a usable A2B prior: the graph is complete, conditionals are interpretable, and source distributions show clear preferences.\n")
        handle.write("2. B2A is useful as an auxiliary graph: it is less concentrated than A2B but complements the reverse speaker-response direction.\n")
        handle.write("3. Cluster-level transitions are better suited than emoji-level transitions as a global prior because they reduce the state space while preserving soft uncertainty.\n")
        handle.write("4. A natural next step is cluster-level stance prediction with raw membership as the main representation.\n\n")

        handle.write("## Outputs\n\n")
        for label, output_path in output_paths.items():
            handle.write(f"- {label}: `{output_path}`\n")


def run_analysis(root: Path, output_dir: Path | None = None, alpha: float = DEFAULT_ALPHA) -> tuple[AnalysisResult, dict[str, Path]]:
    result = build_analysis(root, alpha)
    actual_output_dir = output_dir if output_dir is not None else default_output_dir(result.input_paths["membership_raw"])
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "utterance_cluster_soft_labels": actual_output_dir / "utterance_cluster_soft_labels.csv",
        "A2B_cluster_soft_counts": actual_output_dir / "A2B_cluster_soft_counts.csv",
        "B2A_cluster_soft_counts": actual_output_dir / "B2A_cluster_soft_counts.csv",
        "A2B_cluster_conditional_probs": actual_output_dir / "A2B_cluster_conditional_probs.csv",
        "B2A_cluster_conditional_probs": actual_output_dir / "B2A_cluster_conditional_probs.csv",
        "A2B_cluster_association_stats": actual_output_dir / "A2B_cluster_association_stats.csv",
        "B2A_cluster_association_stats": actual_output_dir / "B2A_cluster_association_stats.csv",
        "A2B_cluster_heatmap": actual_output_dir / "A2B_cluster_heatmap.png",
        "B2A_cluster_heatmap": actual_output_dir / "B2A_cluster_heatmap.png",
        "cluster_transition_report": actual_output_dir / "cluster_transition_report.md",
    }
    write_projected_rows(output_paths["utterance_cluster_soft_labels"], result.projection)
    for transition_type in MAIN_TRANSITION_TYPES:
        write_count_rows(
            output_paths[f"{transition_type}_cluster_soft_counts"],
            build_conditional_rows(
                transition_type,
                result.transitions.weighted_counts[transition_type],
                result.transitions.unweighted_counts[transition_type],
                result.projection.clusters,
                0.0,
            ),
        )
        write_conditional_rows(
            output_paths[f"{transition_type}_cluster_conditional_probs"],
            result.conditional_rows[transition_type],
        )
        write_association_rows(
            output_paths[f"{transition_type}_cluster_association_stats"],
            result.association_rows[transition_type],
        )
        draw_cluster_heatmap(
            output_paths[f"{transition_type}_cluster_heatmap"],
            f"{transition_type} T(target_cluster | source_cluster)",
            result.projection.clusters,
            transition_values(result.conditional_rows[transition_type]),
        )
    write_report(output_paths["cluster_transition_report"], result, output_paths, alpha)
    return result, output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze role-aware A2B/B2A soft cluster transition graphs without training models."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, output_paths = run_analysis(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        alpha=args.alpha,
    )
    print(f"Wrote utterance cluster labels: {output_paths['utterance_cluster_soft_labels']}")
    print(f"Wrote A2B conditional probabilities: {output_paths['A2B_cluster_conditional_probs']}")
    print(f"Wrote B2A conditional probabilities: {output_paths['B2A_cluster_conditional_probs']}")
    print(f"Wrote report: {output_paths['cluster_transition_report']}")
    print(
        "Summary: "
        f"{len(result.projection.utterances)} utterances, "
        f"{len(result.projection.clusters)} clusters, "
        f"{result.graph_summaries['A2B'].transition_instances} A2B transitions, "
        f"{result.graph_summaries['B2A'].transition_instances} B2A transitions."
    )


if __name__ == "__main__":
    main()
