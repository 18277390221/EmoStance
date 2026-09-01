from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_lib import (
    ExperimentError,
    distribution_vector,
    human_distribution,
    jsd_base2,
    load_experiment_dir,
    load_export_files,
    membership_matrix,
    read_json,
    validate_exports,
    write_csv,
    write_json,
)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def load_distributions(experiment_dir: Path, exports_dir: Path) -> tuple[list[dict[str, Any]], list[str], np.ndarray]:
    valid, validation, _ = validate_exports(experiment_dir, exports_dir)
    if not valid:
        raise ExperimentError(f"Export validation failed: {validation['errors']}")
    data = load_experiment_dir(experiment_dir)
    private_by_id = {item["sample_id"]: item for item in data["private_items"]}
    inventory = data["emoji_inventory"]
    emojis = [row["emoji"] for row in inventory]
    matrix = membership_matrix(inventory, data["config"]["clusters"])

    exports = {read_json(path)["annotator_id"]: read_json(path) for path in load_export_files(exports_dir)}
    answers_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotator_id in sorted(exports):
        for answer in exports[annotator_id]["answers"]:
            row = dict(answer)
            row["annotator_id"] = annotator_id
            answers_by_sample[row["sample_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for sample_id, private in private_by_id.items():
        answers = sorted(answers_by_sample[sample_id], key=lambda x: x["annotator_id"])
        rows.append(
            {
                "sample_id": sample_id,
                "disagreement_stratum": private["disagreement_stratum"],
                "llm_e": distribution_vector(private["llm_qE"], emojis),
                "human_e": distribution_vector(human_distribution(answers, emojis, confidence_weighted=False), emojis),
            }
        )
    return rows, emojis, matrix


def mean_region_jsd(rows: list[dict[str, Any]], matrix: np.ndarray) -> float:
    values = []
    for row in rows:
        values.append(jsd_base2(row["llm_e"] @ matrix, row["human_e"] @ matrix))
    return statistics.fmean(values)


def stratum_region_jsd(rows: list[dict[str, Any]], matrix: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for stratum in ("low", "medium", "high"):
        subset = [row for row in rows if row["disagreement_stratum"] == stratum]
        out[stratum] = mean_region_jsd(subset, matrix)
    return out


def run_permutations(
    rows: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    learned_overall = mean_region_jsd(rows, matrix)
    learned_by_stratum = stratum_region_jsd(rows, matrix)

    permutation_rows: list[dict[str, Any]] = []
    row_indices = list(range(matrix.shape[0]))
    for perm_idx in range(1, n_permutations + 1):
        perm = row_indices[:]
        rng.shuffle(perm)
        permuted = matrix[np.asarray(perm), :]
        by_stratum = stratum_region_jsd(rows, permuted)
        record = {
            "permutation_index": perm_idx,
            "overall_region_jsd": mean_region_jsd(rows, permuted),
            "low_region_jsd": by_stratum["low"],
            "medium_region_jsd": by_stratum["medium"],
            "high_region_jsd": by_stratum["high"],
        }
        permutation_rows.append(record)

    random_values = [float(row["overall_region_jsd"]) for row in permutation_rows]
    less_equal = sum(1 for value in random_values if value <= learned_overall)
    summary = {
        "n_permutations": n_permutations,
        "seed": seed,
        "learned_region_jsd": learned_overall,
        "learned_by_disagreement": learned_by_stratum,
        "random_region_jsd": {
            "mean": statistics.fmean(random_values),
            "std": statistics.pstdev(random_values),
            "min": min(random_values),
            "p025": percentile(random_values, 0.025),
            "median": percentile(random_values, 0.5),
            "p975": percentile(random_values, 0.975),
            "max": max(random_values),
        },
        "learned_minus_random_mean": learned_overall - statistics.fmean(random_values),
        "one_sided_empirical_p_random_leq_learned": (less_equal + 1) / (n_permutations + 1),
        "random_permutations_leq_learned": less_equal,
        "learned_below_random_p025": learned_overall < percentile(random_values, 0.025),
    }
    return permutation_rows, summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rv = summary["random_region_jsd"]
    lines = [
        "# Permutation Baseline for Learned Region Mapping",
        "",
        "This analysis tests whether the learned emoji-to-region membership matrix groups LLM-human exact-emoji disagreements more meaningfully than an arbitrary nine-region coarsening.",
        "",
        "For each permutation, the rows of the membership matrix were shuffled across emoji. This preserves the set of nine-region membership rows, the soft-membership shape of each row, and the aggregate region mass/count structure, while breaking the learned correspondence between emoji symbols and membership rows.",
        "",
        f"- Permutations: {summary['n_permutations']}",
        f"- Seed: {summary['seed']}",
        f"- Learned region JSD: {summary['learned_region_jsd']:.6f}",
        f"- Random mean region JSD: {rv['mean']:.6f}",
        f"- Random 2.5 percentile: {rv['p025']:.6f}",
        f"- Random median: {rv['median']:.6f}",
        f"- Random 97.5 percentile: {rv['p975']:.6f}",
        f"- Learned minus random mean: {summary['learned_minus_random_mean']:.6f}",
        f"- Empirical one-sided p-value, P(random <= learned): {summary['one_sided_empirical_p_random_leq_learned']:.6f}",
        f"- Learned below random 2.5 percentile: `{str(summary['learned_below_random_p025']).lower()}`",
        "",
        "Interpretation: lower region JSD indicates closer LLM-human agreement after region projection. If the learned mapping is below the lower tail of the random row-permutation distribution, the learned regions group symbol-level disagreements more meaningfully than arbitrary row assignments.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run row-permutation baseline for the emoji-to-region membership matrix.")
    parser.add_argument("--experiment-dir", default="human_llm_emoji_audit")
    parser.add_argument("--exports-dir", default="human_llm_emoji_audit/results")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        rows, _, matrix = load_distributions(Path(args.experiment_dir), Path(args.exports_dir))
        permutation_rows, summary = run_permutations(rows, matrix, n_permutations=args.n_permutations, seed=args.seed)
    except ExperimentError as exc:
        print(f"permutation_region_baseline: FAIL: {exc}", file=sys.stderr)
        sys.exit(1)

    reports = Path(args.experiment_dir) / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "permutation_region_baseline.json", summary)
    write_csv(
        reports / "permutation_region_baseline_distribution.csv",
        permutation_rows,
        ["permutation_index", "overall_region_jsd", "low_region_jsd", "medium_region_jsd", "high_region_jsd"],
    )
    write_markdown(reports / "permutation_region_baseline.md", summary)
    print("permutation_region_baseline: PASS")
    print(f"learned_region_jsd={summary['learned_region_jsd']:.6f}")
    print(f"random_p025={summary['random_region_jsd']['p025']:.6f}")
    print(f"random_mean={summary['random_region_jsd']['mean']:.6f}")
    print(f"empirical_p={summary['one_sided_empirical_p_random_leq_learned']:.6f}")


if __name__ == "__main__":
    main()
