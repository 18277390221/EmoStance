#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ["reasonable", "questionable_but_acceptable", "clearly_unreasonable"]
SUMMARY_GROUPS = ["overall", "split", "sampled_model", "original_emotion", "confidence_bin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate three emoji audit annotator exports.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Exported annotator JSONL/CSV files.")
    parser.add_argument("--out_dir", default="data_eval/results")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260519)
    return parser.parse_args()


def read_export(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input format: {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def majority_label(labels: list[str]) -> str:
    counts = Counter(labels)
    if counts["clearly_unreasonable"] >= 2:
        return "invalid"
    if counts["questionable_but_acceptable"] >= 2:
        return "ambiguous"
    if counts["reasonable"] == 1 and counts["questionable_but_acceptable"] == 1 and counts["clearly_unreasonable"] == 1:
        return "ambiguous"
    return "valid"


def confidence_bin(value: Any) -> str:
    if value in (None, "", "NA"):
        return "missing"
    try:
        ivalue = int(float(value))
    except (TypeError, ValueError):
        return str(value) if value in {"1-2", "3", "4-5", "missing"} else "missing"
    if ivalue <= 2:
        return "1-2"
    if ivalue == 3:
        return "3"
    if ivalue >= 4:
        return "4-5"
    return "missing"


def build_majority_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_item: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("package_id")), str(row.get("turn_index")))
        by_item[key].append(row)
    majority_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    for (package_id, turn_index), subset in sorted(by_item.items()):
        labels = [str(row.get("label") or "reasonable") for row in subset]
        label_counts = Counter(labels)
        first = subset[0]
        outcome = majority_label(labels)
        annotators = sorted({str(row.get("annotator_id")) for row in subset})
        majority = {
            "package_id": package_id,
            "turn_index": turn_index,
            "dialogue_id": first.get("dialogue_id", ""),
            "split": first.get("split", ""),
            "sampled_model": first.get("sampled_model", ""),
            "original_emotion": first.get("original_emotion", ""),
            "role": first.get("role", ""),
            "english_original_text": first.get("english_original_text", ""),
            "zh_translation": first.get("zh_translation", ""),
            "displayed_emoji": first.get("displayed_emoji", ""),
            "confidence_hidden_if_available": first.get("confidence_hidden_if_available", ""),
            "confidence_bin": first.get("confidence_bin") or confidence_bin(first.get("confidence_hidden_if_available")),
            "annotator_count": len(annotators),
            "reasonable_count": label_counts["reasonable"],
            "questionable_count": label_counts["questionable_but_acceptable"],
            "clearly_unreasonable_count": label_counts["clearly_unreasonable"],
            "majority_label": outcome,
            "annotators": " ".join(annotators),
        }
        majority_rows.append(majority)
        if len(set(labels)) > 1:
            disagreement = dict(majority)
            disagreement["raw_labels"] = json.dumps(labels, ensure_ascii=False)
            disagreement_rows.append(disagreement)
    return majority_rows, disagreement_rows


def proportions(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = len(rows)
    counts = Counter(row["majority_label"] for row in rows)
    if total == 0:
        return {"p_valid": math.nan, "p_ambiguous": math.nan, "p_invalid": math.nan, "p_plausible": math.nan}
    p_valid = counts["valid"] / total
    p_ambiguous = counts["ambiguous"] / total
    p_invalid = counts["invalid"] / total
    return {
        "p_valid": p_valid,
        "p_ambiguous": p_ambiguous,
        "p_invalid": p_invalid,
        "p_plausible": p_valid + p_ambiguous,
    }


def bootstrap_ci(rows: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    by_package: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_package[row["package_id"]].append(row)
    packages = sorted(by_package)
    if not packages:
        return {key: (math.nan, math.nan) for key in ("p_valid", "p_ambiguous", "p_invalid", "p_plausible")}
    samples: dict[str, list[float]] = {key: [] for key in ("p_valid", "p_ambiguous", "p_invalid", "p_plausible")}
    for _ in range(bootstrap):
        resampled: list[dict[str, Any]] = []
        for package_id in (rng.choice(packages) for _ in packages):
            resampled.extend(by_package[package_id])
        prop = proportions(resampled)
        for key in samples:
            samples[key].append(prop[key])
    ci: dict[str, tuple[float, float]] = {}
    for key, values in samples.items():
        values.sort()
        lo = values[int(0.025 * (len(values) - 1))]
        hi = values[int(0.975 * (len(values) - 1))]
        ci[key] = (lo, hi)
    return ci


def summarize_subset(group: str, value: str, rows: list[dict[str, Any]], bootstrap: int, seed: int) -> dict[str, Any]:
    prop = proportions(rows)
    ci = bootstrap_ci(rows, bootstrap, seed) if rows else {}
    return {
        "group": group,
        "value": value,
        "packages": len({row["package_id"] for row in rows}),
        "turns": len(rows),
        "p_valid": prop["p_valid"],
        "p_valid_ci_low": ci.get("p_valid", (math.nan, math.nan))[0],
        "p_valid_ci_high": ci.get("p_valid", (math.nan, math.nan))[1],
        "p_ambiguous": prop["p_ambiguous"],
        "p_ambiguous_ci_low": ci.get("p_ambiguous", (math.nan, math.nan))[0],
        "p_ambiguous_ci_high": ci.get("p_ambiguous", (math.nan, math.nan))[1],
        "p_invalid": prop["p_invalid"],
        "p_invalid_ci_low": ci.get("p_invalid", (math.nan, math.nan))[0],
        "p_invalid_ci_high": ci.get("p_invalid", (math.nan, math.nan))[1],
        "p_plausible": prop["p_plausible"],
        "p_plausible_ci_low": ci.get("p_plausible", (math.nan, math.nan))[0],
        "p_plausible_ci_high": ci.get("p_plausible", (math.nan, math.nan))[1],
    }


def agreement_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_item: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        by_item[(str(row.get("package_id")), str(row.get("turn_index")))].append(str(row.get("label") or "reasonable"))
    complete = [labels for labels in by_item.values() if len(labels) == 3]
    if not complete:
        return {"raw_exact_agreement": math.nan, "pairwise_raw_agreement": math.nan, "fleiss_kappa": math.nan}
    raw_exact = sum(1 for labels in complete if len(set(labels)) == 1) / len(complete)
    pair_agreements = []
    for labels in complete:
        pair_agreements.extend([labels[0] == labels[1], labels[0] == labels[2], labels[1] == labels[2]])
    pairwise = sum(pair_agreements) / len(pair_agreements)

    n = len(complete)
    k = len(LABELS)
    label_index = {label: idx for idx, label in enumerate(LABELS)}
    p_j = [0.0] * k
    p_i_values = []
    for labels in complete:
        counts = [0] * k
        for label in labels:
            if label in label_index:
                counts[label_index[label]] += 1
        for idx, count in enumerate(counts):
            p_j[idx] += count
        p_i_values.append((sum(count * count for count in counts) - 3) / (3 * 2))
    p_j = [value / (n * 3) for value in p_j]
    p_bar = statistics.mean(p_i_values)
    p_e = sum(value * value for value in p_j)
    kappa = (p_bar - p_e) / (1 - p_e) if (1 - p_e) else math.nan
    return {"raw_exact_agreement": raw_exact, "pairwise_raw_agreement": pairwise, "fleiss_kappa": kappa}


def fmt_pct(value: float) -> str:
    if value != value:
        return "NA"
    return f"{value * 100:.1f}%"


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], agreement: dict[str, Any]) -> None:
    overall = next(row for row in summary_rows if row["group"] == "overall")
    lines = [
        "# Emoji Weak-Label Human Audit Summary",
        "",
        "## Methods paragraph",
        "",
        "We conduct a representative dialogue-level human audit on 300 sampled model-dialogue packages. Packages are sampled from the finalized dialogue-level split in proportion to train, development, and test sizes, with the four LLM annotator models balanced across packages. Annotators view one complete dialogue at a time and inspect the emoji assigned by the sampled model to each utterance. For annotation convenience, the interface displays both the original English text and a Simplified Chinese translation produced by a local Mistral-7B-Instruct-v0.3 model. Annotators flag turns as questionable but plausible or clearly unreasonable with respect to the utterance meaning and dialogue context. Each package is reviewed by three independent annotators. We report turn-level valid, ambiguous, invalid, and plausible rates with package-clustered bootstrap confidence intervals. This audit estimates representative corpus-level weak-label validity and is not intended as exhaustive validation of every rare emoji or boundary-cluster case.",
        "",
        "## Paper-ready table",
        "",
        "| Audit subset | Packages | Turns | Valid | Ambiguous | Invalid | Plausible | Agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Overall | {overall['packages']} | {overall['turns']} | {fmt_pct(overall['p_valid'])} | {fmt_pct(overall['p_ambiguous'])} | {fmt_pct(overall['p_invalid'])} | {fmt_pct(overall['p_plausible'])} | {fmt_pct(agreement['raw_exact_agreement'])} |",
        "",
        "Agreement reports exact 3-annotator label agreement over the three audit labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_methods(path: Path) -> None:
    text = (
        "We conduct a representative dialogue-level human audit on 300 sampled model-dialogue packages. "
        "Packages are sampled from the finalized dialogue-level split in proportion to train, development, "
        "and test sizes, with the four LLM annotator models balanced across packages. Annotators view one "
        "complete dialogue at a time and inspect the emoji assigned by the sampled model to each utterance. "
        "For annotation convenience, the interface displays both the original English text and a Simplified "
        "Chinese translation produced by a local Mistral-7B-Instruct-v0.3 model. Annotators flag turns as "
        "questionable but plausible or clearly unreasonable with respect to the utterance meaning and dialogue "
        "context. Each package is reviewed by three independent annotators. We report turn-level valid, "
        "ambiguous, invalid, and plausible rates with package-clustered bootstrap confidence intervals. This "
        "audit estimates representative corpus-level weak-label validity and is not intended as exhaustive "
        "validation of every rare emoji or boundary-cluster case.\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        raw_rows.extend(read_export(Path(input_path)))
    majority_rows, disagreement_rows = build_majority_rows(raw_rows)
    summary_rows: list[dict[str, Any]] = []
    summary_rows.append(summarize_subset("overall", "all", majority_rows, args.bootstrap, args.seed))
    for group in ("split", "sampled_model", "original_emotion", "confidence_bin"):
        values = sorted({str(row.get(group) or "missing") for row in majority_rows})
        for value in values:
            subset = [row for row in majority_rows if str(row.get(group) or "missing") == value]
            summary_rows.append(summarize_subset(group, value, subset, args.bootstrap, args.seed))
    agreement = agreement_stats(raw_rows)
    summary = {
        "num_packages": len({row["package_id"] for row in majority_rows}),
        "num_turn_level_audit_items": len(majority_rows),
        "annotators_per_item": dict(Counter(row["annotator_count"] for row in majority_rows)),
        "overall": summary_rows[0],
        "agreement": agreement,
    }
    (out_dir / "audit_results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(
        out_dir / "audit_results_summary.csv",
        summary_rows,
        [
            "group",
            "value",
            "packages",
            "turns",
            "p_valid",
            "p_valid_ci_low",
            "p_valid_ci_high",
            "p_ambiguous",
            "p_ambiguous_ci_low",
            "p_ambiguous_ci_high",
            "p_invalid",
            "p_invalid_ci_low",
            "p_invalid_ci_high",
            "p_plausible",
            "p_plausible_ci_low",
            "p_plausible_ci_high",
        ],
    )
    write_jsonl(out_dir / "majority_turn_labels.jsonl", majority_rows)
    write_jsonl(out_dir / "disagreement_items.jsonl", disagreement_rows)
    write_markdown(out_dir / "audit_results_summary.md", summary_rows, agreement)
    write_paper_methods(out_dir / "paper_methods_snippet.md")
    print("Done.")
    print(f"Packages: {summary['num_packages']}")
    print(f"Turn-level audit items: {summary['num_turn_level_audit_items']}")
    print(f"p_valid: {summary_rows[0]['p_valid']:.4f}")
    print(f"p_ambiguous: {summary_rows[0]['p_ambiguous']:.4f}")
    print(f"p_invalid: {summary_rows[0]['p_invalid']:.4f}")
    print(f"p_plausible: {summary_rows[0]['p_plausible']:.4f}")
    print(f"Fleiss kappa: {agreement['fleiss_kappa']:.4f}")


if __name__ == "__main__":
    main()
