import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "pre_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/utterance_agreement/outputs"
SPLITS = ("train", "valid", "test")
BUCKETS = ("4_of_4", "3_of_4", "2_of_4", "all_different")
PATTERN_TO_KEY = {
    "4": "pattern_4",
    "3-1": "pattern_3_1",
    "2-2": "pattern_2_2",
    "2-1-1": "pattern_2_1_1",
    "1-1-1-1": "pattern_1_1_1_1",
}
GROUP_TYPES = ("overall", "split", "role", "split_role")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def new_aggregator() -> dict[str, Any]:
    return {
        "utterance_count": 0,
        "bucket_counts": {bucket: 0 for bucket in BUCKETS},
        "pattern_counts": {pattern_key: 0 for pattern_key in PATTERN_TO_KEY.values()},
        "entropy_values": [],
        "majority_vote_fractions": [],
        "top1_prob_values": [],
        "unique_emoji_counts": [],
    }


def normalize_probabilities(weight_by_emoji: dict[str, float]) -> dict[str, float]:
    total = sum(weight_by_emoji.values())
    if total <= 0:
        return {}

    items = sorted(weight_by_emoji.items(), key=lambda item: (-item[1], item[0]))
    return {emoji: round(weight / total, 8) for emoji, weight in items if weight > 0}


def compute_entropy(probabilities: dict[str, float]) -> float:
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return round(entropy, 8)


def classify_agreement(emojis: list[str]) -> tuple[str, str, int, list[str], int]:
    counts = Counter(emojis)
    pattern = "-".join(str(count) for count in sorted(counts.values(), reverse=True))
    majority_vote_count = max(counts.values(), default=0)
    majority_emojis = sorted(
        emoji for emoji, count in counts.items() if count == majority_vote_count
    )

    if pattern == "4":
        bucket = "4_of_4"
    elif pattern == "3-1":
        bucket = "3_of_4"
    elif pattern in {"2-2", "2-1-1"}:
        bucket = "2_of_4"
    elif pattern == "1-1-1-1":
        bucket = "all_different"
    else:
        raise ValueError(f"Unexpected agreement pattern: {pattern}")

    return bucket, pattern, majority_vote_count, majority_emojis, len(counts)


def build_soft_label(
    emoji_map: dict[str, Any],
    confidence_map: dict[str, Any],
    models: list[str],
) -> tuple[dict[str, float], float, str | None, float]:
    weight_by_emoji: dict[str, float] = {}

    for model in models:
        emoji = emoji_map.get(model)
        confidence = confidence_map.get(model)
        if emoji is None or confidence is None:
            continue

        weight = float(confidence)
        if weight <= 0:
            continue

        weight_by_emoji[emoji] = weight_by_emoji.get(emoji, 0.0) + weight

    probabilities = normalize_probabilities(weight_by_emoji)
    if not probabilities:
        return {}, 0.0, None, 0.0

    top1_emoji = next(iter(probabilities))
    top1_prob = probabilities[top1_emoji]
    entropy = compute_entropy(probabilities)
    return probabilities, entropy, top1_emoji, top1_prob


def update_aggregator(
    aggregator: dict[str, Any],
    bucket: str,
    pattern: str,
    entropy: float,
    majority_vote_fraction: float,
    top1_prob: float,
    unique_emoji_count: int,
) -> None:
    aggregator["utterance_count"] += 1
    aggregator["bucket_counts"][bucket] += 1
    aggregator["pattern_counts"][PATTERN_TO_KEY[pattern]] += 1
    aggregator["entropy_values"].append(entropy)
    aggregator["majority_vote_fractions"].append(majority_vote_fraction)
    aggregator["top1_prob_values"].append(top1_prob)
    aggregator["unique_emoji_counts"].append(unique_emoji_count)


def summarize_group(
    group_type: str,
    split: str,
    role: str,
    aggregator: dict[str, Any],
) -> dict[str, Any]:
    utterance_count = aggregator["utterance_count"]
    if utterance_count == 0:
        return {
            "group_type": group_type,
            "split": split,
            "role": role,
            "utterance_count": 0,
            "count_4_of_4": 0,
            "count_3_of_4": 0,
            "count_2_of_4": 0,
            "count_all_different": 0,
            "rate_4_of_4": 0.0,
            "rate_3_of_4": 0.0,
            "rate_2_of_4": 0.0,
            "rate_all_different": 0.0,
            "pattern_4_count": 0,
            "pattern_3_1_count": 0,
            "pattern_2_2_count": 0,
            "pattern_2_1_1_count": 0,
            "pattern_1_1_1_1_count": 0,
            "pattern_4_rate": 0.0,
            "pattern_3_1_rate": 0.0,
            "pattern_2_2_rate": 0.0,
            "pattern_2_1_1_rate": 0.0,
            "pattern_1_1_1_1_rate": 0.0,
            "mean_majority_vote_fraction": 0.0,
            "mean_soft_label_top1_prob": 0.0,
            "mean_entropy": 0.0,
            "median_entropy": 0.0,
            "mean_unique_emoji_count": 0.0,
        }

    bucket_counts = aggregator["bucket_counts"]
    pattern_counts = aggregator["pattern_counts"]
    entropy_values = aggregator["entropy_values"]
    majority_vote_fractions = aggregator["majority_vote_fractions"]
    top1_prob_values = aggregator["top1_prob_values"]
    unique_emoji_counts = aggregator["unique_emoji_counts"]

    return {
        "group_type": group_type,
        "split": split,
        "role": role,
        "utterance_count": utterance_count,
        "count_4_of_4": bucket_counts["4_of_4"],
        "count_3_of_4": bucket_counts["3_of_4"],
        "count_2_of_4": bucket_counts["2_of_4"],
        "count_all_different": bucket_counts["all_different"],
        "rate_4_of_4": round(bucket_counts["4_of_4"] / utterance_count, 8),
        "rate_3_of_4": round(bucket_counts["3_of_4"] / utterance_count, 8),
        "rate_2_of_4": round(bucket_counts["2_of_4"] / utterance_count, 8),
        "rate_all_different": round(bucket_counts["all_different"] / utterance_count, 8),
        "pattern_4_count": pattern_counts["pattern_4"],
        "pattern_3_1_count": pattern_counts["pattern_3_1"],
        "pattern_2_2_count": pattern_counts["pattern_2_2"],
        "pattern_2_1_1_count": pattern_counts["pattern_2_1_1"],
        "pattern_1_1_1_1_count": pattern_counts["pattern_1_1_1_1"],
        "pattern_4_rate": round(pattern_counts["pattern_4"] / utterance_count, 8),
        "pattern_3_1_rate": round(pattern_counts["pattern_3_1"] / utterance_count, 8),
        "pattern_2_2_rate": round(pattern_counts["pattern_2_2"] / utterance_count, 8),
        "pattern_2_1_1_rate": round(pattern_counts["pattern_2_1_1"] / utterance_count, 8),
        "pattern_1_1_1_1_rate": round(pattern_counts["pattern_1_1_1_1"] / utterance_count, 8),
        "mean_majority_vote_fraction": round(
            sum(majority_vote_fractions) / utterance_count,
            8,
        ),
        "mean_soft_label_top1_prob": round(sum(top1_prob_values) / utterance_count, 8),
        "mean_entropy": round(sum(entropy_values) / utterance_count, 8),
        "median_entropy": round(median(entropy_values), 8),
        "mean_unique_emoji_count": round(sum(unique_emoji_counts) / utterance_count, 8),
    }


def build_summary_rows(group_aggregators: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(summarize_group("overall", "all", "all", group_aggregators[("overall", "all", "all")]))

    for split in SPLITS:
        rows.append(summarize_group("split", split, "all", group_aggregators[("split", split, "all")]))

    for role in ("A", "B"):
        rows.append(summarize_group("role", "all", role, group_aggregators[("role", "all", role)]))

    for split in SPLITS:
        for role in ("A", "B"):
            rows.append(
                summarize_group(
                    "split_role",
                    split,
                    role,
                    group_aggregators[("split_role", split, role)],
                )
            )

    return rows


def build_summary_json(
    summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "metadata": metadata,
        "overall": {},
        "by_split": {},
        "by_role": {},
        "by_split_role": {},
    }

    for row in summary_rows:
        clean_row = {key: value for key, value in row.items() if key not in {"group_type", "split", "role"}}

        if row["group_type"] == "overall":
            summary["overall"] = clean_row
        elif row["group_type"] == "split":
            summary["by_split"][row["split"]] = clean_row
        elif row["group_type"] == "role":
            summary["by_role"][row["role"]] = clean_row
        else:
            summary["by_split_role"].setdefault(row["split"], {})[row["role"]] = clean_row

    return summary


def build_report(summary_json: dict[str, Any]) -> str:
    overall = summary_json["overall"]
    split_summary = summary_json["by_split"]
    role_summary = summary_json["by_role"]

    highest_full_agreement_split = max(
        split_summary.items(),
        key=lambda item: item[1]["rate_4_of_4"],
    )
    lowest_entropy_role = min(
        role_summary.items(),
        key=lambda item: item[1]["mean_entropy"],
    )

    lines = [
        "# Utterance-level Model Agreement",
        "",
        "## Metric Definition",
        "",
        "- `4_of_4`: 四个模型选择完全相同的 emoji。",
        "- `3_of_4`: 三个模型一致，另一个模型不同。",
        "- `2_of_4`: 最大一致票数为 2，包含 `2-2` 和 `2-1-1` 两种模式。",
        "- `all_different`: 四个模型全部不同。",
        "- `q(e|u)`: 对单条 utterance，把四个模型给该 emoji 的 confidence 相加后归一化得到的 soft label。",
        "- `entropy`: 对 `q(e|u)` 计算 Shannon entropy，底数为 `log2`。",
        "",
        "## Overall",
        "",
        f"- Utterances analyzed: `{overall['utterance_count']}`",
        f"- `4_of_4` rate: `{overall['rate_4_of_4']:.4f}`",
        f"- `3_of_4` rate: `{overall['rate_3_of_4']:.4f}`",
        f"- `2_of_4` rate: `{overall['rate_2_of_4']:.4f}`",
        f"- `all_different` rate: `{overall['rate_all_different']:.4f}`",
        f"- Mean entropy: `{overall['mean_entropy']:.4f}`",
        f"- Mean soft-label top1 prob: `{overall['mean_soft_label_top1_prob']:.4f}`",
        "",
        "## By Split",
        "",
        f"- Highest `4_of_4` split: `{highest_full_agreement_split[0]}` with rate `{highest_full_agreement_split[1]['rate_4_of_4']:.4f}`",
    ]

    for split in SPLITS:
        split_row = split_summary[split]
        lines.append(
            f"- `{split}`: 4/4=`{split_row['rate_4_of_4']:.4f}`, 3/4=`{split_row['rate_3_of_4']:.4f}`, 2/4=`{split_row['rate_2_of_4']:.4f}`, all-different=`{split_row['rate_all_different']:.4f}`, mean-entropy=`{split_row['mean_entropy']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## By Role",
            "",
            f"- Lower-entropy role: `{lowest_entropy_role[0]}` with mean entropy `{lowest_entropy_role[1]['mean_entropy']:.4f}`",
        ]
    )

    for role in ("A", "B"):
        role_row = role_summary[role]
        lines.append(
            f"- `{role}`: 4/4=`{role_row['rate_4_of_4']:.4f}`, 3/4=`{role_row['rate_3_of_4']:.4f}`, 2/4=`{role_row['rate_2_of_4']:.4f}`, all-different=`{role_row['rate_all_different']:.4f}`, mean-entropy=`{role_row['mean_entropy']:.4f}`"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `agreement_summary.csv`: 总体、按 split、按 role、按 split+role 的汇总指标。",
            "- `agreement_by_split.csv`: 单独导出的 split 级汇总。",
            "- `agreement_by_role.csv`: 单独导出的 role 级汇总。",
            "- `agreement_by_split_role.csv`: 单独导出的 split+role 级汇总。",
            "- `agreement_summary.json`: 与 CSV 对应的层级化汇总结果。",
            "- `utterance_soft_labels.jsonl`: 每条 utterance 的四模型原始标签、confidence-weighted soft label 与 entropy。",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute utterance-level emoji agreement across four models.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing preprocessed split JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for agreement analysis outputs.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    group_aggregators = {
        ("overall", "all", "all"): new_aggregator(),
    }
    for split in SPLITS:
        group_aggregators[("split", split, "all")] = new_aggregator()
        for role in ("A", "B"):
            group_aggregators[("split_role", split, role)] = new_aggregator()
    for role in ("A", "B"):
        group_aggregators[("role", "all", role)] = new_aggregator()

    models_reference: list[str] | None = None
    dialogue_count = 0
    processed_utterances = 0
    skipped_turns = 0
    soft_label_path = args.output_dir / "utterance_soft_labels.jsonl"

    with soft_label_path.open("w", encoding="utf-8") as soft_label_file:
        for split in SPLITS:
            split_path = args.input_dir / f"{split}.json"
            dialogues = load_json(split_path)

            for dialogue in dialogues:
                dialogue_count += 1
                models = dialogue.get("available_models") or models_reference
                if models is None:
                    raise ValueError("Could not determine model list from input data.")
                models = list(models)

                if models_reference is None:
                    models_reference = models

                for turn in dialogue.get("turns", []):
                    emoji_map = turn.get("emoji", {})
                    confidence_map = turn.get("confidence", {})

                    if any(emoji_map.get(model) is None or confidence_map.get(model) is None for model in models):
                        skipped_turns += 1
                        continue

                    emojis = [str(emoji_map[model]) for model in models]
                    bucket, pattern, majority_vote_count, majority_emojis, unique_emoji_count = classify_agreement(emojis)
                    soft_label, entropy, top1_emoji, top1_prob = build_soft_label(emoji_map, confidence_map, models)
                    majority_vote_fraction = round(majority_vote_count / len(models), 8)

                    output_row = {
                        "dialogue_id": dialogue["dialogue_id"],
                        "split": split,
                        "turn_id": turn["turn_id"],
                        "turn_index": turn["turn_index"],
                        "role": turn["role"],
                        "utterance": turn["utterance"],
                        "models": models,
                        "agreement_bucket": bucket,
                        "agreement_pattern": pattern,
                        "unique_emoji_count": unique_emoji_count,
                        "majority_vote_count": majority_vote_count,
                        "majority_vote_fraction": majority_vote_fraction,
                        "majority_emojis": majority_emojis,
                        "model_emojis": {model: emoji_map[model] for model in models},
                        "model_confidences": {model: confidence_map[model] for model in models},
                        "soft_label": soft_label,
                        "soft_label_top1": top1_emoji,
                        "soft_label_top1_prob": top1_prob,
                        "soft_label_entropy": entropy,
                    }
                    soft_label_file.write(json.dumps(output_row, ensure_ascii=False) + "\n")

                    processed_utterances += 1
                    role = turn["role"]
                    update_aggregator(
                        group_aggregators[("overall", "all", "all")],
                        bucket,
                        pattern,
                        entropy,
                        majority_vote_fraction,
                        top1_prob,
                        unique_emoji_count,
                    )
                    update_aggregator(
                        group_aggregators[("split", split, "all")],
                        bucket,
                        pattern,
                        entropy,
                        majority_vote_fraction,
                        top1_prob,
                        unique_emoji_count,
                    )
                    update_aggregator(
                        group_aggregators[("role", "all", role)],
                        bucket,
                        pattern,
                        entropy,
                        majority_vote_fraction,
                        top1_prob,
                        unique_emoji_count,
                    )
                    update_aggregator(
                        group_aggregators[("split_role", split, role)],
                        bucket,
                        pattern,
                        entropy,
                        majority_vote_fraction,
                        top1_prob,
                        unique_emoji_count,
                    )

    summary_rows = build_summary_rows(group_aggregators)
    summary_fieldnames = [
        "group_type",
        "split",
        "role",
        "utterance_count",
        "count_4_of_4",
        "count_3_of_4",
        "count_2_of_4",
        "count_all_different",
        "rate_4_of_4",
        "rate_3_of_4",
        "rate_2_of_4",
        "rate_all_different",
        "pattern_4_count",
        "pattern_3_1_count",
        "pattern_2_2_count",
        "pattern_2_1_1_count",
        "pattern_1_1_1_1_count",
        "pattern_4_rate",
        "pattern_3_1_rate",
        "pattern_2_2_rate",
        "pattern_2_1_1_rate",
        "pattern_1_1_1_1_rate",
        "mean_majority_vote_fraction",
        "mean_soft_label_top1_prob",
        "mean_entropy",
        "median_entropy",
        "mean_unique_emoji_count",
    ]
    write_csv(args.output_dir / "agreement_summary.csv", summary_fieldnames, summary_rows)
    write_csv(
        args.output_dir / "agreement_by_split.csv",
        summary_fieldnames,
        [row for row in summary_rows if row["group_type"] == "split"],
    )
    write_csv(
        args.output_dir / "agreement_by_role.csv",
        summary_fieldnames,
        [row for row in summary_rows if row["group_type"] == "role"],
    )
    write_csv(
        args.output_dir / "agreement_by_split_role.csv",
        summary_fieldnames,
        [row for row in summary_rows if row["group_type"] == "split_role"],
    )

    metadata = {
        "input_dir": str(args.input_dir),
        "input_splits": list(SPLITS),
        "models": models_reference or [],
        "dialogues_processed": dialogue_count,
        "utterances_processed": processed_utterances,
        "skipped_turns_missing_labels": skipped_turns,
        "agreement_bucket_definition": {
            "4_of_4": "all four models chose the same emoji",
            "3_of_4": "three models agreed on one emoji",
            "2_of_4": "largest agreement block size is 2, covering patterns 2-2 and 2-1-1",
            "all_different": "all four models chose different emoji",
        },
        "soft_label_definition": "q(e|u) = normalized sum of model confidence scores assigned to emoji e for utterance u",
        "entropy_definition": "Shannon entropy over q(e|u) using log2",
    }
    summary_json = build_summary_json(summary_rows, metadata)
    (args.output_dir / "agreement_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "agreement_report.md").write_text(
        build_report(summary_json),
        encoding="utf-8",
    )

    print(f"Wrote utterance agreement outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
