import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOFT_LABEL_INPUT = REPO_ROOT / "src/utterance_agreement/outputs/utterance_soft_labels.jsonl"
DEFAULT_PRE_DATA_DIR = REPO_ROOT / "pre_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src/soft_label_table/outputs"
SPLITS = ("train", "valid", "test")
VALIDATION_TOLERANCE = 1e-9


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def classify_agreement_pattern(emojis: list[str]) -> tuple[str, str]:
    counts = Counter(emojis)
    pattern = "-".join(str(count) for count in sorted(counts.values(), reverse=True))

    if pattern == "4":
        return "4_of_4", pattern
    if pattern == "3-1":
        return "3_of_4", pattern
    if pattern in {"2-2", "2-1-1"}:
        return "2_of_4", pattern
    if pattern == "1-1-1-1":
        return "all_different", pattern
    raise ValueError(f"Unexpected agreement pattern: {pattern}")


def compute_soft_label_from_annotations(
    emoji_map: dict[str, Any],
    confidence_map: dict[str, Any],
    models: list[str],
) -> tuple[dict[str, float], dict[str, float], str | None, float, float]:
    weight_by_emoji: dict[str, float] = {}
    confidence_values_by_emoji: dict[str, list[float]] = {}

    for model in models:
        emoji = emoji_map.get(model)
        confidence = confidence_map.get(model)
        if emoji is None or confidence is None:
            continue

        confidence_value = float(confidence)
        if confidence_value <= 0:
            continue

        weight_by_emoji[emoji] = weight_by_emoji.get(emoji, 0.0) + confidence_value
        confidence_values_by_emoji.setdefault(emoji, []).append(confidence_value)

    total_weight = sum(weight_by_emoji.values())
    if total_weight <= 0:
        return {}, {}, None, 0.0, 0.0

    ordered_items = sorted(weight_by_emoji.items(), key=lambda item: (-item[1], item[0]))
    soft_label = {emoji: weight / total_weight for emoji, weight in ordered_items}
    mean_confidence = {
        emoji: sum(values) / len(values)
        for emoji, values in confidence_values_by_emoji.items()
    }

    top1_emoji = next(iter(soft_label))
    top1_prob = soft_label[top1_emoji]
    entropy = 0.0
    for prob in soft_label.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)

    return soft_label, mean_confidence, top1_emoji, top1_prob, entropy


def iter_existing_soft_labels(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            models = list(row.get("models", []))
            emoji_map = row.get("model_emojis", {})
            confidence_map = row.get("model_confidences", {})

            if not models or not emoji_map or not confidence_map:
                raise ValueError(
                    "Existing soft-label file is missing raw annotation fields needed for standardization."
                )

            agreement_type, agreement_pattern = classify_agreement_pattern(
                [str(emoji_map[model]) for model in models]
            )
            soft_label, mean_confidence, top1_emoji, top1_prob, entropy = compute_soft_label_from_annotations(
                emoji_map,
                confidence_map,
                models,
            )

            yield {
                "dialogue_id": row["dialogue_id"],
                "turn_id": row["turn_id"],
                "turn_index": row.get("turn_index", row["turn_id"]),
                "split": row["split"],
                "role": row["role"],
                "text": row.get("utterance", ""),
                "models": models,
                "emoji_map": emoji_map,
                "confidence_map": confidence_map,
                "agreement_type": row.get("agreement_bucket", agreement_type),
                "agreement_pattern": row.get("agreement_pattern", agreement_pattern),
                "soft_label": soft_label,
                "mean_confidence_by_emoji": mean_confidence,
                "top1_emoji": top1_emoji,
                "top1_prob": top1_prob,
                "entropy": entropy,
            }


def iter_pre_data(pre_data_dir: Path) -> Iterable[dict[str, Any]]:
    for split in SPLITS:
        path = pre_data_dir / f"{split}.json"
        dialogues = load_json(path)
        for dialogue in dialogues:
            models = list(dialogue.get("available_models", []))
            if not models:
                raise ValueError(f"Missing available_models in {path}")

            for turn in dialogue.get("turns", []):
                emoji_map = turn.get("emoji", {})
                confidence_map = turn.get("confidence", {})

                if any(emoji_map.get(model) is None or confidence_map.get(model) is None for model in models):
                    continue

                agreement_type, agreement_pattern = classify_agreement_pattern(
                    [str(emoji_map[model]) for model in models]
                )
                soft_label, mean_confidence, top1_emoji, top1_prob, entropy = compute_soft_label_from_annotations(
                    emoji_map,
                    confidence_map,
                    models,
                )

                yield {
                    "dialogue_id": dialogue["dialogue_id"],
                    "turn_id": turn["turn_id"],
                    "turn_index": turn.get("turn_index", turn["turn_id"]),
                    "split": split,
                    "role": turn["role"],
                    "text": turn.get("utterance", ""),
                    "models": models,
                    "emoji_map": emoji_map,
                    "confidence_map": confidence_map,
                    "agreement_type": agreement_type,
                    "agreement_pattern": agreement_pattern,
                    "soft_label": soft_label,
                    "mean_confidence_by_emoji": mean_confidence,
                    "top1_emoji": top1_emoji,
                    "top1_prob": top1_prob,
                    "entropy": entropy,
                }


def build_report(
    source_path: Path,
    utterance_count: int,
    row_count: int,
    avg_nonzero_emoji: float,
    avg_top1_prob: float,
    avg_entropy: float,
    role_entropy: dict[str, float],
    validation_summary: dict[str, Any],
) -> str:
    role_a_entropy = role_entropy.get("A", 0.0)
    role_b_entropy = role_entropy.get("B", 0.0)
    difference = role_a_entropy - role_b_entropy

    if abs(difference) < 1e-12:
        role_statement = "A/B 两个角色的平均 entropy 基本相同。"
    elif difference > 0:
        role_statement = (
            f"A/B 两个角色的平均 entropy 不同，A 更高（A={role_a_entropy:.6f}, B={role_b_entropy:.6f}, 差值={difference:.6f}）。"
        )
    else:
        role_statement = (
            f"A/B 两个角色的平均 entropy 不同，B 更高（A={role_a_entropy:.6f}, B={role_b_entropy:.6f}, 差值={difference:.6f}）。"
        )

    lines = [
        "# Standardized Soft-label Report",
        "",
        "## Source",
        "",
        f"- Source file: `{source_path}`",
        f"- Utterances processed: `{utterance_count}`",
        f"- Utterance-emoji rows generated: `{row_count}`",
        "",
        "## Summary",
        "",
        f"- 平均每条 utterance 的非零 emoji 数量：`{avg_nonzero_emoji:.6f}`",
        f"- 平均 `top1_prob`：`{avg_top1_prob:.6f}`",
        f"- 平均 entropy：`{avg_entropy:.6f}`",
        f"- {role_statement}",
        "",
        "## Validation",
        "",
        f"- Soft-prob sum validated: `{validation_summary['all_utterances_valid']}`",
        f"- Utterances checked: `{validation_summary['utterances_checked']}`",
        f"- Failed utterances: `{validation_summary['utterances_failed']}`",
        f"- Max absolute error: `{validation_summary['max_abs_error']:.12f}`",
        f"- Tolerance: `{validation_summary['tolerance']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standardized utterance-level soft-label table.")
    parser.add_argument(
        "--soft-label-input",
        type=Path,
        default=DEFAULT_SOFT_LABEL_INPUT,
        help="Preferred existing soft-label JSONL file.",
    )
    parser.add_argument(
        "--pre-data-dir",
        type=Path,
        default=DEFAULT_PRE_DATA_DIR,
        help="Fallback directory containing preprocessed train/valid/test JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for standardized soft-label outputs.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    if args.soft_label_input.exists():
        source_path = args.soft_label_input
        utterance_iter = iter_existing_soft_labels(args.soft_label_input)
        source_mode = "existing_soft_label"
    else:
        source_path = args.pre_data_dir
        utterance_iter = iter_pre_data(args.pre_data_dir)
        source_mode = "reconstructed_from_pre_data"

    output_csv_path = args.output_dir / "standardized_soft_label_table.csv"
    validation_path = args.output_dir / "soft_label_validation.json"
    report_path = args.output_dir / "soft_label_report.md"

    fieldnames = [
        "dialogue_id",
        "turn_id",
        "turn_index",
        "split",
        "role",
        "text",
        "emoji",
        "soft_prob",
        "top1_emoji",
        "top1_prob",
        "entropy",
        "agreement_type",
        "agreement_pattern",
        "mean_confidence",
        "support_model_count",
    ]

    utterance_count = 0
    row_count = 0
    total_nonzero_emoji = 0
    total_top1_prob = 0.0
    total_entropy = 0.0
    role_entropy_values: dict[str, list[float]] = {"A": [], "B": []}
    failed_utterances: list[dict[str, Any]] = []
    max_abs_error = 0.0

    with output_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for utterance in utterance_iter:
            utterance_count += 1
            soft_label = utterance["soft_label"]
            confidence_map = utterance["confidence_map"]
            emoji_map = utterance["emoji_map"]
            probs_sum = sum(soft_label.values())
            abs_error = abs(probs_sum - 1.0)
            max_abs_error = max(max_abs_error, abs_error)

            if abs_error > VALIDATION_TOLERANCE:
                failed_utterances.append(
                    {
                        "dialogue_id": utterance["dialogue_id"],
                        "turn_id": utterance["turn_id"],
                        "split": utterance["split"],
                        "role": utterance["role"],
                        "soft_prob_sum": probs_sum,
                        "abs_error": abs_error,
                    }
                )

            nonzero_emoji_count = len(soft_label)
            total_nonzero_emoji += nonzero_emoji_count
            total_top1_prob += utterance["top1_prob"]
            total_entropy += utterance["entropy"]
            role_entropy_values.setdefault(utterance["role"], []).append(utterance["entropy"])

            support_counts = Counter(str(emoji_map[model]) for model in utterance["models"])
            ordered_soft_items = sorted(soft_label.items(), key=lambda item: (-item[1], item[0]))
            for emoji, soft_prob in ordered_soft_items:
                row_count += 1
                writer.writerow(
                    {
                        "dialogue_id": utterance["dialogue_id"],
                        "turn_id": utterance["turn_id"],
                        "turn_index": utterance["turn_index"],
                        "split": utterance["split"],
                        "role": utterance["role"],
                        "text": utterance["text"],
                        "emoji": emoji,
                        "soft_prob": f"{soft_prob:.12f}",
                        "top1_emoji": utterance["top1_emoji"],
                        "top1_prob": f"{utterance['top1_prob']:.12f}",
                        "entropy": f"{utterance['entropy']:.12f}",
                        "agreement_type": utterance["agreement_type"],
                        "agreement_pattern": utterance["agreement_pattern"],
                        "mean_confidence": f"{utterance['mean_confidence_by_emoji'][emoji]:.12f}",
                        "support_model_count": support_counts[emoji],
                    }
                )

    if utterance_count == 0:
        raise ValueError("No utterances were processed; cannot build standardized soft-label table.")

    validation_summary = {
        "source_mode": source_mode,
        "source_path": str(source_path),
        "utterances_checked": utterance_count,
        "utterances_failed": len(failed_utterances),
        "all_utterances_valid": len(failed_utterances) == 0,
        "max_abs_error": max_abs_error,
        "tolerance": VALIDATION_TOLERANCE,
        "failed_examples": failed_utterances[:20],
    }
    validation_path.write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    role_entropy = {
        role: mean(values) if values else 0.0
        for role, values in role_entropy_values.items()
    }

    report_path.write_text(
        build_report(
            source_path=source_path,
            utterance_count=utterance_count,
            row_count=row_count,
            avg_nonzero_emoji=total_nonzero_emoji / utterance_count,
            avg_top1_prob=total_top1_prob / utterance_count,
            avg_entropy=total_entropy / utterance_count,
            role_entropy=role_entropy,
            validation_summary=validation_summary,
        ),
        encoding="utf-8",
    )

    print(f"Wrote standardized soft-label outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
