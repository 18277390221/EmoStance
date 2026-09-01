from __future__ import annotations

import csv
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .discovery import Candidate


VALIDATION_TOLERANCE = 1e-8


@dataclass
class UtteranceRecord:
    utterance_id: str
    dialogue_id: str
    turn_id: str
    turn_index: str
    split: str
    role: str
    text: str
    top1_emoji: str
    top1_prob: float
    entropy: float
    agreement_type: str
    agreement_pattern: str
    nonzero_emoji_count: int
    effective_nonzero_emoji: float
    utterance_mean_confidence: float | None


@dataclass
class SoftLabelEntry:
    utterance_id: str
    emoji: str
    soft_prob: float
    row_mean_confidence: float | None
    row_support_model_count: int | None


@dataclass
class CanonicalData:
    utterances: list[UtteranceRecord]
    entries: list[SoftLabelEntry]
    emojis: list[str]
    observed_counts: dict[str, int]
    source_path: str
    source_mode: str


def normalize_split_name(value: Any) -> str:
    split = str(value).strip().lower()
    return "dev" if split in {"valid", "validation"} else split


def split_counts(data: CanonicalData) -> dict[str, int]:
    counts = Counter(normalize_split_name(utterance.split) for utterance in data.utterances)
    return dict(sorted(counts.items()))


def filter_canonical_data_by_splits(data: CanonicalData, splits: Iterable[str]) -> CanonicalData:
    requested = {normalize_split_name(split) for split in splits if str(split).strip()}
    if not requested:
        return data

    utterances = [
        utterance
        for utterance in data.utterances
        if normalize_split_name(utterance.split) in requested
    ]
    kept_ids = {utterance.utterance_id for utterance in utterances}
    entries = [entry for entry in data.entries if entry.utterance_id in kept_ids]
    observed_counts: dict[str, int] = {}
    for entry in entries:
        observed_counts[entry.emoji] = observed_counts.get(entry.emoji, 0) + (
            entry.row_support_model_count or 1
        )

    if not utterances:
        available = ", ".join(split_counts(data)) or "none"
        raise ValueError(
            "No utterances remain after split filtering. "
            f"Requested {sorted(requested)} but available splits are: {available}."
        )
    if not observed_counts:
        raise ValueError(
            "No emoji labels remain after split filtering. "
            f"Requested splits: {sorted(requested)}."
        )

    emojis = sorted(observed_counts, key=lambda emoji: (-observed_counts[emoji], emoji))
    return CanonicalData(
        utterances=utterances,
        entries=entries,
        emojis=emojis,
        observed_counts=observed_counts,
        source_path=data.source_path,
        source_mode=f"{data.source_mode};split_filter={','.join(sorted(requested))}",
    )


def make_utterance_id(split: str, dialogue_id: str, turn_id: str | int) -> str:
    return f"{split}|{dialogue_id}|{turn_id}"


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def compute_entropy(probabilities: Iterable[float]) -> float:
    entropy = 0.0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


def effective_nonzero_from_entropy(entropy: float) -> float:
    return 2.0**entropy


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in weights.values() if value > 0)
    if total <= 0:
        return {}
    return {
        emoji: value / total
        for emoji, value in sorted(weights.items(), key=lambda item: (-item[1], item[0]))
        if value > 0
    }


def classify_agreement(emojis: list[str]) -> tuple[str, str]:
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
    return "other", pattern


def soft_label_from_model_annotations(
    emoji_map: dict[str, Any],
    confidence_map: dict[str, Any],
    models: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, int], float | None]:
    weight_by_emoji: dict[str, float] = {}
    confidence_sum_by_emoji: dict[str, float] = {}
    support_by_emoji: dict[str, int] = {}
    all_confidences: list[float] = []

    for model in models:
        emoji = emoji_map.get(model)
        confidence = parse_float(confidence_map.get(model))
        if emoji is None or confidence is None or confidence <= 0:
            continue
        emoji_text = str(emoji)
        weight_by_emoji[emoji_text] = weight_by_emoji.get(emoji_text, 0.0) + confidence
        confidence_sum_by_emoji[emoji_text] = confidence_sum_by_emoji.get(emoji_text, 0.0) + confidence
        support_by_emoji[emoji_text] = support_by_emoji.get(emoji_text, 0) + 1
        all_confidences.append(confidence)

    probabilities = normalize_weights(weight_by_emoji)
    mean_conf_by_emoji = {
        emoji: confidence_sum_by_emoji[emoji] / support_by_emoji[emoji]
        for emoji in probabilities
    }
    utterance_mean_confidence = (
        sum(all_confidences) / len(all_confidences) if all_confidences else None
    )
    return probabilities, mean_conf_by_emoji, support_by_emoji, utterance_mean_confidence


def validate_required_columns(path: Path, required: set[str]) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    missing = sorted(required - set(header))
    return missing


def build_from_existing_soft_csv(path: Path) -> CanonicalData:
    missing = validate_required_columns(
        path,
        {
            "dialogue_id",
            "turn_id",
            "split",
            "role",
            "text",
            "emoji",
            "soft_prob",
        },
    )
    if missing:
        raise ValueError(f"Soft-label table {path} is missing required columns: {missing}")

    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utterance_id = row.get("utterance_id") or make_utterance_id(
                row["split"], row["dialogue_id"], row["turn_id"]
            )
            group = groups.setdefault(
                utterance_id,
                {
                    "metadata": row,
                    "rows": [],
                },
            )
            group["rows"].append(row)

    utterances: list[UtteranceRecord] = []
    entries: list[SoftLabelEntry] = []
    observed_counts: dict[str, int] = {}

    for utterance_id, group in groups.items():
        rows: list[dict[str, Any]] = group["rows"]
        metadata = group["metadata"]
        raw_probs = {
            str(row["emoji"]): parse_float(row.get("soft_prob"), 0.0) or 0.0
            for row in rows
        }
        probs = normalize_weights(raw_probs)
        if not probs:
            raise ValueError(f"Utterance {utterance_id} has no positive soft-label mass.")

        top1_emoji, top1_prob = next(iter(probs.items()))
        entropy = compute_entropy(probs.values())
        nonzero_count = len(probs)
        effective_nonzero = effective_nonzero_from_entropy(entropy)

        support_total = 0
        confidence_total = 0.0
        has_support_confidence = False
        for row in rows:
            mean_conf = parse_float(row.get("mean_confidence"))
            support_count = parse_int(row.get("support_model_count"))
            if mean_conf is not None and support_count is not None:
                confidence_total += mean_conf * support_count
                support_total += support_count
                has_support_confidence = True

        if has_support_confidence and support_total > 0:
            utterance_mean_confidence: float | None = confidence_total / support_total
        else:
            weighted_confidences = [
                (parse_float(row.get("mean_confidence")), probs.get(str(row["emoji"]), 0.0))
                for row in rows
            ]
            available = [(conf, prob) for conf, prob in weighted_confidences if conf is not None]
            utterance_mean_confidence = (
                sum(float(conf) * prob for conf, prob in available) / sum(prob for _, prob in available)
                if available and sum(prob for _, prob in available) > 0
                else None
            )

        utterances.append(
            UtteranceRecord(
                utterance_id=utterance_id,
                dialogue_id=str(metadata.get("dialogue_id", "")),
                turn_id=str(metadata.get("turn_id", "")),
                turn_index=str(metadata.get("turn_index", metadata.get("turn_id", ""))),
                split=str(metadata.get("split", "")),
                role=str(metadata.get("role", "")),
                text=str(metadata.get("text", "")),
                top1_emoji=top1_emoji,
                top1_prob=top1_prob,
                entropy=entropy,
                agreement_type=str(metadata.get("agreement_type", "")),
                agreement_pattern=str(metadata.get("agreement_pattern", "")),
                nonzero_emoji_count=nonzero_count,
                effective_nonzero_emoji=effective_nonzero,
                utterance_mean_confidence=utterance_mean_confidence,
            )
        )

        row_by_emoji = {str(row["emoji"]): row for row in rows}
        for emoji, prob in probs.items():
            row = row_by_emoji[emoji]
            support_count = parse_int(row.get("support_model_count"))
            mean_conf = parse_float(row.get("mean_confidence"))
            entries.append(
                SoftLabelEntry(
                    utterance_id=utterance_id,
                    emoji=emoji,
                    soft_prob=prob,
                    row_mean_confidence=mean_conf,
                    row_support_model_count=support_count,
                )
            )
            observed_counts[emoji] = observed_counts.get(emoji, 0) + (support_count or 1)

    emojis = sorted(observed_counts, key=lambda emoji: (-observed_counts[emoji], emoji))
    return CanonicalData(
        utterances=utterances,
        entries=entries,
        emojis=emojis,
        observed_counts=observed_counts,
        source_path=str(path),
        source_mode="validated_existing_soft_label_csv",
    )


def iter_raw_jsonl_utterances(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            models = [str(model) for model in row.get("models", [])]
            emoji_map = row.get("model_emojis", {})
            confidence_map = row.get("model_confidences", {})
            if not models or not emoji_map or not confidence_map:
                continue
            yield {
                "dialogue_id": row["dialogue_id"],
                "turn_id": row["turn_id"],
                "turn_index": row.get("turn_index", row["turn_id"]),
                "split": row["split"],
                "role": row["role"],
                "text": row.get("utterance", row.get("text", "")),
                "models": models,
                "emoji_map": emoji_map,
                "confidence_map": confidence_map,
            }


def iter_multimodel_json_utterances(paths: tuple[Path, ...]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            dialogues = json.load(f)
        if not isinstance(dialogues, list):
            continue
        for dialogue in dialogues:
            models = [str(model) for model in dialogue.get("available_models", [])]
            for turn in dialogue.get("turns", []):
                emoji_map = turn.get("emoji", {})
                confidence_map = turn.get("confidence", {})
                if not models or not isinstance(emoji_map, dict) or not isinstance(confidence_map, dict):
                    continue
                yield {
                    "dialogue_id": dialogue["dialogue_id"],
                    "turn_id": turn["turn_id"],
                    "turn_index": turn.get("turn_index", turn["turn_id"]),
                    "split": dialogue.get("split", turn.get("split", "")),
                    "role": turn.get("role", turn.get("speaker", "")),
                    "text": turn.get("utterance", ""),
                    "models": models,
                    "emoji_map": emoji_map,
                    "confidence_map": confidence_map,
                }


def iter_single_model_json_utterances(paths: tuple[Path, ...]) -> Iterable[dict[str, Any]]:
    grouped: OrderedDict[tuple[str, str, str], dict[str, Any]] = OrderedDict()
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            dialogues = json.load(f)
        if not isinstance(dialogues, list):
            continue
        model = path.parent.name
        fallback_split = path.name.split("_", 1)[0].lower()
        if fallback_split == "valid":
            fallback_split = "dev"
        for dialogue in dialogues:
            split = normalize_split_name(dialogue.get("split", fallback_split))
            dialogue_id = str(dialogue.get("dialogue_id", dialogue.get("id", "")))
            for index, turn in enumerate(dialogue.get("turns", [])):
                annotation = turn.get("emoji_annotation") or turn.get("annotation") or {}
                emoji = annotation.get("selected_emoji", annotation.get("emoji", annotation.get("label")))
                confidence = annotation.get("confidence", annotation.get("score", 1.0))
                if emoji is None:
                    continue
                turn_id = str(turn.get("turn_id", turn.get("turn", index)))
                key = (split, dialogue_id, turn_id)
                row = grouped.setdefault(
                    key,
                    {
                        "dialogue_id": dialogue_id,
                        "turn_id": turn_id,
                        "turn_index": turn.get("turn_index", turn_id),
                        "split": split,
                        "role": turn.get("role", turn.get("speaker", "")),
                        "text": turn.get("utterance", turn.get("text", "")),
                        "models": [],
                        "emoji_map": {},
                        "confidence_map": {},
                    },
                )
                if model not in row["models"]:
                    row["models"].append(model)
                row["emoji_map"][model] = emoji
                row["confidence_map"][model] = confidence
    yield from grouped.values()


def build_from_raw_annotations(candidate: Candidate) -> CanonicalData:
    if candidate.kind == "utterance_soft_label_jsonl":
        raw_iter = iter_raw_jsonl_utterances(candidate.paths[0])
        source_mode = "reconstructed_from_jsonl_model_annotations"
    elif candidate.kind == "multimodel_dialogue_json_group":
        raw_iter = iter_multimodel_json_utterances(candidate.paths)
        source_mode = "reconstructed_from_multimodel_dialogue_json"
    elif candidate.kind == "single_model_dialogue_json_group":
        raw_iter = iter_single_model_json_utterances(candidate.paths)
        source_mode = "reconstructed_from_single_model_dialogue_json_group"
    else:
        raise ValueError(f"Unsupported raw annotation candidate kind: {candidate.kind}")

    utterances: list[UtteranceRecord] = []
    entries: list[SoftLabelEntry] = []
    observed_counts: dict[str, int] = {}

    for row in raw_iter:
        models = row["models"]
        emoji_map = row["emoji_map"]
        confidence_map = row["confidence_map"]
        probabilities, mean_conf_by_emoji, support_by_emoji, utterance_mean_confidence = (
            soft_label_from_model_annotations(emoji_map, confidence_map, models)
        )
        if not probabilities:
            continue
        agreement_type, agreement_pattern = classify_agreement(
            [str(emoji_map[model]) for model in models if emoji_map.get(model) is not None]
        )
        top1_emoji, top1_prob = next(iter(probabilities.items()))
        entropy = compute_entropy(probabilities.values())
        utterance_id = make_utterance_id(row["split"], row["dialogue_id"], row["turn_id"])
        utterances.append(
            UtteranceRecord(
                utterance_id=utterance_id,
                dialogue_id=str(row["dialogue_id"]),
                turn_id=str(row["turn_id"]),
                turn_index=str(row["turn_index"]),
                split=str(row["split"]),
                role=str(row["role"]),
                text=str(row["text"]),
                top1_emoji=top1_emoji,
                top1_prob=top1_prob,
                entropy=entropy,
                agreement_type=agreement_type,
                agreement_pattern=agreement_pattern,
                nonzero_emoji_count=len(probabilities),
                effective_nonzero_emoji=effective_nonzero_from_entropy(entropy),
                utterance_mean_confidence=utterance_mean_confidence,
            )
        )
        for emoji, prob in probabilities.items():
            support_count = support_by_emoji.get(emoji)
            entries.append(
                SoftLabelEntry(
                    utterance_id=utterance_id,
                    emoji=emoji,
                    soft_prob=prob,
                    row_mean_confidence=mean_conf_by_emoji.get(emoji),
                    row_support_model_count=support_count,
                )
            )
            observed_counts[emoji] = observed_counts.get(emoji, 0) + (support_count or 1)

    if not utterances:
        raise ValueError(f"No utterances could be reconstructed from {candidate.paths}.")

    emojis = sorted(observed_counts, key=lambda emoji: (-observed_counts[emoji], emoji))
    return CanonicalData(
        utterances=utterances,
        entries=entries,
        emojis=emojis,
        observed_counts=observed_counts,
        source_path=", ".join(str(path) for path in candidate.paths),
        source_mode=source_mode,
    )


def write_canonical_outputs(
    data: CanonicalData,
    table_path: Path,
    validation_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    utterance_by_id = {utterance.utterance_id: utterance for utterance in data.utterances}
    prob_sums: dict[str, float] = {utterance.utterance_id: 0.0 for utterance in data.utterances}
    for entry in data.entries:
        prob_sums[entry.utterance_id] += entry.soft_prob

    failed = [
        {
            "utterance_id": utterance_id,
            "prob_sum": prob_sum,
            "abs_error": abs(prob_sum - 1.0),
        }
        for utterance_id, prob_sum in prob_sums.items()
        if abs(prob_sum - 1.0) > VALIDATION_TOLERANCE
    ]
    max_abs_error = max((abs(prob_sum - 1.0) for prob_sum in prob_sums.values()), default=0.0)

    fieldnames = [
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "turn_index",
        "split",
        "role",
        "text",
        "emoji",
        "soft_prob",
        "row_mean_confidence",
        "row_support_model_count",
        "top1_emoji",
        "top1_prob",
        "entropy",
        "agreement_type",
        "agreement_pattern",
        "nonzero_emoji_count",
        "effective_nonzero_emoji",
        "utterance_mean_confidence",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data.entries:
            utterance = utterance_by_id[entry.utterance_id]
            writer.writerow(
                {
                    "utterance_id": utterance.utterance_id,
                    "dialogue_id": utterance.dialogue_id,
                    "turn_id": utterance.turn_id,
                    "turn_index": utterance.turn_index,
                    "split": utterance.split,
                    "role": utterance.role,
                    "text": utterance.text,
                    "emoji": entry.emoji,
                    "soft_prob": f"{entry.soft_prob:.15f}",
                    "row_mean_confidence": (
                        f"{entry.row_mean_confidence:.12f}"
                        if entry.row_mean_confidence is not None
                        else ""
                    ),
                    "row_support_model_count": (
                        entry.row_support_model_count
                        if entry.row_support_model_count is not None
                        else ""
                    ),
                    "top1_emoji": utterance.top1_emoji,
                    "top1_prob": f"{utterance.top1_prob:.15f}",
                    "entropy": f"{utterance.entropy:.15f}",
                    "agreement_type": utterance.agreement_type,
                    "agreement_pattern": utterance.agreement_pattern,
                    "nonzero_emoji_count": utterance.nonzero_emoji_count,
                    "effective_nonzero_emoji": f"{utterance.effective_nonzero_emoji:.12f}",
                    "utterance_mean_confidence": (
                        f"{utterance.utterance_mean_confidence:.12f}"
                        if utterance.utterance_mean_confidence is not None
                        else ""
                    ),
                }
            )

    entropy_values = [utterance.entropy for utterance in data.utterances]
    top1_values = [utterance.top1_prob for utterance in data.utterances]
    nonzero_values = [utterance.nonzero_emoji_count for utterance in data.utterances]
    mean_conf_values = [
        utterance.utterance_mean_confidence
        for utterance in data.utterances
        if utterance.utterance_mean_confidence is not None
    ]
    validation = {
        "source_mode": data.source_mode,
        "source_path": data.source_path,
        "split_counts": split_counts(data),
        "utterance_count": len(data.utterances),
        "row_count": len(data.entries),
        "observed_emoji_count": len(data.emojis),
        "probability_sum": {
            "all_utterances_valid": not failed,
            "tolerance": VALIDATION_TOLERANCE,
            "max_abs_error": max_abs_error,
            "failed_count": len(failed),
            "failed_examples": failed[:20],
        },
        "utterance_level_stats": {
            "mean_top1_prob": sum(top1_values) / len(top1_values),
            "mean_entropy": sum(entropy_values) / len(entropy_values),
            "mean_nonzero_emoji_count": sum(nonzero_values) / len(nonzero_values),
            "mean_effective_nonzero_emoji": sum(
                utterance.effective_nonzero_emoji for utterance in data.utterances
            )
            / len(data.utterances),
            "mean_utterance_confidence": (
                sum(mean_conf_values) / len(mean_conf_values) if mean_conf_values else None
            ),
        },
        "row_level_stats": {
            "mean_soft_prob": sum(entry.soft_prob for entry in data.entries) / len(data.entries),
            "rows_with_row_mean_confidence": sum(
                1 for entry in data.entries if entry.row_mean_confidence is not None
            ),
            "rows_with_support_model_count": sum(
                1 for entry in data.entries if entry.row_support_model_count is not None
            ),
        },
    }
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Canonical Soft-label Validation",
        "",
        f"- Source mode: `{data.source_mode}`",
        f"- Source path: `{data.source_path}`",
        f"- Split counts: `{validation['split_counts']}`",
        f"- Utterances: `{len(data.utterances)}`",
        f"- Utterance-emoji rows: `{len(data.entries)}`",
        f"- Observed emojis: `{len(data.emojis)}`",
        f"- Probability sums valid: `{not failed}`",
        f"- Max probability-sum absolute error: `{max_abs_error:.12g}`",
        "",
        "## Utterance-level Statistics",
        "",
        f"- Mean top1 probability: `{validation['utterance_level_stats']['mean_top1_prob']:.6f}`",
        f"- Mean entropy: `{validation['utterance_level_stats']['mean_entropy']:.6f}`",
        f"- Mean non-zero emoji count: `{validation['utterance_level_stats']['mean_nonzero_emoji_count']:.6f}`",
        f"- Mean effective non-zero emoji count: `{validation['utterance_level_stats']['mean_effective_nonzero_emoji']:.6f}`",
        "",
        "## Row-level Statistics",
        "",
        f"- Mean row soft probability: `{validation['row_level_stats']['mean_soft_prob']:.6f}`",
        f"- Rows with row-level mean confidence: `{validation['row_level_stats']['rows_with_row_mean_confidence']}`",
        f"- Rows with support model count: `{validation['row_level_stats']['rows_with_support_model_count']}`",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return validation
