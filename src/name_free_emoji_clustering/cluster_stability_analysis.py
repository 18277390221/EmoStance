from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from .soft_membership import cluster_sort_key


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
VALIDATION_TOLERANCE = 1e-6
AGREEMENT_BUCKETS = ("4_of_4", "3_of_4", "2_of_4", "all_different")
AGREEMENT_RANK = {"all_different": 1, "2_of_4": 2, "3_of_4": 3, "4_of_4": 4}
REPRESENTATIONS = ("emoji", "cluster_raw", "cluster_sharp")


@dataclass(frozen=True)
class InputCandidate:
    path: Path
    kind: str
    score: int
    fields: tuple[str, ...]
    rationale: str


@dataclass
class UtteranceSoftLabels:
    utterance_id: str
    dialogue_id: str
    turn_id: str
    turn_index: str
    split: str
    role: str
    text: str
    emoji_probs: dict[str, float] = field(default_factory=dict)
    agreement_type: str = ""
    agreement_pattern: str = ""


@dataclass(frozen=True)
class FourModelLabels:
    utterance_id: str
    split: str
    role: str
    model_emojis: tuple[str, ...]
    emoji_agreement_bucket: str
    agreement_pattern: str


@dataclass(frozen=True)
class MembershipMatrix:
    path: Path
    version: str
    column: str
    membership: dict[str, dict[str, float]]
    top_cluster: dict[str, str]
    clusters: tuple[str, ...]


@dataclass(frozen=True)
class RepresentationMetrics:
    utterance_id: str
    role: str
    split: str
    representation: str
    entropy_bits: float
    top1_prob: float
    nonzero_count: int
    agreement_bucket: str | None
    emoji_agreement_bucket: str | None
    entropy_delta_vs_emoji: float
    top1_delta_vs_emoji: float


@dataclass(frozen=True)
class AggregateRow:
    group_type: str
    group_value: str
    representation: str
    utterance_count: int
    agreement_available_count: int
    mean_entropy_bits: float
    mean_top1_prob: float
    mean_nonzero_count: float
    mean_entropy_delta_vs_emoji: float
    mean_top1_delta_vs_emoji: float
    agreement_improved_rate: float | None
    agreement_degraded_rate: float | None
    agreement_unchanged_rate: float | None
    bucket_counts: dict[str, int]
    bucket_rates: dict[str, float]


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


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def make_utterance_id(row: dict[str, Any]) -> str:
    utterance_id = row.get("utterance_id")
    if utterance_id:
        return str(utterance_id)
    return f"{row.get('split', '')}|{row.get('dialogue_id', '')}|{row.get('turn_id', '')}"


def entropy_bits(probabilities: Iterable[float]) -> float:
    result = 0.0
    for probability in probabilities:
        if probability > 0:
            result -= probability * math.log2(probability)
    return result


def top1_probability(probabilities: dict[str, float]) -> float:
    return max(probabilities.values()) if probabilities else 0.0


def agreement_bucket(labels: Iterable[str]) -> str | None:
    labels = [label for label in labels if label]
    if not labels:
        return None
    counts = sorted(Counter(labels).values(), reverse=True)
    if counts[0] >= 4:
        return "4_of_4"
    if counts[0] == 3:
        return "3_of_4"
    if counts[0] == 2:
        return "2_of_4"
    return "all_different"


def inspect_soft_label_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"emoji", "soft_prob", "role", "split"}.issubset(columns):
        return None
    if "utterance_id" not in columns and not {"dialogue_id", "turn_id"}.issubset(columns):
        return None

    score = 100
    path_text = str(path).lower()
    if "canonical" in path_text:
        score += 35
    if "soft_label" in path_text:
        score += 20
    if {"text", "agreement_type", "agreement_pattern"}.issubset(columns):
        score += 25
    if "utterance_id" in columns:
        score += 10
    if "utterance_mean_confidence" in columns:
        score += 5
    return InputCandidate(
        path=path,
        kind="utterance_soft_label_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has utterance identity, role/split/text, emoji, and soft_prob fields.",
    )


def inspect_membership_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"emoji", "cluster_id"}.issubset(columns):
        return None
    if not {"membership_raw", "membership_sharp"}.intersection(columns):
        return None

    score = 100
    path_text = str(path).lower()
    if "membership_raw" in columns:
        score += 25
    if "membership_sharp" in columns:
        score += 25
    if "soft_membership" in path_text:
        score += 30
    if path.name == "emoji_cluster_membership.csv":
        score += 20
    return InputCandidate(
        path=path,
        kind="emoji_cluster_membership_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has emoji-cluster membership columns.",
    )


def inspect_four_model_jsonl(path: Path) -> InputCandidate | None:
    if path.suffix.lower() != ".jsonl":
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            first_line = next(handle, "")
    except (OSError, UnicodeDecodeError):
        return None
    if not first_line:
        return None
    try:
        first = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    if not isinstance(first, dict) or not isinstance(first.get("model_emojis"), dict):
        return None
    if not {"dialogue_id", "turn_id", "split", "role"}.issubset(first):
        return None

    score = 100
    path_text = str(path).lower()
    if "utterance_agreement" in path_text:
        score += 35
    if "agreement_bucket" in first:
        score += 15
    if "agreement_pattern" in first:
        score += 10
    if "soft_label" in first:
        score += 5
    return InputCandidate(
        path=path,
        kind="four_model_label_jsonl",
        score=score,
        fields=tuple(first.keys()),
        rationale="JSONL has per-model emoji labels for each utterance.",
    )


def discover_inputs(root: Path) -> tuple[InputCandidate, InputCandidate, InputCandidate | None]:
    soft_candidates: list[InputCandidate] = []
    membership_candidates: list[InputCandidate] = []
    four_model_candidates: list[InputCandidate] = []

    for path in iter_files(root, {".csv"}):
        soft_candidate = inspect_soft_label_csv(path)
        if soft_candidate is not None:
            soft_candidates.append(soft_candidate)
        membership_candidate = inspect_membership_csv(path)
        if membership_candidate is not None:
            membership_candidates.append(membership_candidate)

    for path in iter_files(root, {".jsonl"}):
        four_model_candidate = inspect_four_model_jsonl(path)
        if four_model_candidate is not None:
            four_model_candidates.append(four_model_candidate)

    if not soft_candidates:
        raise FileNotFoundError(
            f"No utterance-level soft emoji label table found under {root}. "
            "Expected emoji, soft_prob, role, split, and utterance identity columns."
        )
    if not membership_candidates:
        raise FileNotFoundError(
            f"No emoji-cluster membership table found under {root}. "
            "Expected emoji, cluster_id, and membership_raw/membership_sharp columns."
        )

    def sort_key(candidate: InputCandidate) -> tuple[int, float, str]:
        return (
            candidate.score,
            candidate.path.stat().st_mtime if candidate.path.exists() else 0.0,
            str(candidate.path),
        )

    soft = sorted(soft_candidates, key=sort_key, reverse=True)[0]
    membership = sorted(membership_candidates, key=sort_key, reverse=True)[0]
    four_model = (
        sorted(four_model_candidates, key=sort_key, reverse=True)[0]
        if four_model_candidates
        else None
    )
    return soft, membership, four_model


def load_soft_label_table(path: Path) -> tuple[list[UtteranceSoftLabels], float]:
    groups: OrderedDict[str, UtteranceSoftLabels] = OrderedDict()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            utterance_id = make_utterance_id(row)
            group = groups.get(utterance_id)
            if group is None:
                group = UtteranceSoftLabels(
                    utterance_id=utterance_id,
                    dialogue_id=row.get("dialogue_id", ""),
                    turn_id=row.get("turn_id", ""),
                    turn_index=row.get("turn_index", row.get("turn_id", "")),
                    split=row.get("split", ""),
                    role=row.get("role", ""),
                    text=row.get("text", ""),
                    agreement_type=row.get("agreement_type", ""),
                    agreement_pattern=row.get("agreement_pattern", ""),
                )
                groups[utterance_id] = group
            emoji = row.get("emoji", "")
            probability = parse_float(row.get("soft_prob"))
            if emoji and probability is not None and probability > 0:
                group.emoji_probs[emoji] = group.emoji_probs.get(emoji, 0.0) + probability

    max_sum_error = 0.0
    utterances: list[UtteranceSoftLabels] = []
    for group in groups.values():
        total = sum(group.emoji_probs.values())
        if total <= 0:
            continue
        max_sum_error = max(max_sum_error, abs(total - 1.0))
        if abs(total - 1.0) > VALIDATION_TOLERANCE:
            group.emoji_probs = {
                emoji: probability / total
                for emoji, probability in group.emoji_probs.items()
            }
        utterances.append(group)

    if not utterances:
        raise ValueError(f"No usable soft emoji labels found in {path}.")
    return utterances, max_sum_error


def load_membership(path: Path, version: str, column: str) -> MembershipMatrix:
    membership: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Membership table {path} does not contain `{column}`.")
        for row in reader:
            emoji = row.get("emoji", "")
            cluster_id = row.get("cluster_id", "")
            probability = parse_float(row.get(column))
            if emoji and cluster_id and probability is not None and probability > 0:
                membership[emoji][cluster_id] = membership[emoji].get(cluster_id, 0.0) + probability

    if not membership:
        raise ValueError(f"No usable membership rows found in {path}.")

    top_cluster: dict[str, str] = {}
    clusters = tuple(
        sorted(
            {cluster_id for probs in membership.values() for cluster_id in probs},
            key=cluster_sort_key,
        )
    )
    for emoji, probabilities in membership.items():
        total = sum(probabilities.values())
        if abs(total - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Membership probabilities for an observed emoji sum to {total:.12f}, not 1.0."
            )
        top_cluster[emoji] = sorted(
            probabilities.items(),
            key=lambda item: (-item[1], cluster_sort_key(item[0])),
        )[0][0]

    return MembershipMatrix(
        path=path,
        version=version,
        column=column,
        membership={emoji: dict(probs) for emoji, probs in membership.items()},
        top_cluster=top_cluster,
        clusters=clusters,
    )


def load_four_model_labels(path: Path | None) -> dict[str, FourModelLabels]:
    if path is None:
        return {}
    labels: dict[str, FourModelLabels] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            model_emojis_obj = obj.get("model_emojis", {})
            if not isinstance(model_emojis_obj, dict):
                continue
            utterance_id = make_utterance_id(obj)
            model_emojis = tuple(str(value) for _, value in sorted(model_emojis_obj.items()))
            bucket = agreement_bucket(model_emojis) or str(obj.get("agreement_bucket", ""))
            labels[utterance_id] = FourModelLabels(
                utterance_id=utterance_id,
                split=str(obj.get("split", "")),
                role=str(obj.get("role", "")),
                model_emojis=model_emojis,
                emoji_agreement_bucket=bucket,
                agreement_pattern=str(obj.get("agreement_pattern", "")),
            )
    return labels


def project_to_cluster_distribution(
    emoji_probs: dict[str, float],
    matrix: MembershipMatrix,
) -> tuple[dict[str, float], dict[str, float]]:
    cluster_probs: dict[str, float] = defaultdict(float)
    missing: dict[str, float] = defaultdict(float)
    for emoji, emoji_probability in emoji_probs.items():
        cluster_membership = matrix.membership.get(emoji)
        if cluster_membership is None:
            missing[emoji] += emoji_probability
            continue
        for cluster_id, membership_probability in cluster_membership.items():
            cluster_probs[cluster_id] += emoji_probability * membership_probability

    total = sum(cluster_probs.values())
    if total > 0 and abs(total - 1.0) > VALIDATION_TOLERANCE and not missing:
        cluster_probs = {
            cluster_id: probability / total
            for cluster_id, probability in cluster_probs.items()
        }
    return dict(cluster_probs), dict(missing)


def cluster_agreement_from_model_labels(
    model_emojis: tuple[str, ...],
    matrix: MembershipMatrix,
) -> tuple[str | None, int]:
    cluster_labels: list[str] = []
    missing_count = 0
    for emoji in model_emojis:
        cluster_id = matrix.top_cluster.get(emoji)
        if cluster_id is None:
            missing_count += 1
            continue
        cluster_labels.append(cluster_id)
    return agreement_bucket(cluster_labels), missing_count


def build_representation_metrics(
    utterances: list[UtteranceSoftLabels],
    matrices: tuple[MembershipMatrix, ...],
    four_model_labels: dict[str, FourModelLabels],
) -> tuple[list[RepresentationMetrics], dict[str, float]]:
    rows: list[RepresentationMetrics] = []
    missing_membership_mass: dict[str, float] = defaultdict(float)
    missing_model_label_count: dict[str, float] = defaultdict(float)

    for utterance in utterances:
        four_model = four_model_labels.get(utterance.utterance_id)
        emoji_bucket = (
            four_model.emoji_agreement_bucket
            if four_model is not None
            else utterance.agreement_type or None
        )
        emoji_entropy = entropy_bits(utterance.emoji_probs.values())
        emoji_top1 = top1_probability(utterance.emoji_probs)
        rows.append(
            RepresentationMetrics(
                utterance_id=utterance.utterance_id,
                role=utterance.role,
                split=utterance.split,
                representation="emoji",
                entropy_bits=emoji_entropy,
                top1_prob=emoji_top1,
                nonzero_count=sum(1 for probability in utterance.emoji_probs.values() if probability > 0),
                agreement_bucket=emoji_bucket,
                emoji_agreement_bucket=emoji_bucket,
                entropy_delta_vs_emoji=0.0,
                top1_delta_vs_emoji=0.0,
            )
        )

        for matrix in matrices:
            cluster_probs, missing = project_to_cluster_distribution(utterance.emoji_probs, matrix)
            for emoji, mass in missing.items():
                missing_membership_mass[f"{matrix.version}:{emoji}"] += mass
            total = sum(cluster_probs.values())
            if not cluster_probs or missing or abs(total - 1.0) > VALIDATION_TOLERANCE:
                continue

            cluster_bucket = None
            if four_model is not None:
                cluster_bucket, missing_count = cluster_agreement_from_model_labels(
                    four_model.model_emojis,
                    matrix,
                )
                if missing_count:
                    missing_model_label_count[matrix.version] += missing_count

            cluster_entropy = entropy_bits(cluster_probs.values())
            cluster_top1 = top1_probability(cluster_probs)
            rows.append(
                RepresentationMetrics(
                    utterance_id=utterance.utterance_id,
                    role=utterance.role,
                    split=utterance.split,
                    representation=f"cluster_{matrix.version}",
                    entropy_bits=cluster_entropy,
                    top1_prob=cluster_top1,
                    nonzero_count=sum(1 for probability in cluster_probs.values() if probability > 0),
                    agreement_bucket=cluster_bucket,
                    emoji_agreement_bucket=emoji_bucket,
                    entropy_delta_vs_emoji=cluster_entropy - emoji_entropy,
                    top1_delta_vs_emoji=cluster_top1 - emoji_top1,
                )
            )

    diagnostics = {
        "missing_membership_entries": float(len(missing_membership_mass)),
        "missing_membership_mass": float(sum(missing_membership_mass.values())),
        "missing_model_label_count": float(sum(missing_model_label_count.values())),
    }
    return rows, diagnostics


def aggregate_rows(
    metrics: list[RepresentationMetrics],
    group_type: str,
    group_getter,
) -> list[AggregateRow]:
    grouped: dict[tuple[str, str], list[RepresentationMetrics]] = defaultdict(list)
    for row in metrics:
        group_value = group_getter(row)
        if not group_value:
            continue
        grouped[(str(group_value), row.representation)].append(row)

    aggregate: list[AggregateRow] = []
    for (group_value, representation), rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], REPRESENTATIONS.index(item[0][1]) if item[0][1] in REPRESENTATIONS else 99),
    ):
        bucket_counts = {
            bucket: sum(1 for row in rows if row.agreement_bucket == bucket)
            for bucket in AGREEMENT_BUCKETS
        }
        agreement_available = sum(1 for row in rows if row.agreement_bucket is not None)
        bucket_rates = {
            bucket: (bucket_counts[bucket] / agreement_available if agreement_available else 0.0)
            for bucket in AGREEMENT_BUCKETS
        }

        improved = degraded = unchanged = 0
        comparable = 0
        if representation != "emoji":
            for row in rows:
                if row.agreement_bucket is None or row.emoji_agreement_bucket is None:
                    continue
                comparable += 1
                cluster_rank = AGREEMENT_RANK[row.agreement_bucket]
                emoji_rank = AGREEMENT_RANK[row.emoji_agreement_bucket]
                if cluster_rank > emoji_rank:
                    improved += 1
                elif cluster_rank < emoji_rank:
                    degraded += 1
                else:
                    unchanged += 1

        count = len(rows)
        aggregate.append(
            AggregateRow(
                group_type=group_type,
                group_value=group_value,
                representation=representation,
                utterance_count=count,
                agreement_available_count=agreement_available,
                mean_entropy_bits=sum(row.entropy_bits for row in rows) / count if count else 0.0,
                mean_top1_prob=sum(row.top1_prob for row in rows) / count if count else 0.0,
                mean_nonzero_count=sum(row.nonzero_count for row in rows) / count if count else 0.0,
                mean_entropy_delta_vs_emoji=sum(row.entropy_delta_vs_emoji for row in rows) / count if count else 0.0,
                mean_top1_delta_vs_emoji=sum(row.top1_delta_vs_emoji for row in rows) / count if count else 0.0,
                agreement_improved_rate=improved / comparable if comparable else None,
                agreement_degraded_rate=degraded / comparable if comparable else None,
                agreement_unchanged_rate=unchanged / comparable if comparable else None,
                bucket_counts=bucket_counts,
                bucket_rates=bucket_rates,
            )
        )
    return aggregate


def write_aggregate_csv(path: Path, rows: list[AggregateRow]) -> None:
    fieldnames = [
        "group_type",
        "group_value",
        "representation",
        "utterance_count",
        "agreement_available_count",
        "mean_entropy_bits",
        "mean_top1_prob",
        "mean_nonzero_count",
        "mean_entropy_delta_vs_emoji",
        "mean_top1_delta_vs_emoji",
        "agreement_improved_rate",
        "agreement_degraded_rate",
        "agreement_unchanged_rate",
    ]
    for bucket in AGREEMENT_BUCKETS:
        fieldnames.extend([f"{bucket}_count", f"{bucket}_rate"])

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload: dict[str, Any] = {
                "group_type": row.group_type,
                "group_value": row.group_value,
                "representation": row.representation,
                "utterance_count": row.utterance_count,
                "agreement_available_count": row.agreement_available_count,
                "mean_entropy_bits": f"{row.mean_entropy_bits:.12f}",
                "mean_top1_prob": f"{row.mean_top1_prob:.12f}",
                "mean_nonzero_count": f"{row.mean_nonzero_count:.12f}",
                "mean_entropy_delta_vs_emoji": f"{row.mean_entropy_delta_vs_emoji:.12f}",
                "mean_top1_delta_vs_emoji": f"{row.mean_top1_delta_vs_emoji:.12f}",
                "agreement_improved_rate": (
                    f"{row.agreement_improved_rate:.12f}"
                    if row.agreement_improved_rate is not None
                    else ""
                ),
                "agreement_degraded_rate": (
                    f"{row.agreement_degraded_rate:.12f}"
                    if row.agreement_degraded_rate is not None
                    else ""
                ),
                "agreement_unchanged_rate": (
                    f"{row.agreement_unchanged_rate:.12f}"
                    if row.agreement_unchanged_rate is not None
                    else ""
                ),
            }
            for bucket in AGREEMENT_BUCKETS:
                payload[f"{bucket}_count"] = row.bucket_counts[bucket]
                payload[f"{bucket}_rate"] = f"{row.bucket_rates[bucket]:.12f}"
            writer.writerow(payload)


def bar_color(representation: str) -> tuple[int, int, int]:
    if representation == "emoji":
        return (102, 126, 234)
    if representation == "cluster_raw":
        return (72, 187, 120)
    return (237, 137, 54)


def draw_entropy_chart(path: Path, overall_rows: list[AggregateRow]) -> None:
    rows = {
        row.representation: row
        for row in overall_rows
        if row.group_value == "all"
    }
    labels = ["emoji", "cluster_raw", "cluster_sharp"]
    values = [rows[label].mean_entropy_bits for label in labels if label in rows]
    labels = [label for label in labels if label in rows]
    width, height = 900, 560
    margin_left, margin_right, margin_top, margin_bottom = 90, 50, 80, 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin_left, 25), "Emoji vs Cluster Mean Entropy (bits)", fill=(20, 20, 20))
    max_value = max(values) * 1.15 if values else 1.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    axis_y = margin_top + plot_height
    draw.line((margin_left, margin_top, margin_left, axis_y), fill=(60, 60, 60), width=2)
    draw.line((margin_left, axis_y, width - margin_right, axis_y), fill=(60, 60, 60), width=2)

    for tick in range(6):
        value = max_value * tick / 5
        y = axis_y - int(plot_height * value / max_value)
        draw.line((margin_left - 5, y, width - margin_right, y), fill=(230, 230, 230), width=1)
        draw.text((20, y - 8), f"{value:.2f}", fill=(70, 70, 70))

    slot_width = plot_width / max(len(labels), 1)
    bar_width = int(slot_width * 0.48)
    for index, (label, value) in enumerate(zip(labels, values)):
        center_x = margin_left + int(slot_width * (index + 0.5))
        bar_height = int(plot_height * value / max_value)
        left = center_x - bar_width // 2
        right = center_x + bar_width // 2
        top = axis_y - bar_height
        draw.rectangle((left, top, right, axis_y), fill=bar_color(label))
        draw.text((left, top - 22), f"{value:.3f}", fill=(30, 30, 30))
        draw.text((left - 10, axis_y + 15), label, fill=(30, 30, 30))

    image.save(path)


def select_recommendation(overall_rows: list[AggregateRow]) -> str:
    rows = {row.representation: row for row in overall_rows if row.group_value == "all"}
    raw = rows.get("cluster_raw")
    sharp = rows.get("cluster_sharp")
    if raw is None or sharp is None:
        return "raw"
    entropy_gain = raw.mean_entropy_bits - sharp.mean_entropy_bits
    top1_gain = sharp.mean_top1_prob - raw.mean_top1_prob
    sharp_degraded = sharp.agreement_degraded_rate or 0.0
    raw_degraded = raw.agreement_degraded_rate or 0.0
    if entropy_gain > 0.02 and top1_gain > 0.005 and sharp_degraded <= raw_degraded + 0.005:
        return "sharpened"
    return "raw"


def write_report(
    path: Path,
    soft_candidate: InputCandidate,
    membership_candidate: InputCandidate,
    four_model_candidate: InputCandidate | None,
    overall_rows: list[AggregateRow],
    role_rows: list[AggregateRow],
    split_rows: list[AggregateRow],
    diagnostics: dict[str, float],
    recommendation: str,
    output_paths: dict[str, Path],
) -> None:
    overall = {row.representation: row for row in overall_rows if row.group_value == "all"}
    role_lookup = {(row.group_value, row.representation): row for row in role_rows}
    split_lookup = {(row.group_value, row.representation): row for row in split_rows}

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Cluster Agreement Stability Analysis\n\n")
        handle.write("## Inputs\n\n")
        handle.write(f"- Soft emoji labels + text/role/split: `{soft_candidate.path}`\n")
        handle.write(f"- Emoji-cluster membership: `{membership_candidate.path}`\n")
        if four_model_candidate is not None:
            handle.write(f"- Four-model utterance labels: `{four_model_candidate.path}`\n")
        else:
            handle.write("- Four-model utterance labels: not found; cluster agreement buckets unavailable.\n")
        handle.write("- Emoji names, external emoji lexicons, and transition information were not used.\n\n")

        handle.write("## Overall Comparison\n\n")
        handle.write("| representation | mean entropy bits | mean top1 prob | mean nonzero | 4_of_4 rate | all_different rate | agreement improved |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for representation in REPRESENTATIONS:
            row = overall.get(representation)
            if row is None:
                continue
            improved = row.agreement_improved_rate if row.agreement_improved_rate is not None else 0.0
            handle.write(
                f"| `{representation}` | {row.mean_entropy_bits:.6f} | "
                f"{row.mean_top1_prob:.6f} | {row.mean_nonzero_count:.6f} | "
                f"{row.bucket_rates['4_of_4']:.6f} | {row.bucket_rates['all_different']:.6f} | "
                f"{improved:.6f} |\n"
            )

        emoji = overall.get("emoji")
        raw = overall.get("cluster_raw")
        sharp = overall.get("cluster_sharp")
        handle.write("\n## Findings\n\n")
        if emoji is not None and raw is not None and sharp is not None:
            raw_entropy_drop = emoji.mean_entropy_bits - raw.mean_entropy_bits
            sharp_entropy_drop = emoji.mean_entropy_bits - sharp.mean_entropy_bits
            raw_top1_gain = raw.mean_top1_prob - emoji.mean_top1_prob
            sharp_top1_gain = sharp.mean_top1_prob - emoji.mean_top1_prob
            handle.write(
                f"- Raw cluster projection changes mean entropy by `{raw_entropy_drop:.6f}` bits "
                f"and top1 probability by `{raw_top1_gain:.6f}` versus emoji level.\n"
            )
            handle.write(
                f"- Sharpened cluster projection changes mean entropy by `{sharp_entropy_drop:.6f}` bits "
                f"and top1 probability by `{sharp_top1_gain:.6f}` versus emoji level.\n"
            )
            handle.write(
                "- Cluster agreement buckets improve over emoji buckets for "
                f"`{(raw.agreement_improved_rate or 0.0):.6f}` of utterances with raw membership "
                f"and `{(sharp.agreement_improved_rate or 0.0):.6f}` with sharpened membership.\n"
            )

        handle.write("\n## Role Comparison\n\n")
        for representation in ("emoji", "cluster_raw", "cluster_sharp"):
            a = role_lookup.get(("A", representation))
            b = role_lookup.get(("B", representation))
            if a is None or b is None:
                continue
            handle.write(
                f"- `{representation}`: A entropy `{a.mean_entropy_bits:.6f}`, "
                f"B entropy `{b.mean_entropy_bits:.6f}`; A top1 `{a.mean_top1_prob:.6f}`, "
                f"B top1 `{b.mean_top1_prob:.6f}`.\n"
            )

        handle.write("\n## Split Comparison\n\n")
        for split in sorted({row.group_value for row in split_rows}):
            raw_row = split_lookup.get((split, "cluster_raw"))
            sharp_row = split_lookup.get((split, "cluster_sharp"))
            if raw_row is None or sharp_row is None:
                continue
            handle.write(
                f"- `{split}`: raw entropy `{raw_row.mean_entropy_bits:.6f}`, "
                f"sharpened entropy `{sharp_row.mean_entropy_bits:.6f}`; raw top1 "
                f"`{raw_row.mean_top1_prob:.6f}`, sharpened top1 `{sharp_row.mean_top1_prob:.6f}`.\n"
            )

        handle.write("\n## Diagnostics\n\n")
        handle.write(f"- Missing membership entries: `{diagnostics['missing_membership_entries']:.0f}`\n")
        handle.write(f"- Missing membership mass: `{diagnostics['missing_membership_mass']:.12f}`\n")
        handle.write(f"- Missing model-label mappings: `{diagnostics['missing_model_label_count']:.0f}`\n")
        handle.write(f"- Recommendation for graph construction: `{recommendation}` membership.\n\n")

        handle.write("## Outputs\n\n")
        for label, output_path in output_paths.items():
            handle.write(f"- {label}: `{output_path}`\n")


def default_output_dir(soft_label_path: Path) -> Path:
    for parent in soft_label_path.parents:
        if parent.name == "outputs":
            return parent / "cluster_stability_analysis"
    return soft_label_path.parent / "cluster_stability_analysis"


def run_analysis(root: Path, output_dir: Path | None = None) -> tuple[list[AggregateRow], str, dict[str, Path]]:
    soft_candidate, membership_candidate, four_model_candidate = discover_inputs(root)
    utterances, max_soft_sum_error = load_soft_label_table(soft_candidate.path)
    raw_membership = load_membership(membership_candidate.path, "raw", "membership_raw")
    matrices = [raw_membership]
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
    recommendation = select_recommendation(overall_rows)

    actual_output_dir = output_dir if output_dir is not None else default_output_dir(soft_candidate.path)
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "cluster_agreement_summary": actual_output_dir / "cluster_agreement_summary.csv",
        "cluster_agreement_by_role": actual_output_dir / "cluster_agreement_by_role.csv",
        "cluster_agreement_by_split": actual_output_dir / "cluster_agreement_by_split.csv",
        "cluster_agreement_report": actual_output_dir / "cluster_agreement_report.md",
        "emoji_vs_cluster_entropy": actual_output_dir / "emoji_vs_cluster_entropy.png",
    }
    write_aggregate_csv(output_paths["cluster_agreement_summary"], overall_rows)
    write_aggregate_csv(output_paths["cluster_agreement_by_role"], role_rows)
    write_aggregate_csv(output_paths["cluster_agreement_by_split"], split_rows)
    draw_entropy_chart(output_paths["emoji_vs_cluster_entropy"], overall_rows)
    write_report(
        output_paths["cluster_agreement_report"],
        soft_candidate,
        membership_candidate,
        four_model_candidate,
        overall_rows,
        role_rows,
        split_rows,
        diagnostics,
        recommendation,
        output_paths,
    )
    return overall_rows, recommendation, output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare utterance-level emoji uncertainty with projected cluster uncertainty."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to search for soft labels, membership, and optional four-model labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for cluster stability analysis artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, recommendation, output_paths = run_analysis(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
    )
    overall = {row.representation: row for row in rows if row.group_value == "all"}
    print(f"Wrote summary: {output_paths['cluster_agreement_summary']}")
    print(f"Wrote role breakdown: {output_paths['cluster_agreement_by_role']}")
    print(f"Wrote split breakdown: {output_paths['cluster_agreement_by_split']}")
    print(f"Wrote report: {output_paths['cluster_agreement_report']}")
    print(f"Wrote entropy plot: {output_paths['emoji_vs_cluster_entropy']}")
    if "emoji" in overall and "cluster_raw" in overall and "cluster_sharp" in overall:
        print(
            "Summary: "
            f"emoji entropy {overall['emoji'].mean_entropy_bits:.4f}, "
            f"raw cluster entropy {overall['cluster_raw'].mean_entropy_bits:.4f}, "
            f"sharpened cluster entropy {overall['cluster_sharp'].mean_entropy_bits:.4f}; "
            f"recommendation={recommendation}."
        )


if __name__ == "__main__":
    main()
