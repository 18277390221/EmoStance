from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .soft_membership import cluster_sort_key


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
TRANSITION_TYPES = ("A2B", "B2A", "A2A", "B2B")
MAIN_TRANSITION_TYPES = ("A2B", "B2A")
VALIDATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class InputCandidate:
    path: Path
    kind: str
    score: int
    fields: tuple[str, ...]
    rationale: str


@dataclass
class WorkingUtterance:
    utterance_id: str
    dialogue_id: str
    turn_id: str
    turn_index: str
    split: str
    role: str
    emoji_probs: dict[str, float] = field(default_factory=dict)
    direct_mean_confidence: float | None = None
    confidence_weighted_sum: float = 0.0
    confidence_weight_total: float = 0.0
    confidence_simple_sum: float = 0.0
    confidence_simple_count: int = 0

    @property
    def mean_confidence(self) -> float | None:
        if self.direct_mean_confidence is not None:
            return self.direct_mean_confidence
        if self.confidence_weight_total > 0:
            return self.confidence_weighted_sum / self.confidence_weight_total
        if self.confidence_simple_count > 0:
            return self.confidence_simple_sum / self.confidence_simple_count
        return None


@dataclass(frozen=True)
class ProjectedUtterance:
    utterance_id: str
    dialogue_id: str
    turn_id: str
    turn_index: str
    split: str
    role: str
    cluster_probs: dict[str, float]
    top1_cluster: str
    top1_cluster_prob: float
    cluster_entropy: float
    nonzero_cluster_count: int
    mean_confidence: float | None
    reliability: float


@dataclass(frozen=True)
class ProjectionResult:
    utterances: tuple[ProjectedUtterance, ...]
    clusters: tuple[str, ...]
    missing_emojis: dict[str, float]
    extra_membership_emojis: tuple[str, ...]
    max_cluster_sum_error: float
    max_emoji_sum_error: float


@dataclass(frozen=True)
class TransitionComputation:
    weighted_counts: dict[str, dict[tuple[str, str], float]]
    unweighted_counts: dict[str, dict[tuple[str, str], float]]
    transition_instances: dict[str, int]


@dataclass(frozen=True)
class ConditionalRow:
    transition_type: str
    source_cluster: str
    target_cluster: str
    raw_soft_count: float
    unweighted_soft_count: float
    smoothed_count: float
    source_marginal: float
    target_marginal: float
    conditional_prob: float


@dataclass(frozen=True)
class AssociationRow:
    transition_type: str
    source_cluster: str
    target_cluster: str
    raw_soft_count: float
    source_marginal: float
    target_marginal: float
    joint_prob: float
    pmi: float | None
    ppmi: float
    lift: float


@dataclass(frozen=True)
class TransitionSummary:
    transition_type: str
    transition_instances: int
    total_soft_count: float
    nonzero_edges: int
    possible_edges: int
    density: float
    mean_abs_reliability_delta: float
    max_abs_reliability_delta: float


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


def has_utterance_id_or_parts(columns: set[str]) -> bool:
    return "utterance_id" in columns or {"dialogue_id", "turn_id"}.issubset(columns)


def inspect_soft_label_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"emoji", "soft_prob", "role", "split"}.issubset(columns):
        return None
    if not has_utterance_id_or_parts(columns):
        return None

    score = 100
    if "utterance_id" in columns:
        score += 20
    if "turn_index" in columns:
        score += 15
    if "utterance_mean_confidence" in columns:
        score += 12
    if {"mean_confidence", "support_model_count"}.issubset(columns):
        score += 8
    if {"top1_prob", "entropy"}.intersection(columns):
        score += 4
    return InputCandidate(
        path=path,
        kind="utterance_soft_label_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has utterance identity, role/split, emoji, and soft_prob columns.",
    )


def inspect_membership_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"emoji", "cluster_id"}.issubset(columns):
        return None
    if not {"membership_prob", "membership_raw"}.intersection(columns):
        return None
    score = 100
    if "source_type" in columns:
        score += 10
    if "observed_count" in columns:
        score += 5
    return InputCandidate(
        path=path,
        kind="emoji_cluster_membership_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has one row per emoji-cluster membership probability.",
    )


def inspect_turn_order_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"dialogue_id", "turn_id", "role", "split"}.issubset(columns):
        return None
    score = 50
    if "turn_index" in columns:
        score += 25
    if "utterance_id" in columns:
        score += 10
    if "soft_prob" in columns and "emoji" in columns:
        score += 8
    return InputCandidate(
        path=path,
        kind="turn_order_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has dialogue, turn, split, and role columns suitable for turn ordering.",
    )


def discover_input_candidates(root: Path) -> tuple[InputCandidate, InputCandidate, InputCandidate]:
    soft_candidates: list[InputCandidate] = []
    membership_candidates: list[InputCandidate] = []
    order_candidates: list[InputCandidate] = []
    for path in iter_files(root, {".csv"}):
        soft_candidate = inspect_soft_label_csv(path)
        if soft_candidate is not None:
            soft_candidates.append(soft_candidate)
        membership_candidate = inspect_membership_csv(path)
        if membership_candidate is not None:
            membership_candidates.append(membership_candidate)
        order_candidate = inspect_turn_order_csv(path)
        if order_candidate is not None:
            order_candidates.append(order_candidate)

    if not soft_candidates:
        raise FileNotFoundError(
            f"No canonical utterance soft-label CSV found under {root}. Expected columns "
            "`emoji`, `soft_prob`, `role`, `split`, and utterance identity columns."
        )
    if not membership_candidates:
        raise FileNotFoundError(
            f"No emoji-cluster membership CSV found under {root}. Expected columns "
            "`emoji`, `cluster_id`, and `membership_raw` or `membership_prob`."
        )
    if not order_candidates:
        raise FileNotFoundError(
            f"No turn-order table found under {root}. Expected columns `dialogue_id`, "
            "`turn_id`, `role`, and `split`."
        )

    sort_key = lambda candidate: (
        candidate.score,
        candidate.path.stat().st_mtime if candidate.path.exists() else 0.0,
        str(candidate.path),
    )
    return (
        sorted(soft_candidates, key=sort_key, reverse=True)[0],
        sorted(membership_candidates, key=sort_key, reverse=True)[0],
        sorted(order_candidates, key=sort_key, reverse=True)[0],
    )


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_intish_order(value: str) -> tuple[int, str]:
    try:
        return int(float(value)), value
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group(0)), str(value)
    return math.inf, str(value)


def make_utterance_id(row: dict[str, str]) -> str:
    utterance_id = row.get("utterance_id")
    if utterance_id:
        return utterance_id
    return f"{row.get('split', '')}|{row.get('dialogue_id', '')}|{row.get('turn_id', '')}"


def role_pair(source_role: str, target_role: str) -> str | None:
    source = source_role.strip().upper()
    target = target_role.strip().upper()
    candidate = f"{source}2{target}"
    return candidate if candidate in TRANSITION_TYPES else None


def load_membership_matrix(path: Path) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
    membership: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emoji = row.get("emoji")
            cluster_id = row.get("cluster_id")
            probability = parse_float(row.get("membership_raw"))
            if probability is None:
                probability = parse_float(row.get("membership_prob"))
            if not emoji or not cluster_id or probability is None:
                continue
            membership[emoji][cluster_id] = membership[emoji].get(cluster_id, 0.0) + probability

    if not membership:
        raise ValueError(f"Membership matrix is empty: {path}")

    for emoji, probs in membership.items():
        prob_sum = sum(probs.values())
        if abs(prob_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Membership probabilities for {emoji} sum to {prob_sum:.12f}, not 1.0."
            )
    clusters = tuple(sorted({cluster for probs in membership.values() for cluster in probs}, key=cluster_sort_key))
    return {emoji: dict(probs) for emoji, probs in membership.items()}, clusters


def load_soft_label_utterances(path: Path) -> tuple[list[WorkingUtterance], float]:
    groups: OrderedDict[str, WorkingUtterance] = OrderedDict()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            utterance_id = make_utterance_id(row)
            group = groups.get(utterance_id)
            if group is None:
                group = WorkingUtterance(
                    utterance_id=utterance_id,
                    dialogue_id=row.get("dialogue_id", ""),
                    turn_id=row.get("turn_id", ""),
                    turn_index=row.get("turn_index", row.get("turn_id", "")),
                    split=row.get("split", ""),
                    role=row.get("role", ""),
                )
                groups[utterance_id] = group

            emoji = row.get("emoji")
            soft_prob = parse_float(row.get("soft_prob"))
            if emoji and soft_prob is not None and soft_prob > 0:
                group.emoji_probs[emoji] = group.emoji_probs.get(emoji, 0.0) + soft_prob

            direct_conf = parse_float(row.get("utterance_mean_confidence"))
            if direct_conf is not None:
                group.direct_mean_confidence = direct_conf

            row_conf = parse_float(row.get("row_mean_confidence"))
            if row_conf is None:
                row_conf = parse_float(row.get("mean_confidence"))
            support = parse_float(row.get("row_support_model_count"))
            if support is None:
                support = parse_float(row.get("support_model_count"))
            if row_conf is not None:
                if support is not None and support > 0:
                    group.confidence_weighted_sum += row_conf * support
                    group.confidence_weight_total += support
                else:
                    group.confidence_simple_sum += row_conf
                    group.confidence_simple_count += 1

    max_emoji_sum_error = 0.0
    for group in groups.values():
        total = sum(group.emoji_probs.values())
        max_emoji_sum_error = max(max_emoji_sum_error, abs(total - 1.0))
        if total > 0 and abs(total - 1.0) > VALIDATION_TOLERANCE:
            group.emoji_probs = {
                emoji: probability / total for emoji, probability in group.emoji_probs.items()
            }
    utterances = [group for group in groups.values() if group.emoji_probs]
    if not utterances:
        raise ValueError(f"No utterance soft labels could be loaded from {path}.")
    return utterances, max_emoji_sum_error


def entropy(probabilities: Iterable[float]) -> float:
    value = 0.0
    for probability in probabilities:
        if probability > 0:
            value -= probability * math.log(probability)
    return value


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def reliability_weight(cluster_entropy: float, cluster_count: int, mean_confidence: float | None) -> float:
    if cluster_count <= 1:
        uncertainty_factor = 1.0
    else:
        uncertainty_factor = 1.0 - cluster_entropy / math.log(cluster_count)
    uncertainty_factor = clamp(uncertainty_factor)
    if mean_confidence is None:
        return uncertainty_factor
    return clamp((mean_confidence / 5.0) * uncertainty_factor)


def project_utterances_to_clusters(
    utterances: list[WorkingUtterance],
    membership: dict[str, dict[str, float]],
    clusters: tuple[str, ...],
) -> ProjectionResult:
    projected: list[ProjectedUtterance] = []
    missing_emojis: dict[str, float] = defaultdict(float)
    observed_emojis: set[str] = set()
    max_cluster_sum_error = 0.0

    for utterance in utterances:
        cluster_probs: dict[str, float] = defaultdict(float)
        for emoji, emoji_prob in utterance.emoji_probs.items():
            observed_emojis.add(emoji)
            emoji_membership = membership.get(emoji)
            if emoji_membership is None:
                missing_emojis[emoji] += emoji_prob
                continue
            for cluster_id, membership_prob in emoji_membership.items():
                cluster_probs[cluster_id] += emoji_prob * membership_prob

        cluster_total = sum(cluster_probs.values())
        max_cluster_sum_error = max(max_cluster_sum_error, abs(cluster_total - 1.0))
        if cluster_total <= 0:
            continue
        if abs(cluster_total - 1.0) > VALIDATION_TOLERANCE and not missing_emojis:
            cluster_probs = {
                cluster_id: probability / cluster_total
                for cluster_id, probability in cluster_probs.items()
            }

        if abs(sum(cluster_probs.values()) - 1.0) > VALIDATION_TOLERANCE:
            continue

        nonzero = {
            cluster_id: probability
            for cluster_id, probability in cluster_probs.items()
            if probability > 1e-12
        }
        top1_cluster, top1_cluster_prob = sorted(
            nonzero.items(),
            key=lambda item: (-item[1], cluster_sort_key(item[0])),
        )[0]
        cluster_entropy = entropy(nonzero.values())
        reliability = reliability_weight(cluster_entropy, len(clusters), utterance.mean_confidence)
        projected.append(
            ProjectedUtterance(
                utterance_id=utterance.utterance_id,
                dialogue_id=utterance.dialogue_id,
                turn_id=utterance.turn_id,
                turn_index=utterance.turn_index,
                split=utterance.split,
                role=utterance.role,
                cluster_probs=dict(sorted(nonzero.items(), key=lambda item: cluster_sort_key(item[0]))),
                top1_cluster=top1_cluster,
                top1_cluster_prob=top1_cluster_prob,
                cluster_entropy=cluster_entropy,
                nonzero_cluster_count=len(nonzero),
                mean_confidence=utterance.mean_confidence,
                reliability=reliability,
            )
        )

    if missing_emojis:
        missing_preview = ", ".join(
            f"{emoji}:{mass:.6f}"
            for emoji, mass in sorted(missing_emojis.items(), key=lambda item: (-item[1], item[0]))[:20]
        )
        raise ValueError(
            "Cannot project soft labels because emojis are missing from the membership matrix: "
            f"{missing_preview}"
        )

    extra_membership_emojis = tuple(sorted(set(membership) - observed_emojis))
    return ProjectionResult(
        utterances=tuple(projected),
        clusters=clusters,
        missing_emojis=dict(missing_emojis),
        extra_membership_emojis=extra_membership_emojis,
        max_cluster_sum_error=max_cluster_sum_error,
        max_emoji_sum_error=0.0,
    )


def validate_projected_utterances(projection: ProjectionResult) -> None:
    for utterance in projection.utterances:
        prob_sum = sum(utterance.cluster_probs.values())
        if abs(prob_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Cluster probabilities for {utterance.utterance_id} sum to {prob_sum:.12f}."
            )


def group_dialogues(utterances: Iterable[ProjectedUtterance]) -> dict[tuple[str, str], list[ProjectedUtterance]]:
    grouped: dict[tuple[str, str], list[ProjectedUtterance]] = defaultdict(list)
    for utterance in utterances:
        grouped[(utterance.split, utterance.dialogue_id)].append(utterance)
    for dialogue_utterances in grouped.values():
        dialogue_utterances.sort(
            key=lambda item: (parse_intish_order(item.turn_index), parse_intish_order(item.turn_id))
        )
    return grouped


def add_transition(
    source: ProjectedUtterance,
    target: ProjectedUtterance,
    transition_type: str,
    weighted_counts: dict[str, dict[tuple[str, str], float]],
    unweighted_counts: dict[str, dict[tuple[str, str], float]],
) -> None:
    reliability = math.sqrt(max(0.0, source.reliability * target.reliability))
    for source_cluster, source_prob in source.cluster_probs.items():
        for target_cluster, target_prob in target.cluster_probs.items():
            value = source_prob * target_prob
            pair = (source_cluster, target_cluster)
            unweighted_counts[transition_type][pair] = unweighted_counts[transition_type].get(pair, 0.0) + value
            weighted_counts[transition_type][pair] = weighted_counts[transition_type].get(pair, 0.0) + value * reliability


def build_transition_counts(utterances: Iterable[ProjectedUtterance]) -> TransitionComputation:
    weighted_counts: dict[str, dict[tuple[str, str], float]] = {transition_type: {} for transition_type in TRANSITION_TYPES}
    unweighted_counts: dict[str, dict[tuple[str, str], float]] = {transition_type: {} for transition_type in TRANSITION_TYPES}
    transition_instances = {transition_type: 0 for transition_type in TRANSITION_TYPES}

    for dialogue_utterances in group_dialogues(utterances).values():
        for index, source in enumerate(dialogue_utterances):
            if index + 1 < len(dialogue_utterances):
                target = dialogue_utterances[index + 1]
                transition_type = role_pair(source.role, target.role)
                if transition_type in MAIN_TRANSITION_TYPES:
                    transition_instances[transition_type] += 1
                    add_transition(
                        source,
                        target,
                        transition_type,
                        weighted_counts,
                        unweighted_counts,
                    )

            if index + 2 < len(dialogue_utterances):
                target = dialogue_utterances[index + 2]
                transition_type = role_pair(source.role, target.role)
                if transition_type in {"A2A", "B2B"}:
                    transition_instances[transition_type] += 1
                    add_transition(
                        source,
                        target,
                        transition_type,
                        weighted_counts,
                        unweighted_counts,
                    )

    return TransitionComputation(
        weighted_counts=weighted_counts,
        unweighted_counts=unweighted_counts,
        transition_instances=transition_instances,
    )


def source_target_totals(
    counts: dict[tuple[str, str], float],
    clusters: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float], float]:
    source_totals = {cluster: 0.0 for cluster in clusters}
    target_totals = {cluster: 0.0 for cluster in clusters}
    total = 0.0
    for (source_cluster, target_cluster), value in counts.items():
        source_totals[source_cluster] = source_totals.get(source_cluster, 0.0) + value
        target_totals[target_cluster] = target_totals.get(target_cluster, 0.0) + value
        total += value
    return source_totals, target_totals, total


def build_conditional_rows(
    transition_type: str,
    weighted_counts: dict[tuple[str, str], float],
    unweighted_counts: dict[tuple[str, str], float],
    clusters: tuple[str, ...],
    alpha: float,
) -> list[ConditionalRow]:
    source_totals, target_totals, total = source_target_totals(weighted_counts, clusters)
    rows: list[ConditionalRow] = []
    for source_cluster in clusters:
        denominator = source_totals[source_cluster] + alpha * len(clusters)
        source_prob_sum = 0.0
        for target_cluster in clusters:
            raw_count = weighted_counts.get((source_cluster, target_cluster), 0.0)
            smoothed_count = raw_count + alpha
            conditional_prob = smoothed_count / denominator if denominator > 0 else 1.0 / len(clusters)
            source_prob_sum += conditional_prob
            rows.append(
                ConditionalRow(
                    transition_type=transition_type,
                    source_cluster=source_cluster,
                    target_cluster=target_cluster,
                    raw_soft_count=raw_count,
                    unweighted_soft_count=unweighted_counts.get((source_cluster, target_cluster), 0.0),
                    smoothed_count=smoothed_count,
                    source_marginal=source_totals[source_cluster] / total if total > 0 else 0.0,
                    target_marginal=target_totals[target_cluster] / total if total > 0 else 0.0,
                    conditional_prob=conditional_prob,
                )
            )
        if abs(source_prob_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Conditional probabilities for {transition_type} {source_cluster} "
                f"sum to {source_prob_sum:.12f}."
            )
    return rows


def build_association_rows(
    transition_type: str,
    weighted_counts: dict[tuple[str, str], float],
    clusters: tuple[str, ...],
) -> list[AssociationRow]:
    source_totals, target_totals, total = source_target_totals(weighted_counts, clusters)
    rows: list[AssociationRow] = []
    for source_cluster in clusters:
        for target_cluster in clusters:
            raw_count = weighted_counts.get((source_cluster, target_cluster), 0.0)
            joint_prob = raw_count / total if total > 0 else 0.0
            source_marginal = source_totals[source_cluster] / total if total > 0 else 0.0
            target_marginal = target_totals[target_cluster] / total if total > 0 else 0.0
            denominator = source_marginal * target_marginal
            if joint_prob > 0 and denominator > 0:
                lift = joint_prob / denominator
                pmi = math.log2(lift)
                ppmi = max(0.0, pmi)
            else:
                lift = 0.0
                pmi = None
                ppmi = 0.0
            rows.append(
                AssociationRow(
                    transition_type=transition_type,
                    source_cluster=source_cluster,
                    target_cluster=target_cluster,
                    raw_soft_count=raw_count,
                    source_marginal=source_marginal,
                    target_marginal=target_marginal,
                    joint_prob=joint_prob,
                    pmi=pmi,
                    ppmi=ppmi,
                    lift=lift,
                )
            )
    return rows


def conditional_map(rows: Iterable[ConditionalRow]) -> dict[tuple[str, str], float]:
    return {
        (row.source_cluster, row.target_cluster): row.conditional_prob
        for row in rows
    }


def summarize_transition_type(
    transition_type: str,
    clusters: tuple[str, ...],
    weighted_counts: dict[tuple[str, str], float],
    weighted_conditional: list[ConditionalRow],
    unweighted_conditional: list[ConditionalRow],
    transition_instances: int,
) -> TransitionSummary:
    possible_edges = len(clusters) * len(clusters)
    nonzero_edges = sum(1 for value in weighted_counts.values() if value > 0)
    weighted_map = conditional_map(weighted_conditional)
    unweighted_map = conditional_map(unweighted_conditional)
    diffs = [
        abs(weighted_map.get((source, target), 0.0) - unweighted_map.get((source, target), 0.0))
        for source in clusters
        for target in clusters
    ]
    return TransitionSummary(
        transition_type=transition_type,
        transition_instances=transition_instances,
        total_soft_count=sum(weighted_counts.values()),
        nonzero_edges=nonzero_edges,
        possible_edges=possible_edges,
        density=nonzero_edges / possible_edges if possible_edges else 0.0,
        mean_abs_reliability_delta=sum(diffs) / len(diffs) if diffs else 0.0,
        max_abs_reliability_delta=max(diffs) if diffs else 0.0,
    )


def format_float(value: float) -> str:
    return f"{value:.12f}"


def write_projected_rows(path: Path, projection: ProjectionResult) -> None:
    fieldnames = [
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "turn_index",
        "split",
        "role",
        "cluster_id",
        "cluster_prob",
        "reliability_weight",
        "top1_cluster",
        "top1_cluster_prob",
        "cluster_entropy",
        "nonzero_cluster_count",
        "mean_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for utterance in projection.utterances:
            for cluster_id, probability in utterance.cluster_probs.items():
                writer.writerow(
                    {
                        "utterance_id": utterance.utterance_id,
                        "dialogue_id": utterance.dialogue_id,
                        "turn_id": utterance.turn_id,
                        "turn_index": utterance.turn_index,
                        "split": utterance.split,
                        "role": utterance.role,
                        "cluster_id": cluster_id,
                        "cluster_prob": format_float(probability),
                        "reliability_weight": format_float(utterance.reliability),
                        "top1_cluster": utterance.top1_cluster,
                        "top1_cluster_prob": format_float(utterance.top1_cluster_prob),
                        "cluster_entropy": format_float(utterance.cluster_entropy),
                        "nonzero_cluster_count": utterance.nonzero_cluster_count,
                        "mean_confidence": (
                            format_float(utterance.mean_confidence)
                            if utterance.mean_confidence is not None
                            else ""
                        ),
                    }
                )


def write_utterance_summary(path: Path, projection: ProjectionResult) -> None:
    fieldnames = [
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "turn_index",
        "split",
        "role",
        "top1_cluster",
        "top1_cluster_prob",
        "cluster_entropy",
        "nonzero_cluster_count",
        "reliability_weight",
        "mean_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for utterance in projection.utterances:
            writer.writerow(
                {
                    "utterance_id": utterance.utterance_id,
                    "dialogue_id": utterance.dialogue_id,
                    "turn_id": utterance.turn_id,
                    "turn_index": utterance.turn_index,
                    "split": utterance.split,
                    "role": utterance.role,
                    "top1_cluster": utterance.top1_cluster,
                    "top1_cluster_prob": format_float(utterance.top1_cluster_prob),
                    "cluster_entropy": format_float(utterance.cluster_entropy),
                    "nonzero_cluster_count": utterance.nonzero_cluster_count,
                    "reliability_weight": format_float(utterance.reliability),
                    "mean_confidence": (
                        format_float(utterance.mean_confidence)
                        if utterance.mean_confidence is not None
                        else ""
                    ),
                }
            )


def write_count_rows(path: Path, rows: list[ConditionalRow]) -> None:
    fieldnames = [
        "transition_type",
        "source_cluster",
        "target_cluster",
        "raw_soft_count",
        "unweighted_soft_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "transition_type": row.transition_type,
                    "source_cluster": row.source_cluster,
                    "target_cluster": row.target_cluster,
                    "raw_soft_count": format_float(row.raw_soft_count),
                    "unweighted_soft_count": format_float(row.unweighted_soft_count),
                }
            )


def write_conditional_rows(path: Path, rows: list[ConditionalRow]) -> None:
    fieldnames = [
        "transition_type",
        "source_cluster",
        "target_cluster",
        "raw_soft_count",
        "unweighted_soft_count",
        "smoothed_count",
        "source_marginal",
        "target_marginal",
        "conditional_prob",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "transition_type": row.transition_type,
                    "source_cluster": row.source_cluster,
                    "target_cluster": row.target_cluster,
                    "raw_soft_count": format_float(row.raw_soft_count),
                    "unweighted_soft_count": format_float(row.unweighted_soft_count),
                    "smoothed_count": format_float(row.smoothed_count),
                    "source_marginal": format_float(row.source_marginal),
                    "target_marginal": format_float(row.target_marginal),
                    "conditional_prob": format_float(row.conditional_prob),
                }
            )


def write_association_rows(path: Path, rows: list[AssociationRow]) -> None:
    fieldnames = [
        "transition_type",
        "source_cluster",
        "target_cluster",
        "raw_soft_count",
        "source_marginal",
        "target_marginal",
        "joint_prob",
        "pmi",
        "ppmi",
        "lift",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "transition_type": row.transition_type,
                    "source_cluster": row.source_cluster,
                    "target_cluster": row.target_cluster,
                    "raw_soft_count": format_float(row.raw_soft_count),
                    "source_marginal": format_float(row.source_marginal),
                    "target_marginal": format_float(row.target_marginal),
                    "joint_prob": format_float(row.joint_prob),
                    "pmi": format_float(row.pmi) if row.pmi is not None else "",
                    "ppmi": format_float(row.ppmi),
                    "lift": format_float(row.lift),
                }
            )


def write_heatmap(path: Path, clusters: tuple[str, ...], values: dict[tuple[str, str], float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_cluster", *clusters])
        for source_cluster in clusters:
            writer.writerow(
                [
                    source_cluster,
                    *[
                        format_float(values.get((source_cluster, target_cluster), 0.0))
                        for target_cluster in clusters
                    ],
                ]
            )


def strongest_outgoing(rows: list[ConditionalRow], limit: int = 6) -> list[ConditionalRow]:
    by_source: dict[str, list[ConditionalRow]] = defaultdict(list)
    for row in rows:
        by_source[row.source_cluster].append(row)
    winners = [
        sorted(source_rows, key=lambda row: (-row.conditional_prob, cluster_sort_key(row.target_cluster)))[0]
        for source_rows in by_source.values()
    ]
    return sorted(winners, key=lambda row: (-row.conditional_prob, cluster_sort_key(row.source_cluster)))[:limit]


def common_targets(rows: list[ConditionalRow], limit: int = 6) -> list[tuple[str, float]]:
    target_marginals: dict[str, float] = {}
    for row in rows:
        target_marginals[row.target_cluster] = row.target_marginal
    return sorted(target_marginals.items(), key=lambda item: (-item[1], cluster_sort_key(item[0])))[:limit]


def density_label(density: float) -> str:
    if density >= 0.67:
        return "dense"
    if density <= 0.33:
        return "sparse"
    return "moderately dense"


def reliability_label(summary: TransitionSummary) -> str:
    if summary.max_abs_reliability_delta >= 0.05 or summary.mean_abs_reliability_delta >= 0.01:
        return "noticeable"
    return "small"


def write_report(
    path: Path,
    soft_candidate: InputCandidate,
    membership_candidate: InputCandidate,
    order_candidate: InputCandidate,
    projection: ProjectionResult,
    summaries: dict[str, TransitionSummary],
    conditional_rows: dict[str, list[ConditionalRow]],
    output_paths: dict[str, Path],
) -> None:
    lines = [
        "# Cluster Transition Graph Report",
        "",
        "## Inputs",
        "",
        f"- Soft-label table: `{soft_candidate.path}`",
        f"- Membership matrix: `{membership_candidate.path}`",
        f"- Turn-order source: `{order_candidate.path}`",
        "- Emoji names, aliases, external emoji lexicons, and transition-derived emoji features were not used.",
        "",
        "## Projection",
        "",
        f"- Clusters used: `{len(projection.clusters)}` ({', '.join(projection.clusters)})",
        f"- Projected utterances: `{len(projection.utterances)}`",
        f"- Missing emojis from membership matrix: `{len(projection.missing_emojis)}`",
        f"- Extra membership emojis not observed in soft labels: `{len(projection.extra_membership_emojis)}`",
        f"- Max cluster-probability sum error: `{projection.max_cluster_sum_error:.12g}`",
        "",
        "## Graph Density",
        "",
    ]
    for transition_type in MAIN_TRANSITION_TYPES:
        summary = summaries[transition_type]
        lines.append(
            f"- `{transition_type}` is `{density_label(summary.density)}`: "
            f"`{summary.nonzero_edges}/{summary.possible_edges}` non-zero edges "
            f"(density `{summary.density:.3f}`), `{summary.transition_instances}` transitions."
        )

    lines.extend(["", "## Strongest Outgoing Preferences", ""])
    for transition_type in MAIN_TRANSITION_TYPES:
        lines.append(f"### {transition_type}")
        for row in strongest_outgoing(conditional_rows[transition_type]):
            lines.append(
                f"- `{row.source_cluster}` → `{row.target_cluster}`: "
                f"`P={row.conditional_prob:.3f}`, raw soft count `{row.raw_soft_count:.3f}`."
            )
        lines.append("")

    lines.append("## Most Common Listener Destinations")
    lines.append("")
    for transition_type in MAIN_TRANSITION_TYPES:
        target_text = ", ".join(
            f"`{cluster}` ({marginal:.3f})"
            for cluster, marginal in common_targets(conditional_rows[transition_type])
        )
        lines.append(f"- `{transition_type}` targets: {target_text}")

    lines.extend(["", "## Reliability Weighting", ""])
    for transition_type in MAIN_TRANSITION_TYPES:
        summary = summaries[transition_type]
        lines.append(
            f"- `{transition_type}` reliability effect is `{reliability_label(summary)}`: "
            f"mean |ΔP| `{summary.mean_abs_reliability_delta:.4f}`, "
            f"max |ΔP| `{summary.max_abs_reliability_delta:.4f}`."
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for name, output_path in output_paths.items():
        lines.append(f"- `{name}`: `{output_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(
    path: Path,
    projection: ProjectionResult,
    summaries: dict[str, TransitionSummary],
    soft_candidate: InputCandidate,
    membership_candidate: InputCandidate,
    order_candidate: InputCandidate,
) -> None:
    payload = {
        "soft_label_table": str(soft_candidate.path),
        "membership_matrix": str(membership_candidate.path),
        "turn_order_source": str(order_candidate.path),
        "cluster_count": len(projection.clusters),
        "clusters": list(projection.clusters),
        "utterance_count": len(projection.utterances),
        "missing_emojis": projection.missing_emojis,
        "extra_membership_emojis": list(projection.extra_membership_emojis),
        "transition_summaries": {
            transition_type: {
                "transition_instances": summary.transition_instances,
                "total_soft_count": summary.total_soft_count,
                "nonzero_edges": summary.nonzero_edges,
                "possible_edges": summary.possible_edges,
                "density": summary.density,
                "mean_abs_reliability_delta": summary.mean_abs_reliability_delta,
                "max_abs_reliability_delta": summary.max_abs_reliability_delta,
            }
            for transition_type, summary in summaries.items()
        },
        "hard_constraints": {
            "emoji_names_used": False,
            "aliases_used": False,
            "external_emoji_lexicon_used": False,
            "transition_features_used_for_projection": False,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_cluster_transition_graphs(
    soft_candidate: InputCandidate,
    membership_candidate: InputCandidate,
    order_candidate: InputCandidate,
    output_dir: Path,
    alpha: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
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

    output_paths: dict[str, Path] = {
        "utterance_cluster_soft_labels": output_dir / "utterance_cluster_soft_labels.csv",
        "utterance_cluster_summary": output_dir / "utterance_cluster_summary.csv",
        "report": output_dir / "cluster_transition_report.md",
        "summary_json": output_dir / "cluster_transition_summary.json",
    }
    write_projected_rows(output_paths["utterance_cluster_soft_labels"], projection)
    write_utterance_summary(output_paths["utterance_cluster_summary"], projection)

    conditional_by_type: dict[str, list[ConditionalRow]] = {}
    unweighted_conditional_by_type: dict[str, list[ConditionalRow]] = {}
    summaries: dict[str, TransitionSummary] = {}

    for transition_type in TRANSITION_TYPES:
        weighted_counts = transitions.weighted_counts[transition_type]
        unweighted_counts = transitions.unweighted_counts[transition_type]
        conditional_rows = build_conditional_rows(
            transition_type,
            weighted_counts,
            unweighted_counts,
            clusters,
            alpha,
        )
        unweighted_conditional_rows = build_conditional_rows(
            transition_type,
            unweighted_counts,
            unweighted_counts,
            clusters,
            alpha,
        )
        conditional_by_type[transition_type] = conditional_rows
        unweighted_conditional_by_type[transition_type] = unweighted_conditional_rows
        summaries[transition_type] = summarize_transition_type(
            transition_type,
            clusters,
            weighted_counts,
            conditional_rows,
            unweighted_conditional_rows,
            transitions.transition_instances[transition_type],
        )

        output_paths[f"{transition_type}_cluster_soft_counts"] = (
            output_dir / f"{transition_type}_cluster_soft_counts.csv"
        )
        output_paths[f"{transition_type}_cluster_conditional_probs"] = (
            output_dir / f"{transition_type}_cluster_conditional_probs.csv"
        )
        output_paths[f"{transition_type}_cluster_count_heatmap"] = (
            output_dir / f"heatmap__{transition_type}_cluster_soft_counts.csv"
        )
        output_paths[f"{transition_type}_cluster_conditional_heatmap"] = (
            output_dir / f"heatmap__{transition_type}_cluster_conditional_probs.csv"
        )
        write_count_rows(output_paths[f"{transition_type}_cluster_soft_counts"], conditional_rows)
        write_conditional_rows(output_paths[f"{transition_type}_cluster_conditional_probs"], conditional_rows)
        write_heatmap(
            output_paths[f"{transition_type}_cluster_count_heatmap"],
            clusters,
            {(row.source_cluster, row.target_cluster): row.raw_soft_count for row in conditional_rows},
        )
        write_heatmap(
            output_paths[f"{transition_type}_cluster_conditional_heatmap"],
            clusters,
            {(row.source_cluster, row.target_cluster): row.conditional_prob for row in conditional_rows},
        )

        if transition_type in MAIN_TRANSITION_TYPES:
            association_rows = build_association_rows(transition_type, weighted_counts, clusters)
            output_paths[f"{transition_type}_cluster_association_stats"] = (
                output_dir / f"{transition_type}_cluster_association_stats.csv"
            )
            write_association_rows(
                output_paths[f"{transition_type}_cluster_association_stats"],
                association_rows,
            )

    write_summary_json(
        output_paths["summary_json"],
        projection,
        summaries,
        soft_candidate,
        membership_candidate,
        order_candidate,
    )
    write_report(
        output_paths["report"],
        soft_candidate,
        membership_candidate,
        order_candidate,
        projection,
        summaries,
        conditional_by_type,
        output_paths,
    )
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project utterance soft emoji labels to clusters and build role-aware cluster transitions."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to search.")
    parser.add_argument("--soft-label-table", type=Path, default=None, help="Explicit utterance soft-label CSV.")
    parser.add_argument("--membership", type=Path, default=None, help="Explicit emoji-cluster membership CSV.")
    parser.add_argument("--turn-order-table", type=Path, default=None, help="Explicit turn-order CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Laplace smoothing alpha.")
    return parser.parse_args()


def explicit_or_discovered(
    explicit_path: Path | None,
    discovered: InputCandidate,
    inspector: Any,
) -> InputCandidate:
    if explicit_path is None:
        return discovered
    candidate = inspector(explicit_path.resolve())
    if candidate is None:
        raise ValueError(f"Explicit input does not match expected schema: {explicit_path}")
    return candidate


def main() -> None:
    args = parse_args()
    discovered_soft, discovered_membership, discovered_order = discover_input_candidates(args.root.resolve())
    soft_candidate = explicit_or_discovered(args.soft_label_table, discovered_soft, inspect_soft_label_csv)
    membership_candidate = explicit_or_discovered(args.membership, discovered_membership, inspect_membership_csv)
    order_candidate = explicit_or_discovered(args.turn_order_table, discovered_order, inspect_turn_order_csv)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else soft_candidate.path.parent.parent / "cluster_transitions"
    )
    output_paths = build_cluster_transition_graphs(
        soft_candidate=soft_candidate,
        membership_candidate=membership_candidate,
        order_candidate=order_candidate,
        output_dir=output_dir,
        alpha=args.alpha,
    )
    print(f"Soft-label table: {soft_candidate.path}")
    print(f"Membership matrix: {membership_candidate.path}")
    print(f"Turn-order source: {order_candidate.path}")
    print(f"Wrote report: {output_paths['report']}")
    print(f"Wrote utterance cluster labels: {output_paths['utterance_cluster_soft_labels']}")
    print(f"Wrote A2B conditionals: {output_paths['A2B_cluster_conditional_probs']}")
    print(f"Wrote B2A conditionals: {output_paths['B2A_cluster_conditional_probs']}")


if __name__ == "__main__":
    main()
