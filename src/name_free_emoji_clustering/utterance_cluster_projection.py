from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .soft_membership import cluster_sort_key


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
VALIDATION_TOLERANCE = 1e-6
DEFAULT_EPS = 1e-12
DEFAULT_TOP_CONTRIBUTORS = 5
MEMBERSHIP_COLUMNS = ("membership_raw", "membership_prob", "membership_sharp")


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
class MembershipArtifact:
    path: Path
    membership: dict[str, dict[str, float]]
    clusters: tuple[str, ...]
    membership_column: str
    score: int
    rationale: str


@dataclass(frozen=True)
class LocalVectorArtifact:
    path: Path
    vectors: dict[tuple[str, str], np.ndarray]
    dimension: int
    score: int
    rationale: str


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
    normalized_cluster_entropy: float
    nonzero_cluster_count: int
    mean_confidence: float | None
    reliability_weight: float


@dataclass(frozen=True)
class ClusterContinuousVector:
    utterance_id: str
    dialogue_id: str
    turn_id: str
    turn_index: str
    split: str
    role: str
    cluster_id: str
    cluster_prob: float
    vector: np.ndarray
    vector_norm: float
    contributing_emoji_count: int
    contribution_entropy: float
    top_contributing_emojis: tuple[str, ...]
    top_contribution_weights: tuple[float, ...]


@dataclass(frozen=True)
class ProjectionResult:
    utterances: tuple[ProjectedUtterance, ...]
    continuous_vectors: tuple[ClusterContinuousVector, ...]
    clusters: tuple[str, ...]
    missing_membership_emojis: dict[str, float]
    missing_local_vectors: dict[tuple[str, str], float]
    max_input_emoji_sum_error: float
    max_cluster_sum_error: float


@dataclass(frozen=True)
class ProjectionSummary:
    utterance_count: int
    cluster_count: int
    utterance_cluster_pair_count: int
    continuous_vector_count: int
    average_nonzero_clusters: float
    average_cluster_entropy: float
    average_top1_cluster_prob: float
    role_average_entropy: dict[str, float]
    role_average_top1_prob: dict[str, float]
    b_more_concentrated_than_a: bool | None
    missing_local_vector_count: int
    missing_local_vector_mass: float


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


def has_utterance_identity(columns: set[str]) -> bool:
    return "utterance_id" in columns or {"dialogue_id", "turn_id"}.issubset(columns)


def make_utterance_id(row: dict[str, str]) -> str:
    utterance_id = row.get("utterance_id")
    if utterance_id:
        return utterance_id
    return f"{row.get('split', '')}|{row.get('dialogue_id', '')}|{row.get('turn_id', '')}"


def entropy(probabilities: Iterable[float]) -> float:
    result = 0.0
    for probability in probabilities:
        if probability > 0:
            result -= probability * math.log(probability)
    return result


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def reliability_weight(
    cluster_entropy: float,
    cluster_count: int,
    mean_confidence: float | None,
) -> float:
    if cluster_count <= 1:
        uncertainty_factor = 1.0
    else:
        uncertainty_factor = 1.0 - cluster_entropy / math.log(cluster_count)
    uncertainty_factor = clamp(uncertainty_factor)
    if mean_confidence is None:
        return uncertainty_factor
    return clamp((mean_confidence / 5.0) * uncertainty_factor)


def inspect_soft_label_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"emoji", "soft_prob", "role", "split"}.issubset(columns):
        return None
    if not has_utterance_identity(columns):
        return None

    score = 100
    path_text = str(path).lower()
    if "canonical" in path_text:
        score += 35
    if "soft_label" in path_text:
        score += 20
    if "tables" in path.parts:
        score += 10
    if "utterance_id" in columns:
        score += 10
    if "turn_index" in columns:
        score += 8
    if "utterance_mean_confidence" in columns:
        score += 12
    if {"top1_prob", "entropy"}.intersection(columns):
        score += 4
    return InputCandidate(
        path=path,
        kind="utterance_soft_label_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has utterance identity, role/split, emoji, and soft_prob columns.",
    )


def inspect_membership_csv(path: Path, requested_column: str) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"emoji", "cluster_id"}.issubset(columns):
        return None
    if not set(MEMBERSHIP_COLUMNS).intersection(columns):
        return None

    score = 100
    path_text = str(path).lower()
    if requested_column in columns:
        score += 30
    if "membership_raw" in columns:
        score += 20
    if "membership_sharp" in columns:
        score += 8
    if "source_type" in columns:
        score += 8
    if "observed_count" in columns:
        score += 4
    if "soft_membership" in path_text:
        score += 25
    if path.name == "emoji_cluster_membership.csv":
        score += 20
    return InputCandidate(
        path=path,
        kind="emoji_cluster_membership_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has one row per emoji-cluster membership probability.",
    )


def inspect_local_vector_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    if not {"emoji", "cluster_id", "local_vector_json"}.issubset(columns):
        return None

    score = 100
    path_text = str(path).lower()
    if "local_vectors" in path_text:
        score += 35
    if path.name == "emoji_cluster_local_vectors.csv":
        score += 25
    if "global_vector_json" in columns:
        score += 8
    if "neighbor_emojis_json" in columns:
        score += 5
    return InputCandidate(
        path=path,
        kind="emoji_cluster_local_vectors_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has one row per emoji-cluster local vector.",
    )


def discover_input_candidates(
    root: Path,
    membership_column: str,
) -> tuple[InputCandidate, InputCandidate, InputCandidate]:
    soft_candidates: list[InputCandidate] = []
    membership_candidates: list[InputCandidate] = []
    local_vector_candidates: list[InputCandidate] = []

    for path in iter_files(root, {".csv"}):
        soft_candidate = inspect_soft_label_csv(path)
        if soft_candidate is not None:
            soft_candidates.append(soft_candidate)

        membership_candidate = inspect_membership_csv(path, membership_column)
        if membership_candidate is not None:
            membership_candidates.append(membership_candidate)

        local_vector_candidate = inspect_local_vector_csv(path)
        if local_vector_candidate is not None:
            local_vector_candidates.append(local_vector_candidate)

    if not soft_candidates:
        raise FileNotFoundError(
            f"No canonical utterance soft-label CSV found under {root}. Expected columns "
            "`emoji`, `soft_prob`, `role`, `split`, and utterance identity columns."
        )
    if not membership_candidates:
        raise FileNotFoundError(
            f"No emoji-cluster membership CSV found under {root}. Expected columns "
            f"`emoji`, `cluster_id`, and one of {MEMBERSHIP_COLUMNS}."
        )
    if not local_vector_candidates:
        raise FileNotFoundError(
            f"No emoji cluster-local vector CSV found under {root}. Expected columns "
            "`emoji`, `cluster_id`, and `local_vector_json`."
        )

    def sort_key(candidate: InputCandidate) -> tuple[int, float, str]:
        return (
            candidate.score,
            candidate.path.stat().st_mtime if candidate.path.exists() else 0.0,
            str(candidate.path),
        )

    return (
        sorted(soft_candidates, key=sort_key, reverse=True)[0],
        sorted(membership_candidates, key=sort_key, reverse=True)[0],
        sorted(local_vector_candidates, key=sort_key, reverse=True)[0],
    )


def select_membership_column(header: list[str], requested_column: str) -> str:
    if requested_column in header:
        return requested_column
    for fallback_column in MEMBERSHIP_COLUMNS:
        if fallback_column in header:
            return fallback_column
    raise ValueError(
        f"Membership table does not contain `{requested_column}` or any fallback "
        f"column in {MEMBERSHIP_COLUMNS}."
    )


def load_membership_artifact(path: Path, requested_column: str, score: int, rationale: str) -> MembershipArtifact:
    membership: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Membership table is empty: {path}")
        membership_column = select_membership_column(reader.fieldnames, requested_column)
        for row in reader:
            emoji = row.get("emoji")
            cluster_id = row.get("cluster_id")
            probability = parse_float(row.get(membership_column))
            if not emoji or not cluster_id or probability is None or probability <= 0:
                continue
            membership[emoji][cluster_id] = membership[emoji].get(cluster_id, 0.0) + probability

    if not membership:
        raise ValueError(f"Membership table contains no usable rows: {path}")

    for emoji, probabilities in membership.items():
        probability_sum = sum(probabilities.values())
        if abs(probability_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Membership probabilities for an emoji sum to {probability_sum:.12f}, not 1.0."
            )

    clusters = tuple(
        sorted(
            {cluster_id for probabilities in membership.values() for cluster_id in probabilities},
            key=cluster_sort_key,
        )
    )
    return MembershipArtifact(
        path=path,
        membership={emoji: dict(probabilities) for emoji, probabilities in membership.items()},
        clusters=clusters,
        membership_column=membership_column,
        score=score,
        rationale=rationale,
    )


def parse_vector_json(value: str | None, path: Path) -> np.ndarray | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse vector JSON in {path}: {exc}") from exc
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        vector = np.asarray([float(item) for item in parsed], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Vector JSON contains non-numeric values in {path}.") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Vector JSON contains non-finite values in {path}.")
    return vector


def load_local_vector_artifact(path: Path, score: int, rationale: str) -> LocalVectorArtifact:
    vectors: dict[tuple[str, str], np.ndarray] = {}
    dimension: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emoji = row.get("emoji")
            cluster_id = row.get("cluster_id")
            vector = parse_vector_json(row.get("local_vector_json"), path)
            if not emoji or not cluster_id or vector is None:
                continue
            if dimension is None:
                dimension = int(vector.shape[0])
            elif int(vector.shape[0]) != dimension:
                raise ValueError(
                    f"Local vectors in {path} have inconsistent dimensions: "
                    f"{dimension} and {vector.shape[0]}."
                )
            vectors[(emoji, cluster_id)] = vector

    if not vectors or dimension is None:
        raise ValueError(f"Local vector table contains no usable rows: {path}")

    return LocalVectorArtifact(
        path=path,
        vectors=vectors,
        dimension=dimension,
        score=score,
        rationale=rationale,
    )


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

            direct_confidence = parse_float(row.get("utterance_mean_confidence"))
            if direct_confidence is not None:
                group.direct_mean_confidence = direct_confidence

            row_confidence = parse_float(row.get("row_mean_confidence"))
            if row_confidence is None:
                row_confidence = parse_float(row.get("mean_confidence"))
            support = parse_float(row.get("row_support_model_count"))
            if support is None:
                support = parse_float(row.get("support_model_count"))
            if row_confidence is not None:
                if support is not None and support > 0:
                    group.confidence_weighted_sum += row_confidence * support
                    group.confidence_weight_total += support
                else:
                    group.confidence_simple_sum += row_confidence
                    group.confidence_simple_count += 1

    max_input_sum_error = 0.0
    utterances: list[WorkingUtterance] = []
    for group in groups.values():
        probability_sum = sum(group.emoji_probs.values())
        if probability_sum <= 0:
            continue
        max_input_sum_error = max(max_input_sum_error, abs(probability_sum - 1.0))
        if abs(probability_sum - 1.0) > VALIDATION_TOLERANCE:
            group.emoji_probs = {
                emoji: probability / probability_sum
                for emoji, probability in group.emoji_probs.items()
            }
        utterances.append(group)

    if not utterances:
        raise ValueError(f"No utterance soft labels could be loaded from {path}.")
    return utterances, max_input_sum_error


def sorted_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    return dict(sorted(probabilities.items(), key=lambda item: cluster_sort_key(item[0])))


def normalize_contributions(
    contributions: list[tuple[str, float]],
    denominator: float,
) -> list[tuple[str, float]]:
    if denominator <= 0:
        return []
    normalized = [
        (emoji, contribution / denominator)
        for emoji, contribution in contributions
        if contribution > 0
    ]
    return sorted(normalized, key=lambda item: (-item[1], item[0]))


def build_projection(
    utterances: list[WorkingUtterance],
    membership: dict[str, dict[str, float]],
    local_vectors: dict[tuple[str, str], np.ndarray],
    clusters: tuple[str, ...],
    dimension: int,
    eps: float = DEFAULT_EPS,
    top_contributors: int = DEFAULT_TOP_CONTRIBUTORS,
) -> ProjectionResult:
    projected_utterances: list[ProjectedUtterance] = []
    continuous_vectors: list[ClusterContinuousVector] = []
    missing_membership_emojis: dict[str, float] = defaultdict(float)
    missing_local_vectors: dict[tuple[str, str], float] = defaultdict(float)
    max_cluster_sum_error = 0.0

    for utterance in utterances:
        cluster_probs: dict[str, float] = defaultdict(float)
        cluster_contributions: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for emoji, emoji_probability in utterance.emoji_probs.items():
            emoji_membership = membership.get(emoji)
            if emoji_membership is None:
                missing_membership_emojis[emoji] += emoji_probability
                continue
            for cluster_id, membership_probability in emoji_membership.items():
                contribution = emoji_probability * membership_probability
                if contribution <= 0:
                    continue
                cluster_probs[cluster_id] += contribution
                cluster_contributions[cluster_id].append((emoji, contribution))

        cluster_total = sum(cluster_probs.values())
        max_cluster_sum_error = max(max_cluster_sum_error, abs(cluster_total - 1.0))
        if cluster_total <= 0:
            continue
        if abs(cluster_total - 1.0) > VALIDATION_TOLERANCE:
            continue

        nonzero_cluster_probs = {
            cluster_id: probability
            for cluster_id, probability in cluster_probs.items()
            if probability > VALIDATION_TOLERANCE
        }
        if not nonzero_cluster_probs:
            continue

        top1_cluster, top1_cluster_prob = sorted(
            nonzero_cluster_probs.items(),
            key=lambda item: (-item[1], cluster_sort_key(item[0])),
        )[0]
        cluster_entropy = entropy(nonzero_cluster_probs.values())
        normalized_entropy = (
            clamp(cluster_entropy / math.log(len(clusters)))
            if len(clusters) > 1
            else 0.0
        )
        reliability = reliability_weight(cluster_entropy, len(clusters), utterance.mean_confidence)
        projected = ProjectedUtterance(
            utterance_id=utterance.utterance_id,
            dialogue_id=utterance.dialogue_id,
            turn_id=utterance.turn_id,
            turn_index=utterance.turn_index,
            split=utterance.split,
            role=utterance.role,
            cluster_probs=sorted_probabilities(nonzero_cluster_probs),
            top1_cluster=top1_cluster,
            top1_cluster_prob=top1_cluster_prob,
            cluster_entropy=cluster_entropy,
            normalized_cluster_entropy=normalized_entropy,
            nonzero_cluster_count=len(nonzero_cluster_probs),
            mean_confidence=utterance.mean_confidence,
            reliability_weight=reliability,
        )
        projected_utterances.append(projected)

        for cluster_id, cluster_probability in projected.cluster_probs.items():
            numerator = np.zeros(dimension, dtype=float)
            missing_mass = 0.0
            contributions = cluster_contributions.get(cluster_id, [])
            for emoji, contribution in contributions:
                vector = local_vectors.get((emoji, cluster_id))
                if vector is None:
                    missing_local_vectors[(emoji, cluster_id)] += contribution
                    missing_mass += contribution
                    continue
                numerator += contribution * vector

            if missing_mass > 0:
                continue

            vector = numerator / (cluster_probability + eps)
            normalized = normalize_contributions(contributions, cluster_probability + eps)
            contribution_weights = [weight for _, weight in normalized]
            continuous_vectors.append(
                ClusterContinuousVector(
                    utterance_id=utterance.utterance_id,
                    dialogue_id=utterance.dialogue_id,
                    turn_id=utterance.turn_id,
                    turn_index=utterance.turn_index,
                    split=utterance.split,
                    role=utterance.role,
                    cluster_id=cluster_id,
                    cluster_prob=cluster_probability,
                    vector=vector,
                    vector_norm=float(np.linalg.norm(vector)),
                    contributing_emoji_count=len(contributions),
                    contribution_entropy=entropy(contribution_weights),
                    top_contributing_emojis=tuple(
                        emoji for emoji, _ in normalized[:top_contributors]
                    ),
                    top_contribution_weights=tuple(
                        weight for _, weight in normalized[:top_contributors]
                    ),
                )
            )

    if missing_membership_emojis:
        preview = ", ".join(
            f"{emoji}:{mass:.6f}"
            for emoji, mass in sorted(
                missing_membership_emojis.items(),
                key=lambda item: (-item[1], item[0]),
            )[:20]
        )
        raise ValueError(
            "Cannot project utterance labels because emojis are missing from the "
            f"membership matrix: {preview}"
        )

    if missing_local_vectors:
        preview = ", ".join(
            f"{emoji}/{cluster_id}:{mass:.6f}"
            for (emoji, cluster_id), mass in sorted(
                missing_local_vectors.items(),
                key=lambda item: (-item[1], item[0][1], item[0][0]),
            )[:20]
        )
        raise ValueError(
            "Cannot build cluster-conditioned vectors because emoji-cluster local "
            f"vectors are missing: {preview}"
        )

    return ProjectionResult(
        utterances=tuple(projected_utterances),
        continuous_vectors=tuple(continuous_vectors),
        clusters=clusters,
        missing_membership_emojis=dict(missing_membership_emojis),
        missing_local_vectors=dict(missing_local_vectors),
        max_input_emoji_sum_error=0.0,
        max_cluster_sum_error=max_cluster_sum_error,
    )


def validate_projection(result: ProjectionResult) -> None:
    for utterance in result.utterances:
        probability_sum = sum(utterance.cluster_probs.values())
        if abs(probability_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Cluster probabilities for {utterance.utterance_id} sum to "
                f"{probability_sum:.12f}, not 1.0."
            )

    expected_vector_keys = {
        (utterance.utterance_id, cluster_id)
        for utterance in result.utterances
        for cluster_id in utterance.cluster_probs
    }
    actual_vector_keys = {
        (vector_row.utterance_id, vector_row.cluster_id)
        for vector_row in result.continuous_vectors
    }
    missing_vectors = expected_vector_keys - actual_vector_keys
    if missing_vectors:
        preview = ", ".join(
            f"{utterance_id}/{cluster_id}"
            for utterance_id, cluster_id in sorted(missing_vectors)[:20]
        )
        raise ValueError(
            "Cluster-conditioned vectors are missing for projected utterance-cluster "
            f"pairs: {preview}"
        )


def summarize_projection(result: ProjectionResult) -> ProjectionSummary:
    utterance_count = len(result.utterances)
    utterance_cluster_pair_count = sum(
        len(utterance.cluster_probs) for utterance in result.utterances
    )
    average_nonzero_clusters = (
        sum(utterance.nonzero_cluster_count for utterance in result.utterances) / utterance_count
        if utterance_count
        else 0.0
    )
    average_cluster_entropy = (
        sum(utterance.cluster_entropy for utterance in result.utterances) / utterance_count
        if utterance_count
        else 0.0
    )
    average_top1_cluster_prob = (
        sum(utterance.top1_cluster_prob for utterance in result.utterances) / utterance_count
        if utterance_count
        else 0.0
    )

    entropy_by_role: dict[str, list[float]] = defaultdict(list)
    top1_by_role: dict[str, list[float]] = defaultdict(list)
    for utterance in result.utterances:
        role = utterance.role.strip().upper() or "UNKNOWN"
        entropy_by_role[role].append(utterance.cluster_entropy)
        top1_by_role[role].append(utterance.top1_cluster_prob)

    role_average_entropy = {
        role: sum(values) / len(values)
        for role, values in sorted(entropy_by_role.items())
        if values
    }
    role_average_top1_prob = {
        role: sum(values) / len(values)
        for role, values in sorted(top1_by_role.items())
        if values
    }

    b_more_concentrated_than_a: bool | None = None
    if "A" in role_average_entropy and "B" in role_average_entropy:
        b_more_concentrated_than_a = (
            role_average_entropy["B"] < role_average_entropy["A"]
            and role_average_top1_prob.get("B", 0.0) > role_average_top1_prob.get("A", 0.0)
        )

    return ProjectionSummary(
        utterance_count=utterance_count,
        cluster_count=len(result.clusters),
        utterance_cluster_pair_count=utterance_cluster_pair_count,
        continuous_vector_count=len(result.continuous_vectors),
        average_nonzero_clusters=average_nonzero_clusters,
        average_cluster_entropy=average_cluster_entropy,
        average_top1_cluster_prob=average_top1_cluster_prob,
        role_average_entropy=role_average_entropy,
        role_average_top1_prob=role_average_top1_prob,
        b_more_concentrated_than_a=b_more_concentrated_than_a,
        missing_local_vector_count=len(result.missing_local_vectors),
        missing_local_vector_mass=sum(result.missing_local_vectors.values()),
    )


def format_float(value: float) -> str:
    return f"{value:.12f}"


def vector_to_json(vector: np.ndarray) -> str:
    return json.dumps([round(float(value), 12) for value in vector.tolist()], ensure_ascii=False)


def tuple_to_json(values: tuple[str, ...] | tuple[float, ...]) -> str:
    rounded: list[str | float] = []
    for value in values:
        if isinstance(value, float):
            rounded.append(round(value, 12))
        else:
            rounded.append(value)
    return json.dumps(rounded, ensure_ascii=False)


def write_utterance_cluster_soft_labels(path: Path, result: ProjectionResult) -> None:
    fieldnames = [
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "turn_index",
        "split",
        "role",
        "cluster_id",
        "cluster_prob",
        "top1_cluster",
        "top1_cluster_prob",
        "cluster_entropy",
        "normalized_cluster_entropy",
        "nonzero_cluster_count",
        "reliability_weight",
        "mean_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for utterance in result.utterances:
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
                        "top1_cluster": utterance.top1_cluster,
                        "top1_cluster_prob": format_float(utterance.top1_cluster_prob),
                        "cluster_entropy": format_float(utterance.cluster_entropy),
                        "normalized_cluster_entropy": format_float(utterance.normalized_cluster_entropy),
                        "nonzero_cluster_count": utterance.nonzero_cluster_count,
                        "reliability_weight": format_float(utterance.reliability_weight),
                        "mean_confidence": (
                            format_float(utterance.mean_confidence)
                            if utterance.mean_confidence is not None
                            else ""
                        ),
                    }
                )


def write_utterance_summary(path: Path, result: ProjectionResult) -> None:
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
        "normalized_cluster_entropy",
        "nonzero_cluster_count",
        "mean_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for utterance in result.utterances:
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
                    "normalized_cluster_entropy": format_float(utterance.normalized_cluster_entropy),
                    "nonzero_cluster_count": utterance.nonzero_cluster_count,
                    "mean_confidence": (
                        format_float(utterance.mean_confidence)
                        if utterance.mean_confidence is not None
                        else ""
                    ),
                }
            )


def write_continuous_vectors(path: Path, result: ProjectionResult) -> None:
    fieldnames = [
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "turn_index",
        "split",
        "role",
        "cluster_id",
        "cluster_prob",
        "vector_norm",
        "contributing_emoji_count",
        "contribution_entropy",
        "top_contributing_emojis_json",
        "top_contribution_weights_json",
        "cluster_conditioned_vector_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for vector_row in result.continuous_vectors:
            writer.writerow(
                {
                    "utterance_id": vector_row.utterance_id,
                    "dialogue_id": vector_row.dialogue_id,
                    "turn_id": vector_row.turn_id,
                    "turn_index": vector_row.turn_index,
                    "split": vector_row.split,
                    "role": vector_row.role,
                    "cluster_id": vector_row.cluster_id,
                    "cluster_prob": format_float(vector_row.cluster_prob),
                    "vector_norm": format_float(vector_row.vector_norm),
                    "contributing_emoji_count": vector_row.contributing_emoji_count,
                    "contribution_entropy": format_float(vector_row.contribution_entropy),
                    "top_contributing_emojis_json": tuple_to_json(vector_row.top_contributing_emojis),
                    "top_contribution_weights_json": tuple_to_json(vector_row.top_contribution_weights),
                    "cluster_conditioned_vector_json": vector_to_json(vector_row.vector),
                }
            )


def write_reliability(path: Path, result: ProjectionResult) -> None:
    fieldnames = [
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "turn_index",
        "split",
        "role",
        "reliability_weight",
        "cluster_entropy",
        "normalized_cluster_entropy",
        "top1_cluster",
        "top1_cluster_prob",
        "nonzero_cluster_count",
        "mean_confidence",
        "confidence_available",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for utterance in result.utterances:
            writer.writerow(
                {
                    "utterance_id": utterance.utterance_id,
                    "dialogue_id": utterance.dialogue_id,
                    "turn_id": utterance.turn_id,
                    "turn_index": utterance.turn_index,
                    "split": utterance.split,
                    "role": utterance.role,
                    "reliability_weight": format_float(utterance.reliability_weight),
                    "cluster_entropy": format_float(utterance.cluster_entropy),
                    "normalized_cluster_entropy": format_float(utterance.normalized_cluster_entropy),
                    "top1_cluster": utterance.top1_cluster,
                    "top1_cluster_prob": format_float(utterance.top1_cluster_prob),
                    "nonzero_cluster_count": utterance.nonzero_cluster_count,
                    "mean_confidence": (
                        format_float(utterance.mean_confidence)
                        if utterance.mean_confidence is not None
                        else ""
                    ),
                    "confidence_available": utterance.mean_confidence is not None,
                }
            )


def mixture_examples(
    result: ProjectionResult,
    limit: int = 8,
) -> list[ClusterContinuousVector]:
    candidates = [
        row
        for row in result.continuous_vectors
        if row.contributing_emoji_count > 1 and row.contribution_entropy > 0
    ]
    return sorted(
        candidates,
        key=lambda row: (-row.contribution_entropy, -row.cluster_prob, row.utterance_id, row.cluster_id),
    )[:limit]


def concentration_sentence(summary: ProjectionSummary) -> str:
    if summary.b_more_concentrated_than_a is None:
        return "A/B concentration could not be compared because one role was absent."
    if summary.b_more_concentrated_than_a:
        return "B turns are more concentrated than A turns by lower entropy and higher top-1 probability."
    return "B turns are not more concentrated than A turns by the entropy/top-1 criterion."


def write_summary_json(
    path: Path,
    soft_label_path: Path,
    membership: MembershipArtifact,
    local_vectors: LocalVectorArtifact,
    result: ProjectionResult,
    summary: ProjectionSummary,
    output_paths: dict[str, Path],
) -> None:
    payload = {
        "inputs": {
            "soft_label_table": str(soft_label_path),
            "membership_matrix": str(membership.path),
            "membership_column": membership.membership_column,
            "local_vectors": str(local_vectors.path),
        },
        "outputs": {name: str(path_value) for name, path_value in output_paths.items()},
        "statistics": {
            "utterance_count": summary.utterance_count,
            "cluster_count": summary.cluster_count,
            "utterance_cluster_pair_count": summary.utterance_cluster_pair_count,
            "continuous_vector_count": summary.continuous_vector_count,
            "average_nonzero_clusters": summary.average_nonzero_clusters,
            "average_cluster_entropy": summary.average_cluster_entropy,
            "average_top1_cluster_prob": summary.average_top1_cluster_prob,
            "role_average_entropy": summary.role_average_entropy,
            "role_average_top1_prob": summary.role_average_top1_prob,
            "b_more_concentrated_than_a": summary.b_more_concentrated_than_a,
            "missing_local_vector_count": summary.missing_local_vector_count,
            "missing_local_vector_mass": summary.missing_local_vector_mass,
            "max_cluster_sum_error": result.max_cluster_sum_error,
            "max_input_emoji_sum_error": result.max_input_emoji_sum_error,
        },
        "hard_constraints": {
            "emoji_names_used": False,
            "emoji_aliases_used": False,
            "unicode_names_used": False,
            "transition_information_used": False,
            "external_emoji_lexicon_used": False,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_report(
    path: Path,
    soft_label_path: Path,
    membership: MembershipArtifact,
    local_vectors: LocalVectorArtifact,
    result: ProjectionResult,
    summary: ProjectionSummary,
    output_paths: dict[str, Path],
) -> None:
    examples = mixture_examples(result)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Utterance Cluster Projection Report\n\n")
        handle.write("## Inputs\n\n")
        handle.write(f"- Soft emoji labels: `{soft_label_path}`.\n")
        handle.write(
            f"- Emoji-cluster membership: `{membership.path}` using "
            f"`{membership.membership_column}`.\n"
        )
        handle.write(f"- Cluster-local vectors: `{local_vectors.path}`.\n")
        handle.write("- Emoji names, aliases, Unicode names, external emoji lexicons, and transition information were not used.\n\n")

        handle.write("## Projection Diagnostics\n\n")
        handle.write(f"- Utterances: `{summary.utterance_count}`\n")
        handle.write(f"- Clusters: `{summary.cluster_count}`\n")
        handle.write(f"- Utterance-cluster pairs: `{summary.utterance_cluster_pair_count}`\n")
        handle.write(f"- Continuous vectors: `{summary.continuous_vector_count}`\n")
        handle.write(f"- Average non-zero clusters per utterance: `{summary.average_nonzero_clusters:.6f}`\n")
        handle.write(f"- Average cluster entropy: `{summary.average_cluster_entropy:.6f}`\n")
        handle.write(f"- Average top-1 cluster probability: `{summary.average_top1_cluster_prob:.6f}`\n")
        handle.write(f"- Max cluster probability sum error: `{result.max_cluster_sum_error:.12g}`\n")
        handle.write(f"- Missing emoji-cluster local vectors: `{summary.missing_local_vector_count}`\n\n")

        handle.write("## Role Concentration\n\n")
        for role, role_entropy in summary.role_average_entropy.items():
            top1_prob = summary.role_average_top1_prob.get(role, 0.0)
            handle.write(
                f"- Role `{role}`: average entropy `{role_entropy:.6f}`, "
                f"average top-1 probability `{top1_prob:.6f}`.\n"
            )
        handle.write(f"- {concentration_sentence(summary)}\n\n")

        handle.write("## Within-Cluster Mixture Examples\n\n")
        if examples:
            handle.write("| utterance_id | role | cluster | cluster_prob | top contributors | weights |\n")
            handle.write("|---|---|---|---:|---|---|\n")
            for row in examples:
                contributors = ", ".join(f"`{emoji}`" for emoji in row.top_contributing_emojis)
                weights = ", ".join(f"`{weight:.3f}`" for weight in row.top_contribution_weights)
                handle.write(
                    f"| `{row.utterance_id}` | `{row.role}` | `{row.cluster_id}` | "
                    f"{row.cluster_prob:.4f} | {contributors} | {weights} |\n"
                )
        else:
            handle.write("No utterance-cluster vector had multiple emoji contributors.\n")
        handle.write("\n## Outputs\n\n")
        for label, output_path in output_paths.items():
            handle.write(f"- {label}: `{output_path}`\n")


def default_output_dir(local_vector_path: Path) -> Path:
    parent = local_vector_path.parent
    if parent.name == "local_vectors":
        return parent.parent / "utterance_cluster_projection"
    return parent / "utterance_cluster_projection"


def build_and_write_projection(
    root: Path,
    output_dir: Path | None = None,
    membership_column: str = "membership_raw",
    eps: float = DEFAULT_EPS,
    top_contributors: int = DEFAULT_TOP_CONTRIBUTORS,
) -> tuple[ProjectionResult, ProjectionSummary, dict[str, Path]]:
    soft_candidate, membership_candidate, local_vector_candidate = discover_input_candidates(
        root,
        membership_column,
    )
    soft_utterances, max_input_sum_error = load_soft_label_utterances(soft_candidate.path)
    membership = load_membership_artifact(
        membership_candidate.path,
        membership_column,
        membership_candidate.score,
        membership_candidate.rationale,
    )
    local_vectors = load_local_vector_artifact(
        local_vector_candidate.path,
        local_vector_candidate.score,
        local_vector_candidate.rationale,
    )
    result = build_projection(
        soft_utterances,
        membership.membership,
        local_vectors.vectors,
        membership.clusters,
        local_vectors.dimension,
        eps=eps,
        top_contributors=top_contributors,
    )
    result = ProjectionResult(
        utterances=result.utterances,
        continuous_vectors=result.continuous_vectors,
        clusters=result.clusters,
        missing_membership_emojis=result.missing_membership_emojis,
        missing_local_vectors=result.missing_local_vectors,
        max_input_emoji_sum_error=max_input_sum_error,
        max_cluster_sum_error=result.max_cluster_sum_error,
    )
    validate_projection(result)
    summary = summarize_projection(result)

    actual_output_dir = output_dir if output_dir is not None else default_output_dir(local_vectors.path)
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "utterance_cluster_soft_labels": actual_output_dir / "utterance_cluster_soft_labels.csv",
        "utterance_cluster_summary": actual_output_dir / "utterance_cluster_summary.csv",
        "utterance_cluster_continuous_vectors": actual_output_dir / "utterance_cluster_continuous_vectors.csv",
        "utterance_cluster_reliability": actual_output_dir / "utterance_cluster_reliability.csv",
        "summary_json": actual_output_dir / "utterance_cluster_projection_summary.json",
        "report": actual_output_dir / "utterance_cluster_projection_report.md",
    }
    write_utterance_cluster_soft_labels(output_paths["utterance_cluster_soft_labels"], result)
    write_utterance_summary(output_paths["utterance_cluster_summary"], result)
    write_continuous_vectors(output_paths["utterance_cluster_continuous_vectors"], result)
    write_reliability(output_paths["utterance_cluster_reliability"], result)
    write_summary_json(
        output_paths["summary_json"],
        soft_candidate.path,
        membership,
        local_vectors,
        result,
        summary,
        output_paths,
    )
    write_report(
        output_paths["report"],
        soft_candidate.path,
        membership,
        local_vectors,
        result,
        summary,
        output_paths,
    )
    return result, summary, output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project utterance-level soft emoji labels into anonymous soft cluster "
            "distributions and cluster-conditioned continuous vectors."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to search for soft labels, membership matrix, and local vectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for projection outputs. Defaults near the discovered local-vector artifact.",
    )
    parser.add_argument(
        "--membership-column",
        default="membership_raw",
        help="Membership column to use when available.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=DEFAULT_EPS,
        help="Small denominator stabilizer for cluster-conditioned vector averages.",
    )
    parser.add_argument(
        "--top-contributors",
        type=int,
        default=DEFAULT_TOP_CONTRIBUTORS,
        help="Number of top emoji contributors to save per utterance-cluster vector.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, summary, output_paths = build_and_write_projection(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        membership_column=args.membership_column,
        eps=args.eps,
        top_contributors=max(1, args.top_contributors),
    )
    print(f"Wrote utterance-cluster soft labels: {output_paths['utterance_cluster_soft_labels']}")
    print(f"Wrote utterance-cluster continuous vectors: {output_paths['utterance_cluster_continuous_vectors']}")
    print(f"Wrote reliability table: {output_paths['utterance_cluster_reliability']}")
    print(f"Wrote report: {output_paths['report']}")
    print(
        "Summary: "
        f"{summary.utterance_count} utterances, "
        f"{len(result.clusters)} clusters, "
        f"{summary.utterance_cluster_pair_count} utterance-cluster pairs, "
        f"average non-zero clusters {summary.average_nonzero_clusters:.3f}."
    )


if __name__ == "__main__":
    main()
