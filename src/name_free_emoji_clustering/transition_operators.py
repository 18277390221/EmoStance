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

import numpy as np

from .soft_membership import cluster_sort_key


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
MAIN_TRANSITION_TYPES = ("A2B", "B2A")
VALIDATION_TOLERANCE = 1e-6
DEFAULT_ALPHA = 0.1
DEFAULT_BASE_RIDGE = 1.0
DEFAULT_STRONG_RIDGE_MULTIPLIER = 10.0
DEFAULT_MIN_EFFECTIVE_SUPPORT = 30.0
DEFAULT_MIN_SOFT_SUPPORT = 5.0
DEFAULT_TAU_OUT = 0.15
DEFAULT_TOP_K_OUTPUT = 5
DEFAULT_EXAMPLE_LIMIT = 12


@dataclass(frozen=True)
class InputCandidate:
    path: Path
    kind: str
    score: int
    fields: tuple[str, ...]
    rationale: str


@dataclass
class UtteranceRecord:
    utterance_id: str
    dialogue_id: str
    turn_id: str
    turn_index: str
    split: str
    role: str
    cluster_probs: dict[str, float] = field(default_factory=dict)
    reliability_weight: float = 1.0
    top1_cluster: str = ""
    top1_cluster_prob: float = 0.0
    cluster_entropy: float = 0.0
    vectors: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class VectorArtifact:
    path: Path
    dimension: int
    score: int
    rationale: str


@dataclass(frozen=True)
class LocalVectorArtifact:
    path: Path
    vectors_by_cluster: dict[str, dict[str, np.ndarray]]
    dimension: int
    score: int
    rationale: str


@dataclass(frozen=True)
class TransitionInstance:
    transition_type: str
    source_index: int
    target_index: int
    reliability_factor: float


@dataclass(frozen=True)
class WeightedSample:
    source_index: int
    target_index: int
    weight: float


@dataclass(frozen=True)
class ConditionalRow:
    transition_type: str
    source_cluster: str
    target_cluster: str
    raw_soft_count: float
    unweighted_soft_count: float
    source_marginal: float
    target_marginal: float
    conditional_prob: float
    pmi: float | None
    lift: float


@dataclass(frozen=True)
class OperatorFit:
    transition_type: str
    source_cluster: str
    target_cluster: str
    operator_matrix: np.ndarray
    bias_vector: np.ndarray
    status: str
    backoff_type: str
    sample_count: int
    soft_support: float
    effective_support: float
    ridge_lambda: float
    train_mse: float
    weighted_train_mse: float


@dataclass(frozen=True)
class OperatorExample:
    transition_type: str
    source_utterance_id: str
    target_utterance_id: str
    source_role: str
    target_role: str
    source_top_clusters: tuple[tuple[str, float], ...]
    target_top_clusters: tuple[tuple[str, float], ...]
    selected_source_cluster: str
    predicted_target_cluster: str
    predicted_target_prob: float
    operator_status: str
    predicted_vector: np.ndarray
    top_target_emojis: tuple[str, ...]
    top_target_emoji_weights: tuple[float, ...]
    affect_vector: np.ndarray


@dataclass(frozen=True)
class TransitionBuildResult:
    utterances: tuple[UtteranceRecord, ...]
    clusters: tuple[str, ...]
    transitions: dict[str, list[TransitionInstance]]
    weighted_counts: dict[str, dict[tuple[str, str], float]]
    unweighted_counts: dict[str, dict[tuple[str, str], float]]
    samples: dict[str, dict[tuple[str, str], list[WeightedSample]]]
    conditional_rows: dict[str, list[ConditionalRow]]
    operator_fits: dict[str, list[OperatorFit]]
    examples: tuple[OperatorExample, ...]
    max_cluster_sum_error: float
    low_support_pairs: dict[str, list[OperatorFit]]


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


def parse_order_value(value: str) -> tuple[int, str]:
    try:
        return int(float(value)), value
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group(0)), str(value)
    return math.inf, str(value)


def role_pair(source_role: str, target_role: str) -> str | None:
    source = source_role.strip().upper()
    target = target_role.strip().upper()
    candidate = f"{source}2{target}"
    return candidate if candidate in MAIN_TRANSITION_TYPES else None


def format_float(value: float) -> str:
    return f"{value:.12f}"


def array_to_json(values: np.ndarray) -> str:
    return json.dumps([round(float(value), 12) for value in values.tolist()], ensure_ascii=False)


def pairs_to_json(values: tuple[tuple[str, float], ...]) -> str:
    return json.dumps(
        [[cluster_id, round(float(probability), 12)] for cluster_id, probability in values],
        ensure_ascii=False,
    )


def tuple_to_json(values: tuple[str, ...] | tuple[float, ...]) -> str:
    rounded: list[str | float] = []
    for value in values:
        if isinstance(value, float):
            rounded.append(round(value, 12))
        else:
            rounded.append(value)
    return json.dumps(rounded, ensure_ascii=False)


def inspect_soft_cluster_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    required = {
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "split",
        "role",
        "cluster_id",
        "cluster_prob",
    }
    if not required.issubset(columns):
        return None

    score = 100
    path_text = str(path).lower()
    if "utterance_cluster_projection" in path_text:
        score += 50
    if path.name == "utterance_cluster_soft_labels.csv":
        score += 30
    if "reliability_weight" in columns:
        score += 15
    if "turn_index" in columns:
        score += 8
    if "cluster_entropy" in columns:
        score += 5
    return InputCandidate(
        path=path,
        kind="utterance_cluster_soft_labels_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has utterance-level soft cluster probabilities and dialogue order fields.",
    )


def inspect_continuous_vector_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    if not header:
        return None
    columns = set(header)
    required = {
        "utterance_id",
        "dialogue_id",
        "turn_id",
        "split",
        "role",
        "cluster_id",
        "cluster_conditioned_vector_json",
    }
    if not required.issubset(columns):
        return None

    score = 100
    path_text = str(path).lower()
    if "utterance_cluster_projection" in path_text:
        score += 50
    if path.name == "utterance_cluster_continuous_vectors.csv":
        score += 30
    if "vector_norm" in columns:
        score += 5
    if "top_contributing_emojis_json" in columns:
        score += 5
    return InputCandidate(
        path=path,
        kind="utterance_cluster_continuous_vectors_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has one cluster-conditioned vector per utterance-cluster pair.",
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
    return InputCandidate(
        path=path,
        kind="emoji_cluster_local_vectors_csv",
        score=score,
        fields=tuple(header),
        rationale="CSV has cluster-local emoji vectors for output distributions.",
    )


def discover_input_candidates(root: Path) -> tuple[InputCandidate, InputCandidate, InputCandidate]:
    soft_cluster_candidates: list[InputCandidate] = []
    continuous_vector_candidates: list[InputCandidate] = []
    local_vector_candidates: list[InputCandidate] = []

    for path in iter_files(root, {".csv"}):
        soft_candidate = inspect_soft_cluster_csv(path)
        if soft_candidate is not None:
            soft_cluster_candidates.append(soft_candidate)

        vector_candidate = inspect_continuous_vector_csv(path)
        if vector_candidate is not None:
            continuous_vector_candidates.append(vector_candidate)

        local_vector_candidate = inspect_local_vector_csv(path)
        if local_vector_candidate is not None:
            local_vector_candidates.append(local_vector_candidate)

    if not soft_cluster_candidates:
        raise FileNotFoundError(
            f"No utterance-level soft cluster label CSV found under {root}. Expected "
            "`utterance_id`, `dialogue_id`, `turn_id`, `role`, `cluster_id`, and "
            "`cluster_prob` columns."
        )
    if not continuous_vector_candidates:
        raise FileNotFoundError(
            f"No utterance-level cluster-conditioned continuous vector CSV found under {root}. "
            "Expected `utterance_id`, `cluster_id`, and `cluster_conditioned_vector_json`."
        )
    if not local_vector_candidates:
        raise FileNotFoundError(
            f"No emoji cluster-local vector CSV found under {root}. Expected `emoji`, "
            "`cluster_id`, and `local_vector_json`."
        )

    def sort_key(candidate: InputCandidate) -> tuple[int, float, str]:
        return (
            candidate.score,
            candidate.path.stat().st_mtime if candidate.path.exists() else 0.0,
            str(candidate.path),
        )

    return (
        sorted(soft_cluster_candidates, key=sort_key, reverse=True)[0],
        sorted(continuous_vector_candidates, key=sort_key, reverse=True)[0],
        sorted(local_vector_candidates, key=sort_key, reverse=True)[0],
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


def load_soft_cluster_labels(path: Path) -> tuple[list[UtteranceRecord], tuple[str, ...], float]:
    utterance_map: OrderedDict[str, UtteranceRecord] = OrderedDict()
    clusters_seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            utterance_id = row.get("utterance_id", "")
            cluster_id = row.get("cluster_id", "")
            probability = parse_float(row.get("cluster_prob"))
            if not utterance_id or not cluster_id or probability is None or probability <= 0:
                continue
            record = utterance_map.get(utterance_id)
            if record is None:
                reliability = parse_float(row.get("reliability_weight"))
                record = UtteranceRecord(
                    utterance_id=utterance_id,
                    dialogue_id=row.get("dialogue_id", ""),
                    turn_id=row.get("turn_id", ""),
                    turn_index=row.get("turn_index", row.get("turn_id", "")),
                    split=row.get("split", ""),
                    role=row.get("role", ""),
                    reliability_weight=reliability if reliability is not None else 1.0,
                    top1_cluster=row.get("top1_cluster", ""),
                    top1_cluster_prob=parse_float(row.get("top1_cluster_prob")) or 0.0,
                    cluster_entropy=parse_float(row.get("cluster_entropy")) or 0.0,
                )
                utterance_map[utterance_id] = record
            record.cluster_probs[cluster_id] = record.cluster_probs.get(cluster_id, 0.0) + probability
            clusters_seen.add(cluster_id)

    if not utterance_map:
        raise ValueError(f"No usable utterance soft cluster labels found in {path}.")

    max_sum_error = 0.0
    for record in utterance_map.values():
        probability_sum = sum(record.cluster_probs.values())
        max_sum_error = max(max_sum_error, abs(probability_sum - 1.0))
        if abs(probability_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Cluster probabilities for {record.utterance_id} sum to "
                f"{probability_sum:.12f}, not 1.0."
            )
        record.cluster_probs = dict(
            sorted(record.cluster_probs.items(), key=lambda item: cluster_sort_key(item[0]))
        )
        if not record.top1_cluster:
            record.top1_cluster, record.top1_cluster_prob = sorted(
                record.cluster_probs.items(),
                key=lambda item: (-item[1], cluster_sort_key(item[0])),
            )[0]

    clusters = tuple(sorted(clusters_seen, key=cluster_sort_key))
    return list(utterance_map.values()), clusters, max_sum_error


def load_utterance_vectors(path: Path, utterance_map: dict[str, UtteranceRecord]) -> VectorArtifact:
    dimension: int | None = None
    loaded_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            utterance_id = row.get("utterance_id", "")
            cluster_id = row.get("cluster_id", "")
            record = utterance_map.get(utterance_id)
            if record is None or not cluster_id:
                continue
            vector = parse_vector_json(row.get("cluster_conditioned_vector_json"), path)
            if vector is None:
                continue
            if dimension is None:
                dimension = int(vector.shape[0])
            elif int(vector.shape[0]) != dimension:
                raise ValueError(
                    f"Continuous vectors in {path} have inconsistent dimensions: "
                    f"{dimension} and {vector.shape[0]}."
                )
            record.vectors[cluster_id] = vector
            loaded_count += 1

    if dimension is None or loaded_count == 0:
        raise ValueError(f"No usable utterance continuous vectors found in {path}.")

    missing_pairs = [
        (record.utterance_id, cluster_id)
        for record in utterance_map.values()
        for cluster_id in record.cluster_probs
        if cluster_id not in record.vectors
    ]
    if missing_pairs:
        preview = ", ".join(
            f"{utterance_id}/{cluster_id}"
            for utterance_id, cluster_id in missing_pairs[:20]
        )
        raise ValueError(f"Missing utterance cluster-conditioned vectors: {preview}")

    return VectorArtifact(
        path=path,
        dimension=dimension,
        score=0,
        rationale="Loaded cluster-conditioned continuous vectors.",
    )


def load_local_vectors(path: Path) -> LocalVectorArtifact:
    vectors_by_cluster: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    dimension: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            emoji = row.get("emoji", "")
            cluster_id = row.get("cluster_id", "")
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
            vectors_by_cluster[cluster_id][emoji] = vector

    if dimension is None or not vectors_by_cluster:
        raise ValueError(f"No usable emoji cluster-local vectors found in {path}.")

    return LocalVectorArtifact(
        path=path,
        vectors_by_cluster={cluster_id: dict(vectors) for cluster_id, vectors in vectors_by_cluster.items()},
        dimension=dimension,
        score=0,
        rationale="Loaded cluster-local emoji vectors for soft output interface.",
    )


def group_dialogues(utterances: Iterable[UtteranceRecord]) -> dict[tuple[str, str], list[int]]:
    grouped: dict[tuple[str, str], list[tuple[int, UtteranceRecord]]] = defaultdict(list)
    for index, record in enumerate(utterances):
        grouped[(record.split, record.dialogue_id)].append((index, record))

    ordered: dict[tuple[str, str], list[int]] = {}
    for key, records in grouped.items():
        records.sort(
            key=lambda item: (
                parse_order_value(item[1].turn_index),
                parse_order_value(item[1].turn_id),
                item[1].utterance_id,
            )
        )
        ordered[key] = [index for index, _ in records]
    return ordered


def build_transition_samples(
    utterances: list[UtteranceRecord],
) -> tuple[
    dict[str, list[TransitionInstance]],
    dict[str, dict[tuple[str, str], float]],
    dict[str, dict[tuple[str, str], float]],
    dict[str, dict[tuple[str, str], list[WeightedSample]]],
]:
    transitions: dict[str, list[TransitionInstance]] = {transition_type: [] for transition_type in MAIN_TRANSITION_TYPES}
    weighted_counts: dict[str, dict[tuple[str, str], float]] = {transition_type: defaultdict(float) for transition_type in MAIN_TRANSITION_TYPES}
    unweighted_counts: dict[str, dict[tuple[str, str], float]] = {transition_type: defaultdict(float) for transition_type in MAIN_TRANSITION_TYPES}
    samples: dict[str, dict[tuple[str, str], list[WeightedSample]]] = {
        transition_type: defaultdict(list)
        for transition_type in MAIN_TRANSITION_TYPES
    }

    for dialogue_indices in group_dialogues(utterances).values():
        for source_index, target_index in zip(dialogue_indices, dialogue_indices[1:]):
            source = utterances[source_index]
            target = utterances[target_index]
            transition_type = role_pair(source.role, target.role)
            if transition_type is None:
                continue
            reliability_factor = math.sqrt(
                max(0.0, source.reliability_weight * target.reliability_weight)
            )
            transitions[transition_type].append(
                TransitionInstance(
                    transition_type=transition_type,
                    source_index=source_index,
                    target_index=target_index,
                    reliability_factor=reliability_factor,
                )
            )
            for source_cluster, source_probability in source.cluster_probs.items():
                for target_cluster, target_probability in target.cluster_probs.items():
                    unweighted_value = source_probability * target_probability
                    weighted_value = unweighted_value * reliability_factor
                    pair = (source_cluster, target_cluster)
                    unweighted_counts[transition_type][pair] += unweighted_value
                    weighted_counts[transition_type][pair] += weighted_value
                    if weighted_value > 0:
                        samples[transition_type][pair].append(
                            WeightedSample(
                                source_index=source_index,
                                target_index=target_index,
                                weight=weighted_value,
                            )
                        )

    return transitions, weighted_counts, unweighted_counts, samples


def source_target_totals(
    counts: dict[tuple[str, str], float],
    clusters: tuple[str, ...],
) -> tuple[dict[str, float], dict[str, float], float]:
    source_totals = {cluster_id: 0.0 for cluster_id in clusters}
    target_totals = {cluster_id: 0.0 for cluster_id in clusters}
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
        probability_sum = 0.0
        for target_cluster in clusters:
            raw_soft_count = weighted_counts.get((source_cluster, target_cluster), 0.0)
            conditional_prob = (
                (raw_soft_count + alpha) / denominator
                if denominator > 0
                else 1.0 / len(clusters)
            )
            source_marginal = source_totals[source_cluster] / total if total > 0 else 0.0
            target_marginal = target_totals[target_cluster] / total if total > 0 else 0.0
            joint_prob = raw_soft_count / total if total > 0 else 0.0
            denominator_for_lift = source_marginal * target_marginal
            if joint_prob > 0 and denominator_for_lift > 0:
                lift = joint_prob / denominator_for_lift
                pmi = math.log2(lift)
            else:
                lift = 0.0
                pmi = None
            probability_sum += conditional_prob
            rows.append(
                ConditionalRow(
                    transition_type=transition_type,
                    source_cluster=source_cluster,
                    target_cluster=target_cluster,
                    raw_soft_count=raw_soft_count,
                    unweighted_soft_count=unweighted_counts.get((source_cluster, target_cluster), 0.0),
                    source_marginal=source_marginal,
                    target_marginal=target_marginal,
                    conditional_prob=conditional_prob,
                    pmi=pmi,
                    lift=lift,
                )
            )
        if abs(probability_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Conditional probabilities for {transition_type} {source_cluster} "
                f"sum to {probability_sum:.12f}."
            )
    return rows


def collect_sample_arrays(
    utterances: tuple[UtteranceRecord, ...] | list[UtteranceRecord],
    samples: list[WeightedSample],
    source_cluster: str,
    target_cluster: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_vectors = np.vstack(
        [utterances[sample.source_index].vectors[source_cluster] for sample in samples]
    )
    target_vectors = np.vstack(
        [utterances[sample.target_index].vectors[target_cluster] for sample in samples]
    )
    sample_weights = np.asarray([sample.weight for sample in samples], dtype=float)
    return source_vectors, target_vectors, sample_weights


def effective_support(sample_weights: np.ndarray) -> float:
    weight_sum = float(sample_weights.sum())
    squared_sum = float(np.dot(sample_weights, sample_weights))
    return (weight_sum * weight_sum / squared_sum) if squared_sum > 0 else 0.0


def solve_weighted_ridge(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    sample_weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    sample_count, dimension = source_vectors.shape
    augmented_source = np.hstack([source_vectors, np.ones((sample_count, 1), dtype=float)])
    sqrt_weights = np.sqrt(np.maximum(sample_weights, 0.0))[:, None]
    weighted_source = augmented_source * sqrt_weights
    weighted_target = target_vectors * sqrt_weights

    normal_matrix = weighted_source.T @ weighted_source
    regularizer = np.eye(dimension + 1, dtype=float) * ridge_lambda
    regularizer[-1, -1] = 0.0
    normal_matrix += regularizer
    right_hand_side = weighted_source.T @ weighted_target
    coefficients = np.linalg.solve(normal_matrix, right_hand_side)

    predicted = augmented_source @ coefficients
    squared_errors = np.sum((predicted - target_vectors) ** 2, axis=1)
    train_mse = float(np.mean(squared_errors)) if sample_count else 0.0
    weight_sum = float(sample_weights.sum())
    weighted_mse = float(np.dot(sample_weights, squared_errors) / weight_sum) if weight_sum > 0 else 0.0
    operator_matrix = coefficients[:-1, :].T
    bias_vector = coefficients[-1, :]
    return operator_matrix, bias_vector, train_mse, weighted_mse


def evaluate_operator(
    source_vectors: np.ndarray,
    target_vectors: np.ndarray,
    sample_weights: np.ndarray,
    operator_matrix: np.ndarray,
    bias_vector: np.ndarray,
) -> tuple[float, float]:
    predicted = source_vectors @ operator_matrix.T + bias_vector
    squared_errors = np.sum((predicted - target_vectors) ** 2, axis=1)
    train_mse = float(np.mean(squared_errors)) if len(squared_errors) else 0.0
    weight_sum = float(sample_weights.sum())
    weighted_mse = float(np.dot(sample_weights, squared_errors) / weight_sum) if weight_sum > 0 else 0.0
    return train_mse, weighted_mse


def fit_operator_from_samples(
    transition_type: str,
    source_cluster: str,
    target_cluster: str,
    utterances: tuple[UtteranceRecord, ...],
    samples: list[WeightedSample],
    ridge_lambda: float,
    status: str,
    backoff_type: str,
) -> OperatorFit:
    source_vectors, target_vectors, sample_weights = collect_sample_arrays(
        utterances,
        samples,
        source_cluster,
        target_cluster,
    )
    operator_matrix, bias_vector, train_mse, weighted_mse = solve_weighted_ridge(
        source_vectors,
        target_vectors,
        sample_weights,
        ridge_lambda,
    )
    return OperatorFit(
        transition_type=transition_type,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        operator_matrix=operator_matrix,
        bias_vector=bias_vector,
        status=status,
        backoff_type=backoff_type,
        sample_count=len(samples),
        soft_support=float(sample_weights.sum()),
        effective_support=effective_support(sample_weights),
        ridge_lambda=ridge_lambda,
        train_mse=train_mse,
        weighted_train_mse=weighted_mse,
    )


def fit_shared_operator(
    transition_type: str,
    utterances: tuple[UtteranceRecord, ...],
    pair_samples: dict[tuple[str, str], list[WeightedSample]],
    dimension: int,
    ridge_lambda: float,
) -> OperatorFit:
    normal_matrix = np.zeros((dimension + 1, dimension + 1), dtype=float)
    right_hand_side = np.zeros((dimension + 1, dimension), dtype=float)
    all_weights: list[np.ndarray] = []
    sample_count = 0
    soft_support = 0.0

    for (source_cluster, target_cluster), samples in pair_samples.items():
        if not samples:
            continue
        source_vectors, target_vectors, sample_weights = collect_sample_arrays(
            utterances,
            samples,
            source_cluster,
            target_cluster,
        )
        augmented_source = np.hstack(
            [source_vectors, np.ones((source_vectors.shape[0], 1), dtype=float)]
        )
        sqrt_weights = np.sqrt(np.maximum(sample_weights, 0.0))[:, None]
        weighted_source = augmented_source * sqrt_weights
        weighted_target = target_vectors * sqrt_weights
        normal_matrix += weighted_source.T @ weighted_source
        right_hand_side += weighted_source.T @ weighted_target
        all_weights.append(sample_weights)
        sample_count += len(samples)
        soft_support += float(sample_weights.sum())

    regularizer = np.eye(dimension + 1, dtype=float) * ridge_lambda
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(normal_matrix + regularizer, right_hand_side)
    operator_matrix = coefficients[:-1, :].T
    bias_vector = coefficients[-1, :]

    squared_error_sum = 0.0
    weighted_squared_error_sum = 0.0
    for (source_cluster, target_cluster), samples in pair_samples.items():
        if not samples:
            continue
        source_vectors, target_vectors, sample_weights = collect_sample_arrays(
            utterances,
            samples,
            source_cluster,
            target_cluster,
        )
        predicted = source_vectors @ operator_matrix.T + bias_vector
        squared_errors = np.sum((predicted - target_vectors) ** 2, axis=1)
        squared_error_sum += float(np.sum(squared_errors))
        weighted_squared_error_sum += float(np.dot(sample_weights, squared_errors))

    concatenated_weights = np.concatenate(all_weights) if all_weights else np.asarray([], dtype=float)
    return OperatorFit(
        transition_type=transition_type,
        source_cluster="shared",
        target_cluster="shared",
        operator_matrix=operator_matrix,
        bias_vector=bias_vector,
        status="shared_operator",
        backoff_type="none",
        sample_count=sample_count,
        soft_support=soft_support,
        effective_support=effective_support(concatenated_weights),
        ridge_lambda=ridge_lambda,
        train_mse=squared_error_sum / sample_count if sample_count else 0.0,
        weighted_train_mse=weighted_squared_error_sum / soft_support if soft_support > 0 else 0.0,
    )


def target_centroid(
    target_cluster: str,
    utterances: tuple[UtteranceRecord, ...],
    pair_samples: dict[tuple[str, str], list[WeightedSample]],
    dimension: int,
) -> np.ndarray:
    numerator = np.zeros(dimension, dtype=float)
    denominator = 0.0
    for (_, sample_target_cluster), samples in pair_samples.items():
        if sample_target_cluster != target_cluster:
            continue
        for sample in samples:
            weight = sample.weight
            numerator += weight * utterances[sample.target_index].vectors[target_cluster]
            denominator += weight
    if denominator <= 0:
        return np.zeros(dimension, dtype=float)
    return numerator / denominator


def backoff_fit(
    transition_type: str,
    source_cluster: str,
    target_cluster: str,
    samples: list[WeightedSample],
    utterances: tuple[UtteranceRecord, ...],
    shared_fit: OperatorFit | None,
    centroid: np.ndarray,
    ridge_lambda: float,
    reason: str,
) -> OperatorFit:
    sample_weights = np.asarray([sample.weight for sample in samples], dtype=float)
    if shared_fit is not None and shared_fit.sample_count > 0:
        operator_matrix = shared_fit.operator_matrix
        bias_vector = shared_fit.bias_vector
        backoff_type = "shared_operator"
    else:
        dimension = centroid.shape[0]
        operator_matrix = np.zeros((dimension, dimension), dtype=float)
        bias_vector = centroid
        backoff_type = "target_centroid"

    if samples:
        source_vectors, target_vectors, weights = collect_sample_arrays(
            utterances,
            samples,
            source_cluster,
            target_cluster,
        )
        train_mse, weighted_mse = evaluate_operator(
            source_vectors,
            target_vectors,
            weights,
            operator_matrix,
            bias_vector,
        )
    else:
        train_mse = 0.0
        weighted_mse = 0.0

    return OperatorFit(
        transition_type=transition_type,
        source_cluster=source_cluster,
        target_cluster=target_cluster,
        operator_matrix=operator_matrix,
        bias_vector=bias_vector,
        status=reason,
        backoff_type=backoff_type,
        sample_count=len(samples),
        soft_support=float(sample_weights.sum()) if len(sample_weights) else 0.0,
        effective_support=effective_support(sample_weights) if len(sample_weights) else 0.0,
        ridge_lambda=ridge_lambda,
        train_mse=train_mse,
        weighted_train_mse=weighted_mse,
    )


def fit_transition_operators(
    utterances: tuple[UtteranceRecord, ...],
    clusters: tuple[str, ...],
    samples: dict[str, dict[tuple[str, str], list[WeightedSample]]],
    dimension: int,
    base_ridge: float,
    strong_ridge_multiplier: float,
    min_effective_support: float,
    min_soft_support: float,
) -> tuple[dict[str, list[OperatorFit]], dict[str, list[OperatorFit]], dict[str, OperatorFit]]:
    operator_fits: dict[str, list[OperatorFit]] = {transition_type: [] for transition_type in MAIN_TRANSITION_TYPES}
    low_support_pairs: dict[str, list[OperatorFit]] = {transition_type: [] for transition_type in MAIN_TRANSITION_TYPES}
    shared_fits: dict[str, OperatorFit] = {}

    for transition_type in MAIN_TRANSITION_TYPES:
        shared_fit = fit_shared_operator(
            transition_type,
            utterances,
            samples[transition_type],
            dimension,
            base_ridge,
        )
        shared_fits[transition_type] = shared_fit
        centroids = {
            target_cluster: target_centroid(
                target_cluster,
                utterances,
                samples[transition_type],
                dimension,
            )
            for target_cluster in clusters
        }

        for source_cluster in clusters:
            for target_cluster in clusters:
                pair = (source_cluster, target_cluster)
                pair_samples = samples[transition_type].get(pair, [])
                sample_weights = np.asarray([sample.weight for sample in pair_samples], dtype=float)
                soft_support = float(sample_weights.sum()) if len(sample_weights) else 0.0
                pair_effective_support = effective_support(sample_weights) if len(sample_weights) else 0.0

                if soft_support < min_soft_support or len(pair_samples) < 2:
                    fit = backoff_fit(
                        transition_type,
                        source_cluster,
                        target_cluster,
                        pair_samples,
                        utterances,
                        shared_fit,
                        centroids[target_cluster],
                        base_ridge,
                        "backoff_low_soft_support",
                    )
                    low_support_pairs[transition_type].append(fit)
                else:
                    ridge_lambda = base_ridge
                    status = "learned_pair_operator"
                    if pair_effective_support < min_effective_support:
                        scale = min_effective_support / max(pair_effective_support, 1e-12)
                        ridge_lambda = base_ridge * strong_ridge_multiplier * scale
                        status = "learned_pair_operator_strong_regularization"
                    fit = fit_operator_from_samples(
                        transition_type,
                        source_cluster,
                        target_cluster,
                        utterances,
                        pair_samples,
                        ridge_lambda,
                        status,
                        "none",
                    )
                    if pair_effective_support < min_effective_support:
                        low_support_pairs[transition_type].append(fit)
                operator_fits[transition_type].append(fit)

    return operator_fits, low_support_pairs, shared_fits


def conditional_lookup(rows: list[ConditionalRow]) -> dict[tuple[str, str], ConditionalRow]:
    return {(row.source_cluster, row.target_cluster): row for row in rows}


def operator_lookup(rows: list[OperatorFit]) -> dict[tuple[str, str], OperatorFit]:
    return {(row.source_cluster, row.target_cluster): row for row in rows}


def top_clusters(probabilities: dict[str, float], limit: int = 3) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            probabilities.items(),
            key=lambda item: (-item[1], cluster_sort_key(item[0])),
        )[:limit]
    )


def cosine_scores(query_vector: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:
    query_norm = float(np.linalg.norm(query_vector))
    candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
    denominator = np.maximum(candidate_norms * query_norm, 1e-12)
    return candidate_vectors @ query_vector / denominator


def softmax(values: np.ndarray, tau: float) -> np.ndarray:
    scaled = values / max(tau, 1e-12)
    shifted = scaled - np.max(scaled)
    exp_values = np.exp(shifted)
    total = float(exp_values.sum())
    if total <= 0 or not math.isfinite(total):
        return np.full_like(exp_values, 1.0 / len(exp_values), dtype=float)
    return exp_values / total


def output_distribution_for_cluster(
    target_cluster: str,
    predicted_vector: np.ndarray,
    local_vectors: LocalVectorArtifact,
    tau_out: float,
    top_k: int,
) -> tuple[tuple[str, ...], tuple[float, ...], np.ndarray]:
    emoji_vectors = local_vectors.vectors_by_cluster.get(target_cluster, {})
    if not emoji_vectors:
        return (), (), np.zeros(local_vectors.dimension, dtype=float)

    emojis = tuple(sorted(emoji_vectors))
    matrix = np.vstack([emoji_vectors[emoji] for emoji in emojis])
    scores = cosine_scores(predicted_vector, matrix)
    selected_count = min(top_k, len(emojis))
    top_indices = np.argsort(-scores)[:selected_count]
    selected_scores = scores[top_indices]
    weights = softmax(selected_scores, tau_out)
    selected_matrix = matrix[top_indices]
    affect_vector = weights @ selected_matrix
    selected_emojis = tuple(emojis[index] for index in top_indices.tolist())
    return selected_emojis, tuple(float(weight) for weight in weights.tolist()), affect_vector


def build_examples(
    utterances: tuple[UtteranceRecord, ...],
    transitions: dict[str, list[TransitionInstance]],
    conditional_rows: dict[str, list[ConditionalRow]],
    operator_fits: dict[str, list[OperatorFit]],
    local_vectors: LocalVectorArtifact,
    tau_out: float,
    top_k: int,
    example_limit: int,
) -> tuple[OperatorExample, ...]:
    examples: list[OperatorExample] = []
    desired_by_type = {
        "A2B": max(1, math.ceil(example_limit * 2 / 3)),
        "B2A": max(1, example_limit - max(1, math.ceil(example_limit * 2 / 3))),
    }
    for transition_type in MAIN_TRANSITION_TYPES:
        conditional_by_pair = conditional_lookup(conditional_rows[transition_type])
        operators_by_pair = operator_lookup(operator_fits[transition_type])
        sorted_transitions = sorted(
            transitions[transition_type],
            key=lambda item: (
                -item.reliability_factor,
                utterances[item.source_index].utterance_id,
                utterances[item.target_index].utterance_id,
            ),
        )
        type_examples = 0
        for transition in sorted_transitions:
            if type_examples >= desired_by_type[transition_type]:
                break
            source = utterances[transition.source_index]
            target = utterances[transition.target_index]
            source_cluster = top_clusters(source.cluster_probs, 1)[0][0]
            source_vector = source.vectors.get(source_cluster)
            if source_vector is None:
                continue
            target_candidates = [
                conditional_by_pair[(source_cluster, target_cluster)]
                for target_cluster in source.cluster_probs.keys() | target.cluster_probs.keys() | {
                    row.target_cluster for row in conditional_rows[transition_type]
                    if row.source_cluster == source_cluster
                }
                if (source_cluster, target_cluster) in conditional_by_pair
            ]
            if not target_candidates:
                continue
            predicted_row = sorted(
                target_candidates,
                key=lambda row: (-row.conditional_prob, cluster_sort_key(row.target_cluster)),
            )[0]
            target_cluster = predicted_row.target_cluster
            fit = operators_by_pair.get((source_cluster, target_cluster))
            if fit is None:
                continue
            predicted_vector = source_vector @ fit.operator_matrix.T + fit.bias_vector
            target_emojis, target_weights, affect_vector = output_distribution_for_cluster(
                target_cluster,
                predicted_vector,
                local_vectors,
                tau_out,
                top_k,
            )
            examples.append(
                OperatorExample(
                    transition_type=transition_type,
                    source_utterance_id=source.utterance_id,
                    target_utterance_id=target.utterance_id,
                    source_role=source.role,
                    target_role=target.role,
                    source_top_clusters=top_clusters(source.cluster_probs),
                    target_top_clusters=top_clusters(target.cluster_probs),
                    selected_source_cluster=source_cluster,
                    predicted_target_cluster=target_cluster,
                    predicted_target_prob=predicted_row.conditional_prob,
                    operator_status=fit.status,
                    predicted_vector=predicted_vector,
                    top_target_emojis=target_emojis,
                    top_target_emoji_weights=target_weights,
                    affect_vector=affect_vector,
                )
            )
            type_examples += 1
    return tuple(examples[:example_limit])


def validate_conditional_rows(rows: list[ConditionalRow], clusters: tuple[str, ...]) -> None:
    rows_by_source: dict[str, list[ConditionalRow]] = defaultdict(list)
    for row in rows:
        rows_by_source[row.source_cluster].append(row)
    for source_cluster in clusters:
        probability_sum = sum(row.conditional_prob for row in rows_by_source[source_cluster])
        if abs(probability_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Conditional probabilities for {source_cluster} sum to "
                f"{probability_sum:.12f}, not 1.0."
            )


def build_transition_priors_and_operators(
    utterances: list[UtteranceRecord],
    clusters: tuple[str, ...],
    local_vectors: LocalVectorArtifact,
    alpha: float,
    base_ridge: float,
    strong_ridge_multiplier: float,
    min_effective_support: float,
    min_soft_support: float,
    tau_out: float,
    top_k_output: int,
    example_limit: int,
    max_cluster_sum_error: float,
) -> TransitionBuildResult:
    utterance_tuple = tuple(utterances)
    transitions, weighted_counts, unweighted_counts, samples = build_transition_samples(utterances)
    conditional_rows: dict[str, list[ConditionalRow]] = {}
    for transition_type in MAIN_TRANSITION_TYPES:
        rows = build_conditional_rows(
            transition_type,
            weighted_counts[transition_type],
            unweighted_counts[transition_type],
            clusters,
            alpha,
        )
        validate_conditional_rows(rows, clusters)
        conditional_rows[transition_type] = rows

    operator_fits, low_support_pairs, _ = fit_transition_operators(
        utterance_tuple,
        clusters,
        samples,
        local_vectors.dimension,
        base_ridge,
        strong_ridge_multiplier,
        min_effective_support,
        min_soft_support,
    )
    examples = build_examples(
        utterance_tuple,
        transitions,
        conditional_rows,
        operator_fits,
        local_vectors,
        tau_out,
        top_k_output,
        example_limit,
    )
    return TransitionBuildResult(
        utterances=utterance_tuple,
        clusters=clusters,
        transitions=transitions,
        weighted_counts={key: dict(value) for key, value in weighted_counts.items()},
        unweighted_counts={key: dict(value) for key, value in unweighted_counts.items()},
        samples={key: {pair: list(rows) for pair, rows in value.items()} for key, value in samples.items()},
        conditional_rows=conditional_rows,
        operator_fits=operator_fits,
        examples=examples,
        max_cluster_sum_error=max_cluster_sum_error,
        low_support_pairs=low_support_pairs,
    )


def write_soft_counts(path: Path, transition_type: str, result: TransitionBuildResult) -> None:
    fieldnames = [
        "transition_type",
        "source_cluster",
        "target_cluster",
        "raw_soft_count",
        "unweighted_soft_count",
        "transition_instances",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source_cluster in result.clusters:
            for target_cluster in result.clusters:
                pair = (source_cluster, target_cluster)
                writer.writerow(
                    {
                        "transition_type": transition_type,
                        "source_cluster": source_cluster,
                        "target_cluster": target_cluster,
                        "raw_soft_count": format_float(result.weighted_counts[transition_type].get(pair, 0.0)),
                        "unweighted_soft_count": format_float(result.unweighted_counts[transition_type].get(pair, 0.0)),
                        "transition_instances": len(result.transitions[transition_type]),
                    }
                )


def write_conditional_probs(path: Path, rows: list[ConditionalRow]) -> None:
    fieldnames = [
        "transition_type",
        "source_cluster",
        "target_cluster",
        "raw_soft_count",
        "unweighted_soft_count",
        "source_marginal",
        "target_marginal",
        "conditional_prob",
        "pmi",
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
                    "unweighted_soft_count": format_float(row.unweighted_soft_count),
                    "source_marginal": format_float(row.source_marginal),
                    "target_marginal": format_float(row.target_marginal),
                    "conditional_prob": format_float(row.conditional_prob),
                    "pmi": format_float(row.pmi) if row.pmi is not None else "",
                    "lift": format_float(row.lift),
                }
            )


def write_operator_table(
    path: Path,
    matrix_path: Path,
    rows: list[OperatorFit],
    conditional_rows: list[ConditionalRow],
) -> None:
    conditionals = conditional_lookup(conditional_rows)
    fieldnames = [
        "transition_type",
        "source_cluster",
        "target_cluster",
        "operator_status",
        "backoff_type",
        "sample_count",
        "soft_support",
        "effective_weighted_support",
        "ridge_lambda",
        "train_mse",
        "weighted_train_mse",
        "conditional_prob",
        "source_marginal",
        "target_marginal",
        "operator_matrix_npz",
        "operator_matrix_key",
        "bias_key",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pair = (row.source_cluster, row.target_cluster)
            conditional = conditionals[pair]
            key_suffix = f"{row.source_cluster}__{row.target_cluster}"
            writer.writerow(
                {
                    "transition_type": row.transition_type,
                    "source_cluster": row.source_cluster,
                    "target_cluster": row.target_cluster,
                    "operator_status": row.status,
                    "backoff_type": row.backoff_type,
                    "sample_count": row.sample_count,
                    "soft_support": format_float(row.soft_support),
                    "effective_weighted_support": format_float(row.effective_support),
                    "ridge_lambda": format_float(row.ridge_lambda),
                    "train_mse": format_float(row.train_mse),
                    "weighted_train_mse": format_float(row.weighted_train_mse),
                    "conditional_prob": format_float(conditional.conditional_prob),
                    "source_marginal": format_float(conditional.source_marginal),
                    "target_marginal": format_float(conditional.target_marginal),
                    "operator_matrix_npz": str(matrix_path),
                    "operator_matrix_key": f"A__{key_suffix}",
                    "bias_key": f"b__{key_suffix}",
                }
            )


def write_operator_matrices(path: Path, rows: list[OperatorFit]) -> None:
    payload: dict[str, np.ndarray] = {}
    for row in rows:
        key_suffix = f"{row.source_cluster}__{row.target_cluster}"
        payload[f"A__{key_suffix}"] = row.operator_matrix
        payload[f"b__{key_suffix}"] = row.bias_vector
    np.savez_compressed(path, **payload)


def write_examples(path: Path, examples: tuple[OperatorExample, ...]) -> None:
    fieldnames = [
        "transition_type",
        "source_utterance_id",
        "target_utterance_id",
        "source_role",
        "target_role",
        "source_top_clusters_json",
        "target_top_clusters_json",
        "selected_source_cluster",
        "predicted_target_cluster",
        "predicted_target_prob",
        "operator_status",
        "predicted_vector_json",
        "top_target_emojis_json",
        "top_target_emoji_weights_json",
        "affect_vector_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "transition_type": example.transition_type,
                    "source_utterance_id": example.source_utterance_id,
                    "target_utterance_id": example.target_utterance_id,
                    "source_role": example.source_role,
                    "target_role": example.target_role,
                    "source_top_clusters_json": pairs_to_json(example.source_top_clusters),
                    "target_top_clusters_json": pairs_to_json(example.target_top_clusters),
                    "selected_source_cluster": example.selected_source_cluster,
                    "predicted_target_cluster": example.predicted_target_cluster,
                    "predicted_target_prob": format_float(example.predicted_target_prob),
                    "operator_status": example.operator_status,
                    "predicted_vector_json": array_to_json(example.predicted_vector),
                    "top_target_emojis_json": tuple_to_json(example.top_target_emojis),
                    "top_target_emoji_weights_json": tuple_to_json(example.top_target_emoji_weights),
                    "affect_vector_json": array_to_json(example.affect_vector),
                }
            )


def transition_density(result: TransitionBuildResult, transition_type: str) -> float:
    possible_edges = len(result.clusters) * len(result.clusters)
    nonzero_edges = sum(
        1
        for value in result.weighted_counts[transition_type].values()
        if value > 0
    )
    return nonzero_edges / possible_edges if possible_edges else 0.0


def strongest_preferences(rows: list[ConditionalRow], limit: int = 6) -> list[ConditionalRow]:
    by_source: dict[str, list[ConditionalRow]] = defaultdict(list)
    for row in rows:
        by_source[row.source_cluster].append(row)
    winners = [
        sorted(source_rows, key=lambda row: (-row.conditional_prob, cluster_sort_key(row.target_cluster)))[0]
        for source_rows in by_source.values()
    ]
    return sorted(winners, key=lambda row: (-row.conditional_prob, cluster_sort_key(row.source_cluster)))[:limit]


def write_summary_json(
    path: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    result: TransitionBuildResult,
    alpha: float,
    base_ridge: float,
    min_effective_support: float,
    min_soft_support: float,
) -> None:
    payload = {
        "inputs": {name: str(path_value) for name, path_value in input_paths.items()},
        "outputs": {name: str(path_value) for name, path_value in output_paths.items()},
        "parameters": {
            "alpha": alpha,
            "base_ridge": base_ridge,
            "min_effective_support": min_effective_support,
            "min_soft_support": min_soft_support,
        },
        "statistics": {
            "clusters": len(result.clusters),
            "utterances": len(result.utterances),
            "max_cluster_sum_error": result.max_cluster_sum_error,
            "transition_instances": {
                transition_type: len(result.transitions[transition_type])
                for transition_type in MAIN_TRANSITION_TYPES
            },
            "transition_density": {
                transition_type: transition_density(result, transition_type)
                for transition_type in MAIN_TRANSITION_TYPES
            },
            "low_support_pairs": {
                transition_type: len(result.low_support_pairs[transition_type])
                for transition_type in MAIN_TRANSITION_TYPES
            },
            "operator_status_counts": {
                transition_type: {
                    status: sum(1 for row in result.operator_fits[transition_type] if row.status == status)
                    for status in sorted({row.status for row in result.operator_fits[transition_type]})
                }
                for transition_type in MAIN_TRANSITION_TYPES
            },
        },
        "hard_constraints": {
            "emoji_names_used": False,
            "transition_information_used_for_text_generation": False,
            "hard_coded_input_paths": False,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_report(
    path: Path,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    result: TransitionBuildResult,
    alpha: float,
    min_effective_support: float,
    min_soft_support: float,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Cluster Transition Priors and Operators\n\n")
        handle.write("## Inputs\n\n")
        for label, input_path in input_paths.items():
            handle.write(f"- {label}: `{input_path}`\n")
        handle.write("- Emoji names were not used, and no text generation was started.\n")
        handle.write("- Dialogue adjacency was derived from discovered split/dialogue/turn-order fields.\n\n")

        handle.write("## Transition Priors\n\n")
        handle.write(f"- Alpha smoothing: `{alpha}`\n")
        handle.write(f"- Clusters: `{len(result.clusters)}`\n")
        handle.write(f"- Max cluster probability sum error: `{result.max_cluster_sum_error:.12g}`\n")
        for transition_type in MAIN_TRANSITION_TYPES:
            total_count = sum(result.weighted_counts[transition_type].values())
            density = transition_density(result, transition_type)
            handle.write(
                f"- `{transition_type}`: `{len(result.transitions[transition_type])}` adjacent turns, "
                f"weighted soft count `{total_count:.6f}`, density `{density:.3f}`.\n"
            )
        handle.write("\n## Strongest Conditional Preferences\n\n")
        for transition_type in MAIN_TRANSITION_TYPES:
            handle.write(f"### {transition_type}\n\n")
            for row in strongest_preferences(result.conditional_rows[transition_type]):
                pmi_text = f"`{row.pmi:.6f}`" if row.pmi is not None else "`NA`"
                handle.write(
                    f"- `{row.source_cluster}` -> `{row.target_cluster}`: "
                    f"T=`{row.conditional_prob:.6f}`, PMI={pmi_text}, "
                    f"lift=`{row.lift:.6f}`"
                )
                handle.write("\n")
            handle.write("\n")

        handle.write("## Operator Diagnostics\n\n")
        handle.write(f"- Low effective support threshold: `{min_effective_support}`\n")
        handle.write(f"- Low soft support backoff threshold: `{min_soft_support}`\n")
        for transition_type in MAIN_TRANSITION_TYPES:
            statuses: dict[str, int] = defaultdict(int)
            for row in result.operator_fits[transition_type]:
                statuses[row.status] += 1
            status_text = ", ".join(f"`{status}`={count}" for status, count in sorted(statuses.items()))
            mean_weighted_mse = (
                sum(row.weighted_train_mse for row in result.operator_fits[transition_type])
                / len(result.operator_fits[transition_type])
            )
            handle.write(
                f"- `{transition_type}` operator statuses: {status_text}; "
                f"mean weighted MSE `{mean_weighted_mse:.6f}`.\n"
            )
            low_rows = sorted(
                result.low_support_pairs[transition_type],
                key=lambda row: (row.soft_support, row.effective_support, cluster_sort_key(row.source_cluster), cluster_sort_key(row.target_cluster)),
            )[:8]
            if low_rows:
                handle.write(f"- `{transition_type}` lowest-support pairs: ")
                handle.write(
                    ", ".join(
                        f"`{row.source_cluster}->{row.target_cluster}` "
                        f"support={row.soft_support:.3f} eff={row.effective_support:.3f}"
                        for row in low_rows
                    )
                )
                handle.write(".\n")
        handle.write("\n## Example Interface\n\n")
        handle.write(
            "- Examples keep a soft target-cluster emoji distribution and an affect vector; "
            "they do not collapse to a nearest emoji.\n"
        )
        for example in result.examples[:6]:
            emoji_preview = ", ".join(
                f"`{emoji}`:{weight:.3f}"
                for emoji, weight in zip(example.top_target_emojis, example.top_target_emoji_weights)
            )
            handle.write(
                f"- `{example.transition_type}` `{example.source_utterance_id}` -> "
                f"`{example.target_utterance_id}` predicts `{example.predicted_target_cluster}` "
                f"with {emoji_preview}.\n"
            )

        handle.write("\n## Outputs\n\n")
        for label, output_path in output_paths.items():
            handle.write(f"- {label}: `{output_path}`\n")


def default_output_dir(vector_path: Path) -> Path:
    parent = vector_path.parent
    if parent.name == "utterance_cluster_projection":
        return parent.parent / "transition_operators"
    return parent / "transition_operators"


def build_and_write_transition_operators(
    root: Path,
    output_dir: Path | None = None,
    alpha: float = DEFAULT_ALPHA,
    base_ridge: float = DEFAULT_BASE_RIDGE,
    strong_ridge_multiplier: float = DEFAULT_STRONG_RIDGE_MULTIPLIER,
    min_effective_support: float = DEFAULT_MIN_EFFECTIVE_SUPPORT,
    min_soft_support: float = DEFAULT_MIN_SOFT_SUPPORT,
    tau_out: float = DEFAULT_TAU_OUT,
    top_k_output: int = DEFAULT_TOP_K_OUTPUT,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
) -> tuple[TransitionBuildResult, dict[str, Path]]:
    soft_candidate, vector_candidate, local_vector_candidate = discover_input_candidates(root)
    utterances, clusters, max_sum_error = load_soft_cluster_labels(soft_candidate.path)
    utterance_map = {record.utterance_id: record for record in utterances}
    vector_artifact = load_utterance_vectors(vector_candidate.path, utterance_map)
    local_vectors = load_local_vectors(local_vector_candidate.path)
    if vector_artifact.dimension != local_vectors.dimension:
        raise ValueError(
            "Utterance continuous vectors and emoji local vectors have different dimensions: "
            f"{vector_artifact.dimension} vs {local_vectors.dimension}."
        )

    result = build_transition_priors_and_operators(
        utterances=utterances,
        clusters=clusters,
        local_vectors=local_vectors,
        alpha=alpha,
        base_ridge=base_ridge,
        strong_ridge_multiplier=strong_ridge_multiplier,
        min_effective_support=min_effective_support,
        min_soft_support=min_soft_support,
        tau_out=tau_out,
        top_k_output=top_k_output,
        example_limit=example_limit,
        max_cluster_sum_error=max_sum_error,
    )

    actual_output_dir = output_dir if output_dir is not None else default_output_dir(vector_candidate.path)
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "A2B_cluster_soft_counts": actual_output_dir / "A2B_cluster_soft_counts.csv",
        "B2A_cluster_soft_counts": actual_output_dir / "B2A_cluster_soft_counts.csv",
        "A2B_cluster_conditional_probs": actual_output_dir / "A2B_cluster_conditional_probs.csv",
        "B2A_cluster_conditional_probs": actual_output_dir / "B2A_cluster_conditional_probs.csv",
        "A2B_operator_table": actual_output_dir / "A2B_operator_table.csv",
        "B2A_operator_table": actual_output_dir / "B2A_operator_table.csv",
        "A2B_operator_matrices": actual_output_dir / "A2B_operator_matrices.npz",
        "B2A_operator_matrices": actual_output_dir / "B2A_operator_matrices.npz",
        "example_transitions": actual_output_dir / "example_transitions.csv",
        "summary_json": actual_output_dir / "transition_operator_summary.json",
        "operator_diagnostics_report": actual_output_dir / "operator_diagnostics_report.md",
    }

    for transition_type in MAIN_TRANSITION_TYPES:
        write_soft_counts(output_paths[f"{transition_type}_cluster_soft_counts"], transition_type, result)
        write_conditional_probs(
            output_paths[f"{transition_type}_cluster_conditional_probs"],
            result.conditional_rows[transition_type],
        )
        write_operator_matrices(
            output_paths[f"{transition_type}_operator_matrices"],
            result.operator_fits[transition_type],
        )
        write_operator_table(
            output_paths[f"{transition_type}_operator_table"],
            output_paths[f"{transition_type}_operator_matrices"],
            result.operator_fits[transition_type],
            result.conditional_rows[transition_type],
        )
    write_examples(output_paths["example_transitions"], result.examples)
    input_paths = {
        "utterance_cluster_soft_labels": soft_candidate.path,
        "utterance_cluster_continuous_vectors": vector_candidate.path,
        "emoji_cluster_local_vectors": local_vector_candidate.path,
    }
    write_summary_json(
        output_paths["summary_json"],
        input_paths,
        output_paths,
        result,
        alpha,
        base_ridge,
        min_effective_support,
        min_soft_support,
    )
    write_report(
        output_paths["operator_diagnostics_report"],
        input_paths,
        output_paths,
        result,
        alpha,
        min_effective_support,
        min_soft_support,
    )
    return result, output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build role-aware A2B/B2A soft cluster transition priors and "
            "regularized continuous transition operators."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to search for projection outputs and local vectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for transition operator outputs. Defaults near discovered projection artifacts.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--base-ridge", type=float, default=DEFAULT_BASE_RIDGE)
    parser.add_argument("--strong-ridge-multiplier", type=float, default=DEFAULT_STRONG_RIDGE_MULTIPLIER)
    parser.add_argument("--min-effective-support", type=float, default=DEFAULT_MIN_EFFECTIVE_SUPPORT)
    parser.add_argument("--min-soft-support", type=float, default=DEFAULT_MIN_SOFT_SUPPORT)
    parser.add_argument("--tau-out", type=float, default=DEFAULT_TAU_OUT)
    parser.add_argument("--top-k-output", type=int, default=DEFAULT_TOP_K_OUTPUT)
    parser.add_argument("--example-limit", type=int, default=DEFAULT_EXAMPLE_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result, output_paths = build_and_write_transition_operators(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        alpha=args.alpha,
        base_ridge=args.base_ridge,
        strong_ridge_multiplier=args.strong_ridge_multiplier,
        min_effective_support=args.min_effective_support,
        min_soft_support=args.min_soft_support,
        tau_out=args.tau_out,
        top_k_output=max(1, args.top_k_output),
        example_limit=max(1, args.example_limit),
    )
    print(f"Wrote A2B counts: {output_paths['A2B_cluster_soft_counts']}")
    print(f"Wrote B2A counts: {output_paths['B2A_cluster_soft_counts']}")
    print(f"Wrote A2B operators: {output_paths['A2B_operator_table']}")
    print(f"Wrote B2A operators: {output_paths['B2A_operator_table']}")
    print(f"Wrote diagnostics report: {output_paths['operator_diagnostics_report']}")
    print(
        "Summary: "
        f"{len(result.clusters)} clusters, "
        f"{len(result.transitions['A2B'])} A2B transitions, "
        f"{len(result.transitions['B2A'])} B2A transitions, "
        f"{len(result.low_support_pairs['A2B'])} A2B low-support pairs, "
        f"{len(result.low_support_pairs['B2A'])} B2A low-support pairs."
    )


if __name__ == "__main__":
    main()
