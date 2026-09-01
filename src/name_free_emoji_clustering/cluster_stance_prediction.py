from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .embeddings import HashedTfidfSentenceEncoder
from .soft_membership import cluster_sort_key


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
VALIDATION_TOLERANCE = 1e-6
EPS = 1e-12
TEXT_TAU_GRID = (0.05, 0.08, 0.1, 0.15, 0.2, 0.35, 0.5, 0.8)
ALPHA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


@dataclass(frozen=True)
class InputCandidate:
    path: Path
    score: int
    fields: tuple[str, ...]
    rationale: str


@dataclass
class UtteranceClusterLabel:
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
    reliability_weight: float | None
    text: str = ""
    situation: str = ""


@dataclass(frozen=True)
class PredictionInstance:
    instance_id: str
    dialogue_id: str
    split: str
    source_turn_id: str
    target_turn_id: str
    source_text: str
    target_text: str
    situation: str
    source_cluster_probs: dict[str, float]
    target_cluster_probs: dict[str, float]
    source_top1_cluster: str
    target_top1_cluster: str
    target_top1_cluster_prob: float
    target_cluster_entropy: float
    target_nonzero_cluster_count: int
    source_reliability_weight: float | None
    target_reliability_weight: float | None


@dataclass(frozen=True)
class EvaluationRow:
    setting: str
    split: str
    subset: str
    model: str
    sample_count: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    top3_accuracy: float
    cross_entropy: float
    kl_divergence: float
    brier_score: float


@dataclass(frozen=True)
class BestConfig:
    setting: str
    alpha: float
    beta: float
    text_tau: float
    validation_cross_entropy: float
    validation_macro_f1: float


@dataclass(frozen=True)
class PredictionRecord:
    setting: str
    split: str
    model: str
    instance_id: str
    dialogue_id: str
    source_turn_id: str
    target_turn_id: str
    source_top1_cluster: str
    target_top1_cluster: str
    predicted_top1_cluster: str
    target_top1_cluster_prob: float
    target_cluster_entropy: float
    predicted_prob_target_top1: float
    cross_entropy: float
    top3_hit: bool
    prediction_probs: dict[str, float]


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.reader(handle), [])
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


def parse_order_value(value: str) -> tuple[int, str]:
    try:
        return int(float(value)), value
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group(0)), str(value)
    return math.inf, str(value)


def entropy(probabilities: Iterable[float]) -> float:
    result = 0.0
    for probability in probabilities:
        if probability > 0:
            result -= probability * math.log(probability)
    return result


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    totals = np.sum(exp_logits, axis=1, keepdims=True)
    return np.divide(exp_logits, totals, out=np.full_like(exp_logits, 1.0 / logits.shape[1]), where=totals > 0)


def normalize_distribution(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(values.astype(float, copy=True), 0.0)
    total = float(np.sum(clipped))
    if total <= 0:
        return np.full_like(clipped, 1.0 / len(clipped), dtype=float)
    return clipped / total


def inspect_cluster_label_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    columns = set(header)
    required = {"dialogue_id", "turn_id", "role", "split", "cluster_id"}
    if not required.issubset(columns):
        return None
    if "cluster_prob" not in columns and "cluster_probability" not in columns:
        return None
    score = 100
    text = str(path).lower()
    if "cluster_transition_analysis" in text:
        score += 45
    if "utterance_cluster_projection" in text:
        score += 35
    if path.name == "utterance_cluster_soft_labels.csv":
        score += 30
    if "reliability_weight" in columns:
        score += 10
    if "cluster_entropy" in columns:
        score += 10
    return InputCandidate(path, score, tuple(header), "utterance-level soft cluster labels")


def inspect_text_csv(path: Path) -> InputCandidate | None:
    header = read_csv_header(path)
    columns = set(header)
    if not {"dialogue_id", "turn_id", "role", "text", "split"}.issubset(columns):
        return None
    score = 80
    text = str(path).lower()
    if "canonical" in text:
        score += 35
    if "soft_label" in text:
        score += 15
    if "situation" in columns:
        score += 20
    if "utterance_id" in columns:
        score += 5
    return InputCandidate(path, score, tuple(header), "utterance text/role/split table")


def inspect_graph_csv(path: Path, transition_type: str) -> InputCandidate | None:
    header = read_csv_header(path)
    columns = set(header)
    if not {"source_cluster", "target_cluster", "conditional_prob"}.issubset(columns):
        return None
    score = 100
    text = str(path).lower()
    if transition_type.lower() in text:
        score += 40
    if "cluster_transition_analysis" in text:
        score += 35
    if "transition_operators" in text:
        score += 25
    if "pmi" in columns and "lift" in columns:
        score += 15
    return InputCandidate(path, score, tuple(header), f"{transition_type} cluster conditional graph")


def discover_inputs(root: Path) -> tuple[InputCandidate, InputCandidate, InputCandidate, InputCandidate | None]:
    cluster_candidates: list[InputCandidate] = []
    text_candidates: list[InputCandidate] = []
    a2b_candidates: list[InputCandidate] = []
    b2a_candidates: list[InputCandidate] = []
    for path in iter_files(root, {".csv"}):
        cluster_candidate = inspect_cluster_label_csv(path)
        if cluster_candidate is not None:
            cluster_candidates.append(cluster_candidate)
        text_candidate = inspect_text_csv(path)
        if text_candidate is not None:
            text_candidates.append(text_candidate)
        a2b_candidate = inspect_graph_csv(path, "A2B")
        if a2b_candidate is not None:
            a2b_candidates.append(a2b_candidate)
        b2a_candidate = inspect_graph_csv(path, "B2A")
        if b2a_candidate is not None:
            b2a_candidates.append(b2a_candidate)

    if not cluster_candidates:
        raise FileNotFoundError("No utterance-level soft cluster label table found.")
    if not text_candidates:
        raise FileNotFoundError("No utterance text/role/split table found.")
    if not a2b_candidates:
        raise FileNotFoundError("No A2B cluster conditional probability table found.")

    def sort_key(candidate: InputCandidate) -> tuple[int, float, str]:
        return (
            candidate.score,
            candidate.path.stat().st_mtime if candidate.path.exists() else 0.0,
            str(candidate.path),
        )

    return (
        sorted(cluster_candidates, key=sort_key, reverse=True)[0],
        sorted(text_candidates, key=sort_key, reverse=True)[0],
        sorted(a2b_candidates, key=sort_key, reverse=True)[0],
        sorted(b2a_candidates, key=sort_key, reverse=True)[0] if b2a_candidates else None,
    )


def load_cluster_labels(path: Path) -> tuple[list[UtteranceClusterLabel], tuple[str, ...]]:
    groups: OrderedDict[str, UtteranceClusterLabel] = OrderedDict()
    clusters: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            utterance_id = make_utterance_id(row)
            cluster_id = row.get("cluster_id", "")
            probability = parse_float(row.get("cluster_prob") or row.get("cluster_probability"))
            if not utterance_id or not cluster_id or probability is None or probability <= 0:
                continue
            label = groups.get(utterance_id)
            if label is None:
                label = UtteranceClusterLabel(
                    utterance_id=utterance_id,
                    dialogue_id=row.get("dialogue_id", ""),
                    turn_id=row.get("turn_id", ""),
                    turn_index=row.get("turn_index", row.get("turn_id", "")),
                    split=row.get("split", ""),
                    role=row.get("role", ""),
                    cluster_probs={},
                    top1_cluster=row.get("top1_cluster", ""),
                    top1_cluster_prob=parse_float(row.get("top1_cluster_prob")) or 0.0,
                    cluster_entropy=parse_float(row.get("cluster_entropy")) or 0.0,
                    nonzero_cluster_count=int(float(row.get("nonzero_cluster_count") or 0)),
                    reliability_weight=parse_float(row.get("reliability_weight")),
                )
                groups[utterance_id] = label
            label.cluster_probs[cluster_id] = label.cluster_probs.get(cluster_id, 0.0) + probability
            clusters.add(cluster_id)

    for label in groups.values():
        total = sum(label.cluster_probs.values())
        if abs(total - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(f"Cluster distribution for {label.utterance_id} sums to {total:.12f}.")
        label.cluster_probs = dict(sorted(label.cluster_probs.items(), key=lambda item: cluster_sort_key(item[0])))
        if not label.top1_cluster:
            label.top1_cluster, label.top1_cluster_prob = sorted(
                label.cluster_probs.items(), key=lambda item: (-item[1], cluster_sort_key(item[0]))
            )[0]
        if label.cluster_entropy == 0.0 and len(label.cluster_probs) > 1:
            label.cluster_entropy = entropy(label.cluster_probs.values())
        if label.nonzero_cluster_count == 0:
            label.nonzero_cluster_count = sum(1 for value in label.cluster_probs.values() if value > 0)
    return list(groups.values()), tuple(sorted(clusters, key=cluster_sort_key))


def load_text_table(path: Path) -> dict[tuple[str, str, str], tuple[str, str]]:
    texts: dict[tuple[str, str, str], tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("split", ""), row.get("dialogue_id", ""), str(row.get("turn_id", "")))
            text = row.get("text", "")
            situation = row.get("situation", "")
            if key not in texts or text:
                texts[key] = (text, situation)
    return texts


def discover_situations(root: Path) -> dict[tuple[str, str], str]:
    situations: dict[tuple[str, str], str] = {}
    for path in iter_files(root, {".json"}):
        if "pre_data" not in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            split = str(item.get("split", ""))
            dialogue_id = str(item.get("dialogue_id", ""))
            situation = str(item.get("situation", ""))
            if split and dialogue_id and situation:
                situations[(split, dialogue_id)] = situation
    return situations


def enrich_labels_with_text(
    labels: list[UtteranceClusterLabel],
    texts: dict[tuple[str, str, str], tuple[str, str]],
    situations: dict[tuple[str, str], str],
) -> None:
    for label in labels:
        text, inline_situation = texts.get((label.split, label.dialogue_id, str(label.turn_id)), ("", ""))
        label.text = text
        label.situation = inline_situation or situations.get((label.split, label.dialogue_id), "")


def build_a2b_instances(labels: list[UtteranceClusterLabel]) -> list[PredictionInstance]:
    grouped: dict[tuple[str, str], list[UtteranceClusterLabel]] = defaultdict(list)
    for label in labels:
        grouped[(label.split, label.dialogue_id)].append(label)
    instances: list[PredictionInstance] = []
    for dialogue_labels in grouped.values():
        dialogue_labels.sort(key=lambda item: (parse_order_value(item.turn_index), parse_order_value(item.turn_id)))
        for source, target in zip(dialogue_labels, dialogue_labels[1:]):
            if source.role.strip().upper() != "A" or target.role.strip().upper() != "B":
                continue
            instances.append(
                PredictionInstance(
                    instance_id=f"{source.split}|{source.dialogue_id}|{source.turn_id}->{target.turn_id}",
                    dialogue_id=source.dialogue_id,
                    split=source.split,
                    source_turn_id=source.turn_id,
                    target_turn_id=target.turn_id,
                    source_text=source.text,
                    target_text=target.text,
                    situation=source.situation,
                    source_cluster_probs=dict(source.cluster_probs),
                    target_cluster_probs=dict(target.cluster_probs),
                    source_top1_cluster=source.top1_cluster,
                    target_top1_cluster=target.top1_cluster,
                    target_top1_cluster_prob=target.top1_cluster_prob,
                    target_cluster_entropy=target.cluster_entropy,
                    target_nonzero_cluster_count=target.nonzero_cluster_count,
                    source_reliability_weight=source.reliability_weight,
                    target_reliability_weight=target.reliability_weight,
                )
            )
    return instances


def load_graph(path: Path, clusters: tuple[str, ...], transition_type: str) -> dict[str, dict[str, float]]:
    graph: dict[str, dict[str, float]] = {source: {target: 0.0 for target in clusters} for source in clusters}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("transition_type") and row.get("transition_type") != transition_type:
                continue
            source = row.get("source_cluster", "")
            target = row.get("target_cluster", "")
            probability = parse_float(row.get("conditional_prob"))
            if source in graph and target in graph[source] and probability is not None:
                graph[source][target] = probability
    for source, row in graph.items():
        total = sum(row.values())
        if abs(total - 1.0) > VALIDATION_TOLERANCE:
            graph[source] = dict(zip(clusters, normalize_distribution(np.array([row[target] for target in clusters]))))
    return graph


def distribution_array(distribution: dict[str, float], clusters: tuple[str, ...]) -> np.ndarray:
    return np.array([distribution.get(cluster, 0.0) for cluster in clusters], dtype=float)


def graph_prior(instance: PredictionInstance, graph: dict[str, dict[str, float]], clusters: tuple[str, ...]) -> np.ndarray:
    prior = np.zeros(len(clusters), dtype=float)
    for source_cluster, source_probability in instance.source_cluster_probs.items():
        row = graph.get(source_cluster)
        if row is None:
            continue
        prior += source_probability * distribution_array(row, clusters)
    prior = normalize_distribution(prior)
    if abs(float(prior.sum()) - 1.0) > VALIDATION_TOLERANCE:
        raise ValueError(f"Graph prior for {instance.instance_id} does not sum to 1.")
    return prior


def context_text(instance: PredictionInstance, setting: str) -> str:
    if setting == "situation_aware" and instance.situation:
        return f"{instance.situation}\n{instance.source_text}"
    return instance.source_text


def target_matrix(instances: list[PredictionInstance], clusters: tuple[str, ...]) -> np.ndarray:
    return np.vstack([distribution_array(instance.target_cluster_probs, clusters) for instance in instances])


def hard_targets(instances: list[PredictionInstance], cluster_to_index: dict[str, int]) -> np.ndarray:
    return np.array([cluster_to_index[instance.target_top1_cluster] for instance in instances], dtype=int)


class CentroidSoftClassifier:
    def __init__(self, tau: float = 0.15) -> None:
        self.tau = tau
        self.centroids: np.ndarray | None = None
        self.class_prior: np.ndarray | None = None

    def fit(self, features: np.ndarray, soft_targets: np.ndarray) -> None:
        weights = np.sum(soft_targets, axis=0)
        centroids = soft_targets.T @ features
        centroids = np.divide(centroids, np.maximum(weights[:, None], EPS))
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = np.divide(centroids, norms, out=np.zeros_like(centroids), where=norms > 0)
        self.centroids = centroids
        self.class_prior = normalize_distribution(weights)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.centroids is None or self.class_prior is None:
            raise ValueError("Classifier must be fit before prediction.")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        safe = np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)
        logits = safe @ self.centroids.T / max(self.tau, EPS)
        logits += np.log(np.maximum(self.class_prior, EPS))[None, :] * 0.15
        return softmax(logits)


def build_text_features(
    train_instances: list[PredictionInstance],
    all_instances: list[PredictionInstance],
    setting: str,
    clusters: tuple[str, ...],
    include_source_cluster: bool,
) -> tuple[np.ndarray, np.ndarray]:
    encoder = HashedTfidfSentenceEncoder(dim=256)
    train_texts = [context_text(instance, setting) for instance in train_instances]
    all_texts = [context_text(instance, setting) for instance in all_instances]
    encoder.fit(train_texts)
    train_features = encoder.transform(train_texts).astype(float)
    all_features = encoder.transform(all_texts).astype(float)
    if include_source_cluster:
        train_source = np.vstack([distribution_array(instance.source_cluster_probs, clusters) for instance in train_instances])
        all_source = np.vstack([distribution_array(instance.source_cluster_probs, clusters) for instance in all_instances])
        train_features = np.hstack([train_features, train_source])
        all_features = np.hstack([all_features, all_source])
    return train_features, all_features


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(-np.mean(np.sum(y_true * np.log(np.maximum(y_pred, EPS)), axis=1)))


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    target_entropy = -np.sum(y_true * np.log(np.maximum(y_true, EPS)), axis=1)
    ce = -np.sum(y_true * np.log(np.maximum(y_pred, EPS)), axis=1)
    return float(np.mean(ce - target_entropy))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1) / y_true.shape[1]))


def f1_scores(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> tuple[float, float]:
    f1_values: list[float] = []
    supports: list[int] = []
    for class_index in range(class_count):
        tp = int(np.sum((y_true == class_index) & (y_pred == class_index)))
        fp = int(np.sum((y_true != class_index) & (y_pred == class_index)))
        fn = int(np.sum((y_true == class_index) & (y_pred != class_index)))
        support = int(np.sum(y_true == class_index))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        supports.append(support)
    macro = sum(f1_values) / class_count if class_count else 0.0
    total = sum(supports)
    weighted = sum(f1 * support for f1, support in zip(f1_values, supports)) / total if total else 0.0
    return macro, weighted


def evaluate_predictions(
    instances: list[PredictionInstance],
    probabilities: np.ndarray,
    clusters: tuple[str, ...],
    setting: str,
    split: str,
    subset: str,
    model: str,
) -> EvaluationRow:
    cluster_to_index = {cluster: index for index, cluster in enumerate(clusters)}
    y_true_hard = hard_targets(instances, cluster_to_index)
    y_true_soft = target_matrix(instances, clusters)
    y_pred_hard = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(y_true_hard == y_pred_hard)) if len(instances) else 0.0
    top3 = np.argsort(-probabilities, axis=1)[:, : min(3, len(clusters))]
    top3_accuracy = float(np.mean([y_true_hard[index] in top3[index] for index in range(len(instances))])) if len(instances) else 0.0
    macro_f1, weighted_f1 = f1_scores(y_true_hard, y_pred_hard, len(clusters))
    return EvaluationRow(
        setting=setting,
        split=split,
        subset=subset,
        model=model,
        sample_count=len(instances),
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        top3_accuracy=top3_accuracy,
        cross_entropy=cross_entropy(y_true_soft, probabilities),
        kl_divergence=kl_divergence(y_true_soft, probabilities),
        brier_score=brier_score(y_true_soft, probabilities),
    )


def subset_mask(instances: list[PredictionInstance], subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones(len(instances), dtype=bool)
    if subset == "high_agreement":
        return np.array([instance.target_top1_cluster_prob >= 0.75 for instance in instances], dtype=bool)
    if subset == "low_agreement":
        entropies = np.array([instance.target_cluster_entropy for instance in instances], dtype=float)
        threshold = float(np.median(entropies)) if len(entropies) else 0.0
        return entropies >= threshold
    raise ValueError(f"Unknown subset: {subset}")


def filter_probabilities(probabilities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return probabilities[mask]


def filter_instances(instances: list[PredictionInstance], mask: np.ndarray) -> list[PredictionInstance]:
    return [instance for instance, keep in zip(instances, mask.tolist()) if keep]


def evaluate_by_groups(
    instances: list[PredictionInstance],
    probabilities: np.ndarray,
    clusters: tuple[str, ...],
    setting: str,
    model: str,
) -> list[EvaluationRow]:
    rows: list[EvaluationRow] = []
    for split in ("train", "valid", "test"):
        split_mask = np.array([instance.split == split for instance in instances], dtype=bool)
        split_instances = filter_instances(instances, split_mask)
        split_probabilities = filter_probabilities(probabilities, split_mask)
        if not split_instances:
            continue
        for subset in ("all", "high_agreement", "low_agreement"):
            mask = subset_mask(split_instances, subset)
            if not np.any(mask):
                continue
            rows.append(
                evaluate_predictions(
                    filter_instances(split_instances, mask),
                    filter_probabilities(split_probabilities, mask),
                    clusters,
                    setting,
                    split,
                    subset,
                    model,
                )
            )
    return rows


def one_hot_majority(instances: list[PredictionInstance], clusters: tuple[str, ...]) -> np.ndarray:
    train_counts = Counter(instance.target_top1_cluster for instance in instances if instance.split == "train")
    majority_cluster = sorted(train_counts.items(), key=lambda item: (-item[1], cluster_sort_key(item[0])))[0][0]
    majority_index = clusters.index(majority_cluster)
    probabilities = np.full((len(instances), len(clusters)), EPS, dtype=float)
    probabilities[:, majority_index] = 1.0
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return probabilities


def markov_probabilities(
    instances: list[PredictionInstance],
    graph: dict[str, dict[str, float]],
    clusters: tuple[str, ...],
    use_soft_source: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for instance in instances:
        if use_soft_source:
            rows.append(graph_prior(instance, graph, clusters))
        else:
            rows.append(distribution_array(graph.get(instance.source_top1_cluster, {}), clusters))
    return np.vstack([normalize_distribution(row) for row in rows])


def fit_text_model(
    train_instances: list[PredictionInstance],
    all_instances: list[PredictionInstance],
    clusters: tuple[str, ...],
    setting: str,
    include_source_cluster: bool,
    tau: float,
) -> np.ndarray:
    train_features, all_features = build_text_features(
        train_instances,
        all_instances,
        setting,
        clusters,
        include_source_cluster=include_source_cluster,
    )
    model = CentroidSoftClassifier(tau=tau)
    model.fit(train_features, target_matrix(train_instances, clusters))
    return model.predict_proba(all_features)


def choose_text_tau(
    train_instances: list[PredictionInstance],
    all_instances: list[PredictionInstance],
    clusters: tuple[str, ...],
    setting: str,
    include_source_cluster: bool,
) -> tuple[float, np.ndarray]:
    best_tau = TEXT_TAU_GRID[0]
    best_probabilities: np.ndarray | None = None
    best_ce = math.inf
    valid_mask = np.array([instance.split == "valid" for instance in all_instances], dtype=bool)
    valid_instances = filter_instances(all_instances, valid_mask)
    for tau in TEXT_TAU_GRID:
        probabilities = fit_text_model(
            train_instances,
            all_instances,
            clusters,
            setting,
            include_source_cluster,
            tau,
        )
        ce = cross_entropy(target_matrix(valid_instances, clusters), probabilities[valid_mask])
        if ce < best_ce:
            best_ce = ce
            best_tau = tau
            best_probabilities = probabilities
    if best_probabilities is None:
        raise ValueError("Could not fit text model.")
    return best_tau, best_probabilities


def graph_poe(text_probabilities: np.ndarray, graph_probabilities: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    logits = alpha * np.log(np.maximum(text_probabilities, EPS)) + beta * np.log(np.maximum(graph_probabilities, EPS))
    return softmax(logits)


def choose_graph_config(
    instances: list[PredictionInstance],
    clusters: tuple[str, ...],
    setting: str,
    text_tau: float,
    text_probabilities: np.ndarray,
    graph_probabilities: np.ndarray,
) -> tuple[BestConfig, np.ndarray]:
    valid_mask = np.array([instance.split == "valid" for instance in instances], dtype=bool)
    valid_instances = filter_instances(instances, valid_mask)
    best: BestConfig | None = None
    best_probabilities: np.ndarray | None = None
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            if alpha == 0 and beta == 0:
                continue
            probabilities = graph_poe(text_probabilities, graph_probabilities, alpha, beta)
            evaluation = evaluate_predictions(
                valid_instances,
                probabilities[valid_mask],
                clusters,
                setting,
                "valid",
                "all",
                "GraphPoE",
            )
            if best is None or evaluation.cross_entropy < best.validation_cross_entropy:
                best = BestConfig(
                    setting=setting,
                    alpha=alpha,
                    beta=beta,
                    text_tau=text_tau,
                    validation_cross_entropy=evaluation.cross_entropy,
                    validation_macro_f1=evaluation.macro_f1,
                )
                best_probabilities = probabilities
    if best is None or best_probabilities is None:
        raise ValueError("No graph PoE config was selected.")
    return best, best_probabilities


def prediction_records(
    instances: list[PredictionInstance],
    probabilities: np.ndarray,
    clusters: tuple[str, ...],
    setting: str,
    model: str,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for instance, row in zip(instances, probabilities):
        target_index = clusters.index(instance.target_top1_cluster)
        top_indices = np.argsort(-row)[: min(3, len(clusters))]
        records.append(
            PredictionRecord(
                setting=setting,
                split=instance.split,
                model=model,
                instance_id=instance.instance_id,
                dialogue_id=instance.dialogue_id,
                source_turn_id=instance.source_turn_id,
                target_turn_id=instance.target_turn_id,
                source_top1_cluster=instance.source_top1_cluster,
                target_top1_cluster=instance.target_top1_cluster,
                predicted_top1_cluster=clusters[int(np.argmax(row))],
                target_top1_cluster_prob=instance.target_top1_cluster_prob,
                target_cluster_entropy=instance.target_cluster_entropy,
                predicted_prob_target_top1=float(row[target_index]),
                cross_entropy=float(-np.sum(distribution_array(instance.target_cluster_probs, clusters) * np.log(np.maximum(row, EPS)))),
                top3_hit=target_index in top_indices,
                prediction_probs={cluster: float(row[index]) for index, cluster in enumerate(clusters)},
            )
        )
    return records


def write_instances(path: Path, instances: list[PredictionInstance]) -> None:
    fieldnames = [
        "instance_id",
        "dialogue_id",
        "split",
        "source_turn_id",
        "target_turn_id",
        "source_text",
        "target_text",
        "situation",
        "source_top1_cluster",
        "target_top1_cluster",
        "target_top1_cluster_prob",
        "target_cluster_entropy",
        "target_nonzero_cluster_count",
        "source_reliability_weight",
        "target_reliability_weight",
        "source_cluster_distribution_json",
        "target_cluster_distribution_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for instance in instances:
            writer.writerow(
                {
                    "instance_id": instance.instance_id,
                    "dialogue_id": instance.dialogue_id,
                    "split": instance.split,
                    "source_turn_id": instance.source_turn_id,
                    "target_turn_id": instance.target_turn_id,
                    "source_text": instance.source_text,
                    "target_text": instance.target_text,
                    "situation": instance.situation,
                    "source_top1_cluster": instance.source_top1_cluster,
                    "target_top1_cluster": instance.target_top1_cluster,
                    "target_top1_cluster_prob": f"{instance.target_top1_cluster_prob:.12f}",
                    "target_cluster_entropy": f"{instance.target_cluster_entropy:.12f}",
                    "target_nonzero_cluster_count": instance.target_nonzero_cluster_count,
                    "source_reliability_weight": "" if instance.source_reliability_weight is None else f"{instance.source_reliability_weight:.12f}",
                    "target_reliability_weight": "" if instance.target_reliability_weight is None else f"{instance.target_reliability_weight:.12f}",
                    "source_cluster_distribution_json": json.dumps(instance.source_cluster_probs, ensure_ascii=False, sort_keys=True),
                    "target_cluster_distribution_json": json.dumps(instance.target_cluster_probs, ensure_ascii=False, sort_keys=True),
                }
            )


def write_metrics(path: Path, rows: list[EvaluationRow]) -> None:
    fieldnames = [
        "setting",
        "split",
        "subset",
        "model",
        "sample_count",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
        "cross_entropy",
        "kl_divergence",
        "brier_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "setting": row.setting,
                    "split": row.split,
                    "subset": row.subset,
                    "model": row.model,
                    "sample_count": row.sample_count,
                    "accuracy": f"{row.accuracy:.12f}",
                    "macro_f1": f"{row.macro_f1:.12f}",
                    "weighted_f1": f"{row.weighted_f1:.12f}",
                    "top3_accuracy": f"{row.top3_accuracy:.12f}",
                    "cross_entropy": f"{row.cross_entropy:.12f}",
                    "kl_divergence": f"{row.kl_divergence:.12f}",
                    "brier_score": f"{row.brier_score:.12f}",
                }
            )


def write_predictions(path: Path, records: list[PredictionRecord]) -> None:
    fieldnames = [
        "setting",
        "split",
        "model",
        "instance_id",
        "dialogue_id",
        "source_turn_id",
        "target_turn_id",
        "source_top1_cluster",
        "target_top1_cluster",
        "predicted_top1_cluster",
        "target_top1_cluster_prob",
        "target_cluster_entropy",
        "predicted_prob_target_top1",
        "cross_entropy",
        "top3_hit",
        "prediction_distribution_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "setting": record.setting,
                    "split": record.split,
                    "model": record.model,
                    "instance_id": record.instance_id,
                    "dialogue_id": record.dialogue_id,
                    "source_turn_id": record.source_turn_id,
                    "target_turn_id": record.target_turn_id,
                    "source_top1_cluster": record.source_top1_cluster,
                    "target_top1_cluster": record.target_top1_cluster,
                    "predicted_top1_cluster": record.predicted_top1_cluster,
                    "target_top1_cluster_prob": f"{record.target_top1_cluster_prob:.12f}",
                    "target_cluster_entropy": f"{record.target_cluster_entropy:.12f}",
                    "predicted_prob_target_top1": f"{record.predicted_prob_target_top1:.12f}",
                    "cross_entropy": f"{record.cross_entropy:.12f}",
                    "top3_hit": record.top3_hit,
                    "prediction_distribution_json": json.dumps(record.prediction_probs, ensure_ascii=False, sort_keys=True),
                }
            )


def write_examples(
    path: Path,
    instances: list[PredictionInstance],
    text_records: dict[str, PredictionRecord],
    graph_records: dict[str, PredictionRecord],
    helped: bool,
    limit: int = 80,
) -> None:
    rows: list[dict[str, Any]] = []
    by_id = {instance.instance_id: instance for instance in instances}
    for instance_id, graph_record in graph_records.items():
        text_record = text_records.get(instance_id)
        if text_record is None:
            continue
        text_correct = text_record.predicted_top1_cluster == text_record.target_top1_cluster
        graph_correct = graph_record.predicted_top1_cluster == graph_record.target_top1_cluster
        if helped and not (not text_correct and graph_correct):
            continue
        if not helped and not (text_correct and not graph_correct):
            continue
        instance = by_id[instance_id]
        rows.append(
            {
                "instance_id": instance_id,
                "split": instance.split,
                "dialogue_id": instance.dialogue_id,
                "source_turn_id": instance.source_turn_id,
                "target_turn_id": instance.target_turn_id,
                "source_text": instance.source_text,
                "target_text": instance.target_text,
                "situation": instance.situation,
                "source_top1_cluster": instance.source_top1_cluster,
                "target_top1_cluster": instance.target_top1_cluster,
                "text_predicted_cluster": text_record.predicted_top1_cluster,
                "graph_predicted_cluster": graph_record.predicted_top1_cluster,
                "text_prob_target": f"{text_record.predicted_prob_target_top1:.12f}",
                "graph_prob_target": f"{graph_record.predicted_prob_target_top1:.12f}",
                "target_cluster_entropy": f"{instance.target_cluster_entropy:.12f}",
            }
        )
    rows.sort(key=lambda row: (row["split"] != "test", -float(row["graph_prob_target"]) if helped else -float(row["text_prob_target"])))
    fieldnames = list(rows[0].keys()) if rows else [
        "instance_id",
        "split",
        "dialogue_id",
        "source_turn_id",
        "target_turn_id",
        "source_text",
        "target_text",
        "situation",
        "source_top1_cluster",
        "target_top1_cluster",
        "text_predicted_cluster",
        "graph_predicted_cluster",
        "text_prob_target",
        "graph_prob_target",
        "target_cluster_entropy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows[:limit]:
            writer.writerow(row)


def bar_chart(path: Path, title: str, rows: list[EvaluationRow], metric: str) -> None:
    selected = [
        row for row in rows
        if row.split == "test" and row.subset == "all" and row.setting == "context_only"
    ]
    order = ["MajorityCluster", "RoleAwareMarkovCluster", "TextOnly", "TextPlusSourceCluster", "GraphPoE"]
    by_model = {row.model: row for row in selected}
    values = [getattr(by_model[model], metric) for model in order if model in by_model]
    labels = [model for model in order if model in by_model]
    width, height = 1000, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 25), title, fill=(20, 20, 20))
    left, top, bottom, right = 90, 80, 120, 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    axis_y = top + plot_height
    max_value = max(values) * 1.15 if values else 1.0
    draw.line((left, top, left, axis_y), fill=(60, 60, 60), width=2)
    draw.line((left, axis_y, width - right, axis_y), fill=(60, 60, 60), width=2)
    for tick in range(6):
        value = max_value * tick / 5
        y = axis_y - int(plot_height * value / max_value)
        draw.line((left - 5, y, width - right, y), fill=(230, 230, 230), width=1)
        draw.text((20, y - 8), f"{value:.2f}", fill=(70, 70, 70))
    slot = plot_width / max(len(values), 1)
    colors = [(102, 126, 234), (72, 187, 120), (237, 137, 54), (159, 122, 234), (229, 62, 62)]
    for index, (label, value) in enumerate(zip(labels, values)):
        center = left + int(slot * (index + 0.5))
        bar_width = int(slot * 0.48)
        bar_height = int(plot_height * value / max_value)
        x0, x1 = center - bar_width // 2, center + bar_width // 2
        y0 = axis_y - bar_height
        draw.rectangle((x0, y0, x1, axis_y), fill=colors[index % len(colors)])
        draw.text((x0, y0 - 22), f"{value:.3f}", fill=(30, 30, 30))
        short = label.replace("Cluster", "").replace("RoleAware", "Markov")
        draw.text((x0 - 8, axis_y + 15), short[:18], fill=(30, 30, 30))
    image.save(path)


def default_output_dir(cluster_label_path: Path) -> Path:
    for parent in cluster_label_path.parents:
        if parent.name == "outputs":
            return parent / "cluster_stance_prediction"
    return cluster_label_path.parent / "cluster_stance_prediction"


def run_prediction(root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    cluster_candidate, text_candidate, a2b_candidate, b2a_candidate = discover_inputs(root)
    labels, clusters = load_cluster_labels(cluster_candidate.path)
    texts = load_text_table(text_candidate.path)
    situations = discover_situations(root)
    enrich_labels_with_text(labels, texts, situations)
    instances = build_a2b_instances(labels)
    if not instances:
        raise ValueError("No A2B prediction instances could be built.")
    a2b_graph = load_graph(a2b_candidate.path, clusters, "A2B")
    train_instances = [instance for instance in instances if instance.split == "train"]
    if not train_instances:
        raise ValueError("No train split A2B instances found.")

    graph_probabilities = markov_probabilities(instances, a2b_graph, clusters, use_soft_source=True)
    majority_probabilities = one_hot_majority(instances, clusters)
    markov_probabilities_top1 = markov_probabilities(instances, a2b_graph, clusters, use_soft_source=False)

    baseline_rows: list[EvaluationRow] = []
    graph_rows: list[EvaluationRow] = []
    graph_prediction_records: list[PredictionRecord] = []
    best_configs: list[BestConfig] = []
    example_context: dict[str, tuple[list[PredictionRecord], list[PredictionRecord]]] = {}

    settings = ["context_only"]
    if any(instance.situation for instance in instances):
        settings.append("situation_aware")

    for setting in settings:
        baseline_rows.extend(evaluate_by_groups(instances, majority_probabilities, clusters, setting, "MajorityCluster"))
        baseline_rows.extend(evaluate_by_groups(instances, markov_probabilities_top1, clusters, setting, "RoleAwareMarkovCluster"))

        text_tau, text_probabilities = choose_text_tau(train_instances, instances, clusters, setting, include_source_cluster=False)
        _, text_plus_probabilities = choose_text_tau(train_instances, instances, clusters, setting, include_source_cluster=True)
        baseline_rows.extend(evaluate_by_groups(instances, text_probabilities, clusters, setting, "TextOnly"))
        baseline_rows.extend(evaluate_by_groups(instances, text_plus_probabilities, clusters, setting, "TextPlusSourceCluster"))

        best_config, graph_poe_probabilities = choose_graph_config(
            instances,
            clusters,
            setting,
            text_tau,
            text_probabilities,
            graph_probabilities,
        )
        best_configs.append(best_config)
        graph_rows.extend(evaluate_by_groups(instances, graph_poe_probabilities, clusters, setting, "GraphPoE"))
        graph_prediction_records.extend(prediction_records(instances, graph_poe_probabilities, clusters, setting, "GraphPoE"))
        if setting == "context_only":
            example_context[setting] = (
                prediction_records(instances, text_probabilities, clusters, setting, "TextOnly"),
                prediction_records(instances, graph_poe_probabilities, clusters, setting, "GraphPoE"),
            )

    actual_output_dir = output_dir if output_dir is not None else default_output_dir(cluster_candidate.path)
    actual_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "a2b_cluster_prediction_instances": actual_output_dir / "a2b_cluster_prediction_instances.csv",
        "cluster_prediction_baseline_metrics": actual_output_dir / "cluster_prediction_baseline_metrics.csv",
        "graph_poe_cluster_metrics": actual_output_dir / "graph_poe_cluster_metrics.csv",
        "graph_poe_best_config": actual_output_dir / "graph_poe_best_config.json",
        "graph_poe_cluster_predictions": actual_output_dir / "graph_poe_cluster_predictions.csv",
        "graph_vs_baseline_report": actual_output_dir / "graph_vs_baseline_report.md",
        "baseline_vs_graph_macro_f1": actual_output_dir / "baseline_vs_graph_macro_f1.png",
        "baseline_vs_graph_soft_ce": actual_output_dir / "baseline_vs_graph_soft_ce.png",
        "graph_helped_examples": actual_output_dir / "graph_helped_examples.csv",
        "graph_hurt_examples": actual_output_dir / "graph_hurt_examples.csv",
    }
    write_instances(output_paths["a2b_cluster_prediction_instances"], instances)
    write_metrics(output_paths["cluster_prediction_baseline_metrics"], baseline_rows)
    write_metrics(output_paths["graph_poe_cluster_metrics"], graph_rows)
    write_predictions(output_paths["graph_poe_cluster_predictions"], graph_prediction_records)
    output_paths["graph_poe_best_config"].write_text(
        json.dumps(
            {
                "configs": [best.__dict__ for best in best_configs],
                "main_graph": str(a2b_candidate.path),
                "b2a_graph_available": str(b2a_candidate.path) if b2a_candidate else None,
                "text_encoder": "HashedTfidfSentenceEncoder",
                "external_emoji_names_used": False,
                "text_generation_used": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cluster_deltas: list[dict[str, float | str | int]] = []
    if "context_only" in example_context:
        text_records, graph_records = example_context["context_only"]
        cluster_deltas = target_cluster_deltas(text_records, graph_records)
        text_by_id = {record.instance_id: record for record in text_records if record.split == "test"}
        graph_by_id = {record.instance_id: record for record in graph_records if record.split == "test"}
        test_instances = [instance for instance in instances if instance.split == "test"]
        write_examples(output_paths["graph_helped_examples"], test_instances, text_by_id, graph_by_id, helped=True)
        write_examples(output_paths["graph_hurt_examples"], test_instances, text_by_id, graph_by_id, helped=False)

    combined_for_plot = baseline_rows + graph_rows
    bar_chart(output_paths["baseline_vs_graph_macro_f1"], "Test Macro-F1: Baselines vs Graph PoE", combined_for_plot, "macro_f1")
    bar_chart(output_paths["baseline_vs_graph_soft_ce"], "Test Soft Cross-Entropy: Baselines vs Graph PoE", combined_for_plot, "cross_entropy")
    write_report(
        output_paths["graph_vs_baseline_report"],
        cluster_candidate,
        text_candidate,
        a2b_candidate,
        b2a_candidate,
        instances,
        baseline_rows,
        graph_rows,
        best_configs,
        cluster_deltas,
        output_paths,
    )
    return output_paths


def metric_lookup(rows: list[EvaluationRow]) -> dict[tuple[str, str, str, str], EvaluationRow]:
    return {(row.setting, row.split, row.subset, row.model): row for row in rows}


def target_cluster_deltas(
    text_records: list[PredictionRecord],
    graph_records: list[PredictionRecord],
) -> list[dict[str, float | str | int]]:
    text_by_id = {record.instance_id: record for record in text_records if record.split == "test"}
    graph_by_id = {record.instance_id: record for record in graph_records if record.split == "test"}
    grouped: dict[str, list[tuple[PredictionRecord, PredictionRecord]]] = defaultdict(list)
    for instance_id, graph_record in graph_by_id.items():
        text_record = text_by_id.get(instance_id)
        if text_record is not None:
            grouped[graph_record.target_top1_cluster].append((text_record, graph_record))
    rows: list[dict[str, float | str | int]] = []
    for target_cluster, pairs in grouped.items():
        text_acc = sum(1 for text_record, _ in pairs if text_record.predicted_top1_cluster == text_record.target_top1_cluster) / len(pairs)
        graph_acc = sum(1 for _, graph_record in pairs if graph_record.predicted_top1_cluster == graph_record.target_top1_cluster) / len(pairs)
        text_ce = sum(text_record.cross_entropy for text_record, _ in pairs) / len(pairs)
        graph_ce = sum(graph_record.cross_entropy for _, graph_record in pairs) / len(pairs)
        rows.append(
            {
                "target_cluster": target_cluster,
                "sample_count": len(pairs),
                "text_accuracy": text_acc,
                "graph_accuracy": graph_acc,
                "accuracy_delta": graph_acc - text_acc,
                "text_cross_entropy": text_ce,
                "graph_cross_entropy": graph_ce,
                "cross_entropy_delta": graph_ce - text_ce,
            }
        )
    return sorted(rows, key=lambda row: (float(row["cross_entropy_delta"]), -int(row["sample_count"])))


def write_report(
    path: Path,
    cluster_candidate: InputCandidate,
    text_candidate: InputCandidate,
    a2b_candidate: InputCandidate,
    b2a_candidate: InputCandidate | None,
    instances: list[PredictionInstance],
    baseline_rows: list[EvaluationRow],
    graph_rows: list[EvaluationRow],
    best_configs: list[BestConfig],
    cluster_deltas: list[dict[str, float | str | int]],
    output_paths: dict[str, Path],
) -> None:
    baseline = metric_lookup(baseline_rows)
    graph = metric_lookup(graph_rows)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Graph PoE Cluster Stance Prediction MVP\n\n")
        handle.write("## Inputs\n\n")
        handle.write(f"- Soft cluster labels: `{cluster_candidate.path}`\n")
        handle.write(f"- Utterance text table: `{text_candidate.path}`\n")
        handle.write(f"- A2B raw cluster graph: `{a2b_candidate.path}`\n")
        if b2a_candidate is not None:
            handle.write(f"- B2A graph available for future ablation: `{b2a_candidate.path}`\n")
        handle.write("- Emoji names, aliases, Unicode names, and text generation are not used; the text model is a lightweight hashed TF-IDF centroid baseline.\n\n")
        handle.write("## Dataset\n\n")
        counts = Counter(instance.split for instance in instances)
        handle.write(f"- A2B instances: `{len(instances)}`\n")
        for split in ("train", "valid", "test"):
            handle.write(f"- `{split}`: `{counts.get(split, 0)}` samples\n")
        handle.write(f"- Situation-aware setting available: `{any(instance.situation for instance in instances)}`\n\n")

        handle.write("## Dev Search\n\n")
        for config in best_configs:
            handle.write(
                f"- `{config.setting}` best alpha=`{config.alpha}`, beta=`{config.beta}`, "
                f"text_tau=`{config.text_tau}`, valid CE=`{config.validation_cross_entropy:.6f}`, "
                f"valid macro-F1=`{config.validation_macro_f1:.6f}`.\n"
            )
        handle.write("\n## Test Metrics\n\n")
        handle.write("| setting | model | accuracy | macro_f1 | weighted_f1 | top3 | soft_ce | KL | brier |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for setting in sorted({row.setting for row in baseline_rows + graph_rows}):
            models = ["MajorityCluster", "RoleAwareMarkovCluster", "TextOnly", "TextPlusSourceCluster", "GraphPoE"]
            for model in models:
                row = baseline.get((setting, "test", "all", model)) or graph.get((setting, "test", "all", model))
                if row is None:
                    continue
                handle.write(
                    f"| `{setting}` | `{model}` | {row.accuracy:.6f} | {row.macro_f1:.6f} | "
                    f"{row.weighted_f1:.6f} | {row.top3_accuracy:.6f} | {row.cross_entropy:.6f} | "
                    f"{row.kl_divergence:.6f} | {row.brier_score:.6f} |\n"
                )
        handle.write("\n## Subset Analysis\n\n")
        for setting in sorted({row.setting for row in graph_rows}):
            for subset in ("high_agreement", "low_agreement"):
                graph_row = graph.get((setting, "test", subset, "GraphPoE"))
                text_row = baseline.get((setting, "test", subset, "TextOnly"))
                if graph_row is None or text_row is None:
                    continue
                handle.write(
                    f"- `{setting}` `{subset}`: GraphPoE macro-F1 `{graph_row.macro_f1:.6f}` vs "
                    f"TextOnly `{text_row.macro_f1:.6f}`; CE `{graph_row.cross_entropy:.6f}` vs "
                    f"`{text_row.cross_entropy:.6f}`.\n"
                )
        handle.write("\n## Interpretation\n\n")
        main_graph = graph.get(("context_only", "test", "all", "GraphPoE"))
        text = baseline.get(("context_only", "test", "all", "TextOnly"))
        markov = baseline.get(("context_only", "test", "all", "RoleAwareMarkovCluster"))
        text_plus = baseline.get(("context_only", "test", "all", "TextPlusSourceCluster"))
        if main_graph is not None and text is not None and markov is not None and text_plus is not None:
            handle.write(
                f"- Graph prior fusion vs TextOnly: macro-F1 delta `{main_graph.macro_f1 - text.macro_f1:.6f}`, "
                f"soft-CE delta `{main_graph.cross_entropy - text.cross_entropy:.6f}`.\n"
            )
            handle.write(
                f"- GraphPoE vs Markov: macro-F1 delta `{main_graph.macro_f1 - markov.macro_f1:.6f}`, "
                f"soft-CE delta `{main_graph.cross_entropy - markov.cross_entropy:.6f}`.\n"
            )
            handle.write(
                f"- GraphPoE vs TextPlusSourceCluster: macro-F1 delta `{main_graph.macro_f1 - text_plus.macro_f1:.6f}`, "
                f"soft-CE delta `{main_graph.cross_entropy - text_plus.cross_entropy:.6f}`.\n"
            )
        handle.write("\n## Target Cluster Effects\n\n")
        if cluster_deltas:
            handle.write("| target_cluster | n | acc_delta | ce_delta | text_ce | graph_ce |\n")
            handle.write("|---|---:|---:|---:|---:|---:|\n")
            for row in cluster_deltas[:8]:
                handle.write(
                    f"| `{row['target_cluster']}` | {row['sample_count']} | "
                    f"{float(row['accuracy_delta']):.6f} | {float(row['cross_entropy_delta']):.6f} | "
                    f"{float(row['text_cross_entropy']):.6f} | {float(row['graph_cross_entropy']):.6f} |\n"
                )
        handle.write(
            "\nInterpretation: this MVP combines the A-turn text representation and the raw A2B cluster graph prior "
            "with logit-level product-of-experts fusion. A positive beta selected on dev indicates that the graph prior "
            "helps soft-distribution calibration. If test macro-F1 does not improve in parallel, the graph is acting more "
            "like a stable global soft prior than a sufficiently fine-grained hard-label discriminator.\n\n"
        )
        handle.write("## Conclusions\n\n")
        if best_configs and all(config.beta <= 0 for config in best_configs):
            handle.write("1. The raw A2B cluster graph should not yet be used directly as a strong main-model prior because dev search did not select a positive beta.\n")
            handle.write("2. B2A is worth testing as an auxiliary ablation, but it should not be added prematurely to the minimal main model.\n")
            handle.write("3. The operator stage should wait until text features are strengthened or the split-safe graph construction is improved.\n")
            handle.write("4. The likely bottleneck is weak text features combined with an overly smooth/global graph prior that cannot adapt enough to specific contexts.\n")
        else:
            handle.write("1. The raw A2B cluster graph is useful as a soft prior for later main models, but should not be used alone as a hard classifier.\n")
            handle.write("2. B2A is worth adding as an auxiliary ablation rather than replacing the A2B main graph.\n")
            handle.write("3. Cluster-local continuous vectors and operator variants can continue, but should be validated alongside stronger text features.\n")
            handle.write("4. The main bottleneck is that both the text centroid baseline and the global graph prior are coarse; soft CE improves clearly, while macro-F1 remains limited.\n")
        handle.write("\n## Outputs\n\n")
        for label, output_path in output_paths.items():
            handle.write(f"- {label}: `{output_path}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP cluster-level stance prediction without text generation.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = run_prediction(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
    )
    print(f"Wrote instances: {output_paths['a2b_cluster_prediction_instances']}")
    print(f"Wrote baseline metrics: {output_paths['cluster_prediction_baseline_metrics']}")
    print(f"Wrote graph metrics: {output_paths['graph_poe_cluster_metrics']}")
    print(f"Wrote report: {output_paths['graph_vs_baseline_report']}")


if __name__ == "__main__":
    main()
