from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .cluster_stance_prediction import (
    ALPHA_GRID,
    BETA_GRID,
    EPS,
    PredictionInstance,
    PredictionRecord,
    choose_text_tau,
    context_text,
    cross_entropy,
    discover_inputs,
    discover_situations,
    distribution_array,
    enrich_labels_with_text,
    evaluate_by_groups,
    evaluate_predictions,
    fit_text_model,
    graph_poe,
    graph_prior,
    hard_targets,
    kl_divergence,
    load_cluster_labels,
    load_graph,
    load_text_table,
    markov_probabilities,
    prediction_records,
    target_matrix,
    write_examples,
    write_metrics,
    write_predictions,
)
from .embeddings import HashedTfidfSentenceEncoder
from .soft_membership import cluster_sort_key


VALIDATION_TOLERANCE = 1e-6
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
TEMP_GRID = (0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0)
RESIDUAL_BETA_GRID = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


@dataclass(frozen=True)
class ModelBundle:
    name: str
    probabilities: np.ndarray
    setting: str


@dataclass(frozen=True)
class StrongTextConfig:
    setting: str
    ridge_lambda: float
    temperature: float
    validation_cross_entropy: float
    validation_macro_f1: float


@dataclass(frozen=True)
class CalibrationConfig:
    setting: str
    method: str
    alpha: float
    beta: float
    validation_cross_entropy: float
    validation_macro_f1: float
    validation_weighted_f1: float
    validation_top3_accuracy: float


@dataclass(frozen=True)
class DiscoveredPriorOutputs:
    instance_path: Path | None
    graph_predictions_path: Path | None
    baseline_metrics_path: Path | None
    graph_metrics_path: Path | None
    best_config_path: Path | None


class RidgeSoftTextClassifier:
    def __init__(self, ridge_lambda: float, temperature: float) -> None:
        self.ridge_lambda = ridge_lambda
        self.temperature = temperature
        self.weights: np.ndarray | None = None
        self.class_prior: np.ndarray | None = None

    def fit(self, features: np.ndarray, soft_targets: np.ndarray) -> None:
        bias = np.ones((features.shape[0], 1), dtype=float)
        augmented = np.hstack([features, bias])
        regularizer = np.eye(augmented.shape[1], dtype=float) * self.ridge_lambda
        regularizer[-1, -1] = 0.0
        lhs = augmented.T @ augmented + regularizer
        rhs = augmented.T @ soft_targets
        self.weights = np.linalg.solve(lhs, rhs)
        self.class_prior = np.maximum(np.mean(soft_targets, axis=0), EPS)
        self.class_prior = self.class_prior / self.class_prior.sum()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None or self.class_prior is None:
            raise ValueError("RidgeSoftTextClassifier must be fit first.")
        augmented = np.hstack([features, np.ones((features.shape[0], 1), dtype=float)])
        logits = augmented @ self.weights
        logits = logits / max(self.temperature, EPS)
        logits = logits + 0.05 * np.log(np.maximum(self.class_prior, EPS))[None, :]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(values, EPS)
    return clipped / np.sum(clipped, axis=1, keepdims=True)


def discover_prior_outputs(root: Path) -> DiscoveredPriorOutputs:
    instance_path = graph_predictions_path = baseline_metrics_path = graph_metrics_path = best_config_path = None
    for path in root.rglob("*"):
        if not path.is_file() or any(part in {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"} for part in path.parts):
            continue
        if path.suffix.lower() == ".csv":
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    header = next(csv.reader(handle), [])
            except Exception:
                continue
            columns = set(header)
            if {"instance_id", "source_text", "target_text", "target_cluster_distribution_json"}.issubset(columns):
                instance_path = path
            if {"setting", "split", "model", "instance_id", "prediction_distribution_json"}.issubset(columns):
                graph_predictions_path = path
            if {"setting", "split", "subset", "model", "macro_f1", "cross_entropy"}.issubset(columns):
                if "graph_poe" in path.name:
                    graph_metrics_path = path
                else:
                    baseline_metrics_path = path
        elif path.suffix.lower() == ".json" and path.name == "graph_poe_best_config.json":
            best_config_path = path
    return DiscoveredPriorOutputs(instance_path, graph_predictions_path, baseline_metrics_path, graph_metrics_path, best_config_path)


def load_instances(path: Path) -> list[PredictionInstance]:
    instances: list[PredictionInstance] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_dist = json.loads(row["source_cluster_distribution_json"])
            target_dist = json.loads(row["target_cluster_distribution_json"])
            if abs(sum(source_dist.values()) - 1.0) > VALIDATION_TOLERANCE:
                raise ValueError(f"Source distribution does not sum to 1 for {row['instance_id']}")
            if abs(sum(target_dist.values()) - 1.0) > VALIDATION_TOLERANCE:
                raise ValueError(f"Target distribution does not sum to 1 for {row['instance_id']}")
            instances.append(
                PredictionInstance(
                    instance_id=row["instance_id"],
                    dialogue_id=row["dialogue_id"],
                    split=row["split"],
                    source_turn_id=row["source_turn_id"],
                    target_turn_id=row["target_turn_id"],
                    source_text=row.get("source_text", ""),
                    target_text=row.get("target_text", ""),
                    situation=row.get("situation", ""),
                    source_cluster_probs={str(k): float(v) for k, v in source_dist.items()},
                    target_cluster_probs={str(k): float(v) for k, v in target_dist.items()},
                    source_top1_cluster=row["source_top1_cluster"],
                    target_top1_cluster=row["target_top1_cluster"],
                    target_top1_cluster_prob=float(row["target_top1_cluster_prob"]),
                    target_cluster_entropy=float(row["target_cluster_entropy"]),
                    target_nonzero_cluster_count=int(float(row["target_nonzero_cluster_count"])),
                    source_reliability_weight=float(row["source_reliability_weight"]) if row.get("source_reliability_weight") else None,
                    target_reliability_weight=float(row["target_reliability_weight"]) if row.get("target_reliability_weight") else None,
                )
            )
    return instances


def infer_clusters(instances: list[PredictionInstance]) -> tuple[str, ...]:
    clusters = {cluster for instance in instances for cluster in instance.target_cluster_probs}
    clusters.update(cluster for instance in instances for cluster in instance.source_cluster_probs)
    return tuple(sorted(clusters, key=cluster_sort_key))


def build_instances_if_needed(root: Path, prior: DiscoveredPriorOutputs) -> tuple[list[PredictionInstance], tuple[str, ...], Path, Path, Path]:
    cluster_candidate, text_candidate, a2b_candidate, _ = discover_inputs(root)
    if prior.instance_path is not None:
        instances = load_instances(prior.instance_path)
        clusters = infer_clusters(instances)
        return instances, clusters, cluster_candidate.path, text_candidate.path, a2b_candidate.path
    labels, clusters = load_cluster_labels(cluster_candidate.path)
    texts = load_text_table(text_candidate.path)
    enrich_labels_with_text(labels, texts, discover_situations(root))
    from .cluster_stance_prediction import build_a2b_instances
    return build_a2b_instances(labels), clusters, cluster_candidate.path, text_candidate.path, a2b_candidate.path


def feature_matrix(train_instances: list[PredictionInstance], all_instances: list[PredictionInstance], setting: str) -> tuple[np.ndarray, np.ndarray]:
    encoder = HashedTfidfSentenceEncoder(dim=512)
    train_texts = [context_text(instance, setting) for instance in train_instances]
    all_texts = [context_text(instance, setting) for instance in all_instances]
    encoder.fit(train_texts)
    return encoder.transform(train_texts).astype(float), encoder.transform(all_texts).astype(float)


def fit_strong_text(
    train_instances: list[PredictionInstance],
    all_instances: list[PredictionInstance],
    clusters: tuple[str, ...],
    setting: str,
) -> tuple[StrongTextConfig, np.ndarray]:
    train_features, all_features = feature_matrix(train_instances, all_instances, setting)
    y_train = target_matrix(train_instances, clusters)
    valid_mask = np.array([instance.split == "valid" for instance in all_instances], dtype=bool)
    valid_instances = [instance for instance in all_instances if instance.split == "valid"]
    best_config: StrongTextConfig | None = None
    best_probs: np.ndarray | None = None
    for ridge_lambda in RIDGE_GRID:
        for temperature in TEMP_GRID:
            model = RidgeSoftTextClassifier(ridge_lambda=ridge_lambda, temperature=temperature)
            model.fit(train_features, y_train)
            probs = model.predict_proba(all_features)
            valid_eval = evaluate_predictions(valid_instances, probs[valid_mask], clusters, setting, "valid", "all", "StrongText")
            if best_config is None or valid_eval.cross_entropy < best_config.validation_cross_entropy:
                best_config = StrongTextConfig(setting, ridge_lambda, temperature, valid_eval.cross_entropy, valid_eval.macro_f1)
                best_probs = probs
    if best_config is None or best_probs is None:
        raise ValueError("Could not fit strong text model")
    return best_config, best_probs


def graph_priors(instances: list[PredictionInstance], graph: dict[str, dict[str, float]], clusters: tuple[str, ...]) -> np.ndarray:
    priors = np.vstack([graph_prior(instance, graph, clusters) for instance in instances])
    errors = np.abs(np.sum(priors, axis=1) - 1.0)
    if float(np.max(errors)) > VALIDATION_TOLERANCE:
        raise ValueError("Graph prior probabilities do not sum to 1")
    return priors


def train_target_prior(instances: list[PredictionInstance], clusters: tuple[str, ...]) -> np.ndarray:
    train = [instance for instance in instances if instance.split == "train"]
    prior = np.mean(target_matrix(train, clusters), axis=0)
    return prior / prior.sum()


def residual_calibration(text_probs: np.ndarray, graph_probs: np.ndarray, target_prior: np.ndarray, beta: float) -> np.ndarray:
    residual = np.log(np.maximum(graph_probs, EPS)) - np.log(np.maximum(target_prior[None, :], EPS))
    logits = np.log(np.maximum(text_probs, EPS)) + beta * residual
    return softmax(logits)


def scan_old_graph_weights(
    instances: list[PredictionInstance],
    clusters: tuple[str, ...],
    text_probs: np.ndarray,
    graph_probs: np.ndarray,
    setting: str,
) -> list[dict[str, Any]]:
    valid_mask = np.array([instance.split == "valid" for instance in instances], dtype=bool)
    valid_instances = [instance for instance in instances if instance.split == "valid"]
    rows: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            if alpha == 0 and beta == 0:
                continue
            probs = graph_poe(text_probs, graph_probs, alpha, beta)
            eval_row = evaluate_predictions(valid_instances, probs[valid_mask], clusters, setting, "valid", "all", "OldTextLateFusionScan")
            rows.append({
                "setting": setting,
                "split": "valid",
                "alpha": alpha,
                "beta": beta,
                "cross_entropy": eval_row.cross_entropy,
                "macro_f1": eval_row.macro_f1,
                "weighted_f1": eval_row.weighted_f1,
                "top3_accuracy": eval_row.top3_accuracy,
            })
    return rows


def choose_calibration(
    instances: list[PredictionInstance],
    clusters: tuple[str, ...],
    setting: str,
    strong_probs: np.ndarray,
    graph_probs: np.ndarray,
    target_prior: np.ndarray,
    method: str,
) -> tuple[CalibrationConfig, np.ndarray]:
    valid_mask = np.array([instance.split == "valid" for instance in instances], dtype=bool)
    valid_instances = [instance for instance in instances if instance.split == "valid"]
    best: CalibrationConfig | None = None
    best_probs: np.ndarray | None = None
    if method == "LateFusion":
        for alpha in ALPHA_GRID:
            for beta in BETA_GRID:
                if alpha == 0 and beta == 0:
                    continue
                probs = graph_poe(strong_probs, graph_probs, alpha, beta)
                eval_row = evaluate_predictions(valid_instances, probs[valid_mask], clusters, setting, "valid", "all", method)
                config = CalibrationConfig(setting, method, alpha, beta, eval_row.cross_entropy, eval_row.macro_f1, eval_row.weighted_f1, eval_row.top3_accuracy)
                if best is None or config.validation_cross_entropy < best.validation_cross_entropy:
                    best, best_probs = config, probs
    elif method == "ResidualCalibration":
        for beta in RESIDUAL_BETA_GRID:
            probs = residual_calibration(strong_probs, graph_probs, target_prior, beta)
            eval_row = evaluate_predictions(valid_instances, probs[valid_mask], clusters, setting, "valid", "all", method)
            config = CalibrationConfig(setting, method, 1.0, beta, eval_row.cross_entropy, eval_row.macro_f1, eval_row.weighted_f1, eval_row.top3_accuracy)
            if best is None or config.validation_cross_entropy < best.validation_cross_entropy:
                best, best_probs = config, probs
    else:
        raise ValueError(method)
    if best is None or best_probs is None:
        raise ValueError(f"Could not choose config for {method}")
    return best, best_probs


def cluster_metrics(instances: list[PredictionInstance], probs: np.ndarray, clusters: tuple[str, ...], model: str, setting: str) -> list[dict[str, Any]]:
    test_mask = np.array([instance.split == "test" for instance in instances], dtype=bool)
    test_instances = [instance for instance in instances if instance.split == "test"]
    test_probs = probs[test_mask]
    y_true = np.array([clusters.index(instance.target_top1_cluster) for instance in test_instances], dtype=int)
    y_pred = np.argmax(test_probs, axis=1)
    y_soft = target_matrix(test_instances, clusters)
    rows: list[dict[str, Any]] = []
    pred_counts = Counter(clusters[int(index)] for index in y_pred)
    for class_index, cluster in enumerate(clusters):
        true_mask = y_true == class_index
        pred_mask = y_pred == class_index
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int(np.sum(true_mask))
        ce = float(-np.mean(np.sum(y_soft[true_mask] * np.log(np.maximum(test_probs[true_mask], EPS)), axis=1))) if support else 0.0
        rows.append({
            "setting": setting,
            "model": model,
            "target_cluster": cluster,
            "support": support,
            "predicted_count": pred_counts.get(cluster, 0),
            "predicted_share": pred_counts.get(cluster, 0) / len(test_instances) if test_instances else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "soft_cross_entropy": ce,
        })
    return rows


def confusion_matrix(instances: list[PredictionInstance], probs: np.ndarray, clusters: tuple[str, ...]) -> np.ndarray:
    test_mask = np.array([instance.split == "test" for instance in instances], dtype=bool)
    test_instances = [instance for instance in instances if instance.split == "test"]
    y_true = np.array([clusters.index(instance.target_top1_cluster) for instance in test_instances], dtype=int)
    y_pred = np.argmax(probs[test_mask], axis=1)
    matrix = np.zeros((len(clusters), len(clusters)), dtype=int)
    for true_index, pred_index in zip(y_true, y_pred):
        matrix[true_index, pred_index] += 1
    return matrix


def draw_matrix(path: Path, title: str, clusters: tuple[str, ...], matrix: np.ndarray) -> None:
    cell = 54
    left, top, right, bottom = 130, 80, 40, 110
    width = left + cell * len(clusters) + right
    height = top + cell * len(clusters) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 25), title, fill=(20, 20, 20))
    max_value = int(np.max(matrix)) if matrix.size else 1
    for col, cluster in enumerate(clusters):
        draw.text((left + col * cell + 8, top - 25), cluster.replace("cluster_", "c"), fill=(40, 40, 40))
    for row, cluster in enumerate(clusters):
        draw.text((20, top + row * cell + 18), cluster, fill=(40, 40, 40))
        for col in range(len(clusters)):
            ratio = matrix[row, col] / max(max_value, 1)
            color = tuple(int(a + (b - a) * ratio) for a, b in zip((245, 247, 250), (34, 102, 173)))
            x0, y0 = left + col * cell, top + row * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color)
            draw.text((x0 + 8, y0 + 19), str(int(matrix[row, col])), fill=(0, 0, 0))
    image.save(path)


def draw_scan_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    alphas = sorted({float(row["alpha"]) for row in rows})
    betas = sorted({float(row["beta"]) for row in rows})
    ce = {(float(row["alpha"]), float(row["beta"])): float(row["cross_entropy"]) for row in rows}
    mf1 = {(float(row["alpha"]), float(row["beta"])): float(row["macro_f1"]) for row in rows}
    cell, left, top = 78, 100, 80
    width, height = left + len(betas) * cell + 50, top + len(alphas) * cell + 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 25), "Dev graph weight scan: CE color, macro-F1 text", fill=(20, 20, 20))
    values = list(ce.values())
    min_ce, max_ce = min(values), max(values)
    for col, beta in enumerate(betas):
        draw.text((left + col * cell + 8, top - 25), f"b={beta:g}", fill=(40, 40, 40))
    for row_index, alpha in enumerate(alphas):
        draw.text((20, top + row_index * cell + 25), f"a={alpha:g}", fill=(40, 40, 40))
        for col, beta in enumerate(betas):
            value = ce[(alpha, beta)]
            ratio = (value - min_ce) / max(max_ce - min_ce, EPS)
            color = tuple(int(a + (b - a) * (1 - ratio)) for a, b in zip((245, 247, 250), (72, 187, 120)))
            x0, y0 = left + col * cell, top + row_index * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color)
            draw.text((x0 + 6, y0 + 18), f"{value:.2f}", fill=(0, 0, 0))
            draw.text((x0 + 6, y0 + 38), f"F1 {mf1[(alpha,beta)]:.2f}", fill=(0, 0, 0))
    image.save(path)


def draw_delta_barplot(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    graph = {row["target_cluster"]: row for row in rows if row["model"] == "GraphPoE" and row["setting"] == "context_only"}
    text = {row["target_cluster"]: row for row in rows if row["model"] == "TextOnly" and row["setting"] == "context_only"}
    clusters = sorted(graph, key=cluster_sort_key)
    deltas = [float(graph[c]["f1"]) - float(text.get(c, {"f1": 0.0})["f1"]) for c in clusters]
    width, height = 1000, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 25), title, fill=(20, 20, 20))
    left, top, bottom, right = 90, 80, 130, 60
    plot_width, plot_height = width - left - right, height - top - bottom
    zero_y = top + plot_height // 2
    max_abs = max(max(abs(v) for v in deltas), 0.01)
    draw.line((left, zero_y, width - right, zero_y), fill=(60, 60, 60), width=2)
    slot = plot_width / max(len(clusters), 1)
    for idx, (cluster, delta) in enumerate(zip(clusters, deltas)):
        center = left + int(slot * (idx + 0.5))
        bar_width = int(slot * 0.55)
        bar_height = int((plot_height / 2 - 20) * abs(delta) / max_abs)
        if delta >= 0:
            y0, y1, color = zero_y - bar_height, zero_y, (72, 187, 120)
        else:
            y0, y1, color = zero_y, zero_y + bar_height, (229, 62, 62)
        draw.rectangle((center - bar_width // 2, y0, center + bar_width // 2, y1), fill=color)
        draw.text((center - 25, y1 + 8 if delta < 0 else y0 - 20), f"{delta:.2f}", fill=(30, 30, 30))
        draw.text((center - 25, height - bottom + 20), cluster.replace("cluster_", "c"), fill=(30, 30, 30))
    image.save(path)


def group_masks(instances: list[PredictionInstance]) -> dict[str, np.ndarray]:
    target_entropy = np.array([instance.target_cluster_entropy for instance in instances], dtype=float)
    source_entropy = np.array([-sum(p * math.log(max(p, EPS)) for p in instance.source_cluster_probs.values()) for instance in instances], dtype=float)
    target_top1 = np.array([instance.target_top1_cluster_prob for instance in instances], dtype=float)
    reliability = np.array([instance.target_reliability_weight if instance.target_reliability_weight is not None else np.nan for instance in instances], dtype=float)
    masks = {
        "target_entropy_high": target_entropy >= np.nanmedian(target_entropy),
        "target_entropy_low": target_entropy < np.nanmedian(target_entropy),
        "source_entropy_high": source_entropy >= np.nanmedian(source_entropy),
        "source_entropy_low": source_entropy < np.nanmedian(source_entropy),
        "target_top1_prob_high": target_top1 >= np.nanmedian(target_top1),
        "target_top1_prob_low": target_top1 < np.nanmedian(target_top1),
    }
    if not np.all(np.isnan(reliability)):
        median_rel = float(np.nanmedian(reliability))
        masks["target_reliability_high"] = np.nan_to_num(reliability, nan=-1) >= median_rel
        masks["target_reliability_low"] = np.nan_to_num(reliability, nan=2) < median_rel
    return masks


def write_scan(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["setting", "split", "alpha", "beta", "cross_entropy", "macro_f1", "weighted_f1", "top3_accuracy"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.12f}" if isinstance(row[key], float) else row[key] for key in fieldnames})


def write_cluster_breakdown(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["setting", "model", "target_cluster", "support", "predicted_count", "predicted_share", "precision", "recall", "f1", "soft_cross_entropy", "f1_delta_vs_textonly", "ce_delta_vs_textonly"]
    text_lookup = {(row["setting"], row["target_cluster"]): row for row in rows if row["model"] == "TextOnly"}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            text_row = text_lookup.get((row["setting"], row["target_cluster"]))
            f1_delta = float(row["f1"]) - float(text_row["f1"]) if text_row else 0.0
            ce_delta = float(row["soft_cross_entropy"]) - float(text_row["soft_cross_entropy"]) if text_row else 0.0
            payload = dict(row)
            payload["f1_delta_vs_textonly"] = f1_delta
            payload["ce_delta_vs_textonly"] = ce_delta
            writer.writerow({key: f"{payload[key]:.12f}" if isinstance(payload[key], float) else payload[key] for key in fieldnames})


def write_uncertainty(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["setting", "group", "model", "sample_count", "accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "cross_entropy", "kl_divergence", "brier_score"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.12f}" if isinstance(row[key], float) else row[key] for key in fieldnames})


def default_output_dir(instance_path: Path | None, root: Path) -> Path:
    if instance_path is not None:
        for parent in instance_path.parents:
            if parent.name == "outputs":
                return parent / "graph_calibration_analysis"
    return root / "src" / "name_free_emoji_clustering" / "outputs" / "graph_calibration_analysis"


def run_analysis(root: Path, output_dir: Path | None = None) -> dict[str, Path]:
    prior = discover_prior_outputs(root)
    instances, clusters, cluster_label_path, text_path, graph_path = build_instances_if_needed(root, prior)
    _, _, a2b_candidate, _ = discover_inputs(root)
    graph = load_graph(a2b_candidate.path, clusters, "A2B")
    train_instances = [instance for instance in instances if instance.split == "train"]
    settings = ["context_only"]
    if any(instance.situation for instance in instances):
        settings.append("situation_aware")

    graph_probs = graph_priors(instances, graph, clusters)
    markov_probs = markov_probabilities(instances, graph, clusters, use_soft_source=False)
    target_prior = train_target_prior(instances, clusters)
    output = output_dir if output_dir is not None else default_output_dir(prior.instance_path, root)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "cluster_wise_error_breakdown": output / "cluster_wise_error_breakdown.csv",
        "graph_weight_scan": output / "graph_weight_scan.csv",
        "stronger_text_baseline_metrics": output / "stronger_text_baseline_metrics.csv",
        "calibration_vs_latefusion_metrics": output / "calibration_vs_latefusion_metrics.csv",
        "uncertainty_group_analysis": output / "uncertainty_group_analysis.csv",
        "graph_calibration_report": output / "graph_calibration_report.md",
        "cluster_confusion_matrix": output / "cluster_confusion_matrix.png",
        "graph_weight_scan_heatmap": output / "graph_weight_scan_heatmap.png",
        "cluster_f1_delta_barplot": output / "cluster_f1_delta_barplot.png",
        "graph_helped_examples": output / "graph_helped_examples.csv",
        "graph_hurt_examples": output / "graph_hurt_examples.csv",
    }

    all_cluster_rows: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    strong_metric_rows = []
    calibration_metric_rows = []
    uncertainty_rows: list[dict[str, Any]] = []
    report_payload: dict[str, Any] = {
        "input_paths": {
            "instances": str(prior.instance_path) if prior.instance_path else "rebuilt",
            "cluster_labels": str(cluster_label_path),
            "text_table": str(text_path),
            "a2b_graph": str(a2b_candidate.path),
            "prior_graph_predictions": str(prior.graph_predictions_path) if prior.graph_predictions_path else "missing; reconstructed",
            "prior_baseline_metrics": str(prior.baseline_metrics_path) if prior.baseline_metrics_path else "missing",
        },
        "settings": {},
    }
    example_records: tuple[list[PredictionRecord], list[PredictionRecord], list[PredictionInstance]] | None = None
    graphpoe_for_confusion: np.ndarray | None = None
    context_cluster_rows: list[dict[str, Any]] = []

    for setting in settings:
        text_tau, text_probs = choose_text_tau(train_instances, instances, clusters, setting, include_source_cluster=False)
        _, text_cluster_probs = choose_text_tau(train_instances, instances, clusters, setting, include_source_cluster=True)
        scan_rows.extend(scan_old_graph_weights(instances, clusters, text_probs, graph_probs, setting))
        strong_config, strong_probs = fit_strong_text(train_instances, instances, clusters, setting)
        late_config, late_probs = choose_calibration(instances, clusters, setting, strong_probs, graph_probs, target_prior, "LateFusion")
        residual_config, residual_probs = choose_calibration(instances, clusters, setting, strong_probs, graph_probs, target_prior, "ResidualCalibration")

        bundles = [
            ModelBundle("TextOnly", text_probs, setting),
            ModelBundle("TextPlusSourceCluster", text_cluster_probs, setting),
            ModelBundle("RoleAwareMarkov", markov_probs, setting),
            ModelBundle("GraphPoE", graph_poe(text_probs, graph_probs, 0.5, 1.0), setting),
            ModelBundle("StrongText", strong_probs, setting),
            ModelBundle("StrongTextLateFusion", late_probs, setting),
            ModelBundle("StrongTextResidualCalibration", residual_probs, setting),
        ]
        for bundle in bundles[:4]:
            rows = cluster_metrics(instances, bundle.probabilities, clusters, bundle.name, setting)
            all_cluster_rows.extend(rows)
            if setting == "context_only":
                context_cluster_rows.extend(rows)
        for bundle in [bundles[4]]:
            strong_metric_rows.extend(evaluate_by_groups(instances, bundle.probabilities, clusters, setting, bundle.name))
        for bundle in bundles[5:]:
            calibration_metric_rows.extend(evaluate_by_groups(instances, bundle.probabilities, clusters, setting, bundle.name))
        masks = group_masks([instance for instance in instances if instance.split == "test"])
        test_instances = [instance for instance in instances if instance.split == "test"]
        test_mask = np.array([instance.split == "test" for instance in instances], dtype=bool)
        uncertainty_models = [
            ("TextOnly", text_probs[test_mask]),
            ("StrongText", strong_probs[test_mask]),
            ("StrongTextResidualCalibration", residual_probs[test_mask]),
        ]
        for group_name, mask in masks.items():
            group_instances = [instance for instance, keep in zip(test_instances, mask.tolist()) if keep]
            if not group_instances:
                continue
            for model_name, probs in uncertainty_models:
                ev = evaluate_predictions(group_instances, probs[mask], clusters, setting, "test", group_name, model_name)
                uncertainty_rows.append({
                    "setting": setting,
                    "group": group_name,
                    "model": model_name,
                    "sample_count": ev.sample_count,
                    "accuracy": ev.accuracy,
                    "macro_f1": ev.macro_f1,
                    "weighted_f1": ev.weighted_f1,
                    "top3_accuracy": ev.top3_accuracy,
                    "cross_entropy": ev.cross_entropy,
                    "kl_divergence": ev.kl_divergence,
                    "brier_score": ev.brier_score,
                })
        report_payload["settings"][setting] = {
            "old_text_tau": text_tau,
            "strong_text_config": strong_config.__dict__,
            "late_fusion_config": late_config.__dict__,
            "residual_config": residual_config.__dict__,
        }
        if setting == "context_only":
            graphpoe_for_confusion = graph_poe(text_probs, graph_probs, 0.5, 1.0)
            example_records = (
                prediction_records(instances, strong_probs, clusters, setting, "StrongText"),
                prediction_records(instances, residual_probs, clusters, setting, "StrongTextResidualCalibration"),
                [instance for instance in instances if instance.split == "test"],
            )

    write_cluster_breakdown(paths["cluster_wise_error_breakdown"], all_cluster_rows)
    write_scan(paths["graph_weight_scan"], scan_rows)
    write_metrics(paths["stronger_text_baseline_metrics"], strong_metric_rows)
    write_metrics(paths["calibration_vs_latefusion_metrics"], calibration_metric_rows)
    write_uncertainty(paths["uncertainty_group_analysis"], uncertainty_rows)
    if graphpoe_for_confusion is not None:
        draw_matrix(paths["cluster_confusion_matrix"], "GraphPoE test confusion matrix", clusters, confusion_matrix(instances, graphpoe_for_confusion, clusters))
    draw_scan_heatmap(paths["graph_weight_scan_heatmap"], [row for row in scan_rows if row["setting"] == "context_only"])
    draw_delta_barplot(paths["cluster_f1_delta_barplot"], context_cluster_rows, "GraphPoE F1 delta vs TextOnly by target cluster")
    if example_records is not None:
        strong_records, residual_records, test_instances = example_records
        strong_by_id = {record.instance_id: record for record in strong_records if record.split == "test"}
        residual_by_id = {record.instance_id: record for record in residual_records if record.split == "test"}
        write_examples(paths["graph_helped_examples"], test_instances, strong_by_id, residual_by_id, helped=True)
        write_examples(paths["graph_hurt_examples"], test_instances, strong_by_id, residual_by_id, helped=False)
    write_report(paths["graph_calibration_report"], report_payload, instances, clusters, all_cluster_rows, scan_rows, strong_metric_rows, calibration_metric_rows, uncertainty_rows, paths)
    return paths


def lookup_metric(rows: list[Any], setting: str, split: str, subset: str, model: str) -> Any | None:
    for row in rows:
        if row.setting == setting and row.split == split and row.subset == subset and row.model == model:
            return row
    return None


def write_report(
    path: Path,
    payload: dict[str, Any],
    instances: list[PredictionInstance],
    clusters: tuple[str, ...],
    cluster_rows: list[dict[str, Any]],
    scan_rows: list[dict[str, Any]],
    strong_rows: list[Any],
    calibration_rows: list[Any],
    uncertainty_rows: list[dict[str, Any]],
    output_paths: dict[str, Path],
) -> None:
    old_text = [row for row in cluster_rows if row["setting"] == "context_only" and row["model"] == "TextOnly"]
    graph = [row for row in cluster_rows if row["setting"] == "context_only" and row["model"] == "GraphPoE"]
    text_by_cluster = {row["target_cluster"]: row for row in old_text}
    graph_deltas = []
    for row in graph:
        base = text_by_cluster.get(row["target_cluster"])
        if base:
            graph_deltas.append((row["target_cluster"], int(row["support"]), float(row["f1"]) - float(base["f1"]), float(row["soft_cross_entropy"]) - float(base["soft_cross_entropy"]), float(row["recall"]) - float(base["recall"])))
    graph_deltas.sort(key=lambda item: item[3])
    best_scan = min([row for row in scan_rows if row["setting"] == "context_only"], key=lambda row: float(row["cross_entropy"]))
    balanced_scan = max([row for row in scan_rows if row["setting"] == "context_only"], key=lambda row: (float(row["macro_f1"]), -float(row["cross_entropy"])))
    strong = lookup_metric(strong_rows, "context_only", "test", "all", "StrongText")
    late = lookup_metric(calibration_rows, "context_only", "test", "all", "StrongTextLateFusion")
    residual = lookup_metric(calibration_rows, "context_only", "test", "all", "StrongTextResidualCalibration")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Graph Calibration Analysis\n\n")
        handle.write("## Inputs\n\n")
        for key, value in payload["input_paths"].items():
            handle.write(f"- {key}: `{value}`\n")
        handle.write("- Emoji names, aliases, Unicode names, text generation, and operators are not used.\n\n")
        handle.write("## Why GraphPoE Soft CE Improved but Macro-F1 Dropped\n\n")
        pred_counts = Counter(row["target_cluster"] for row in graph for _ in range(int(row["predicted_count"])))
        top_predicted = sorted(((row["target_cluster"], int(row["predicted_count"]), float(row["predicted_share"])) for row in graph), key=lambda item: -item[1])[:5]
        handle.write("- GraphPoE predicted cluster concentration (test/context_only): " + ", ".join(f"`{c}`={share:.3f}" for c, _, share in top_predicted) + ".\n")
        hurt = sorted(graph_deltas, key=lambda item: item[2])[:5]
        helped = sorted(graph_deltas, key=lambda item: item[3])[:5]
        handle.write("- Largest F1 drops vs TextOnly: " + ", ".join(f"`{c}` ΔF1={d:.3f} support={s}" for c, s, d, _, _ in hurt) + ".\n")
        handle.write("- Largest soft-CE improvements vs TextOnly: " + ", ".join(f"`{c}` ΔCE={ce:.3f}" for c, _, _, ce, _ in helped) + ".\n")
        handle.write("- Interpretation: GraphPoE calibrated probability mass toward frequent listener destinations, improving soft likelihood and top-3 behavior, but it reduced recall/F1 for several smaller target clusters.\n\n")
        handle.write("## Graph Weight Scan\n\n")
        handle.write(f"- Best dev soft CE config: alpha=`{best_scan['alpha']}`, beta=`{best_scan['beta']}`, CE=`{float(best_scan['cross_entropy']):.6f}`, macro-F1=`{float(best_scan['macro_f1']):.6f}`.\n")
        handle.write(f"- Best dev macro-F1 config: alpha=`{balanced_scan['alpha']}`, beta=`{balanced_scan['beta']}`, CE=`{float(balanced_scan['cross_entropy']):.6f}`, macro-F1=`{float(balanced_scan['macro_f1']):.6f}`.\n")
        handle.write("- If these disagree, the soft-CE optimum is over-relying on the global graph prior for hard macro-F1.\n\n")
        handle.write("## Strong Text and Calibration\n\n")
        if strong and late and residual:
            handle.write("| model | test macro-F1 | test weighted-F1 | test top3 | test soft CE |\n")
            handle.write("|---|---:|---:|---:|---:|\n")
            for row in (strong, late, residual):
                handle.write(f"| `{row.model}` | {row.macro_f1:.6f} | {row.weighted_f1:.6f} | {row.top3_accuracy:.6f} | {row.cross_entropy:.6f} |\n")
            handle.write(f"- Residual calibration vs StrongText: Δmacro-F1=`{residual.macro_f1 - strong.macro_f1:.6f}`, ΔCE=`{residual.cross_entropy - strong.cross_entropy:.6f}`.\n")
        handle.write("\n## Uncertainty Groups\n\n")
        for group in ("target_entropy_high", "target_entropy_low", "target_top1_prob_high", "target_top1_prob_low"):
            rows = [row for row in uncertainty_rows if row["setting"] == "context_only" and row["group"] == group]
            by_model = {row["model"]: row for row in rows}
            if "StrongText" in by_model and "StrongTextResidualCalibration" in by_model:
                st = by_model["StrongText"]
                rc = by_model["StrongTextResidualCalibration"]
                handle.write(f"- `{group}`: residual ΔCE=`{rc['cross_entropy'] - st['cross_entropy']:.6f}`, Δmacro-F1=`{rc['macro_f1'] - st['macro_f1']:.6f}`.\n")
        handle.write("\nInterpretation: beta>0 for GraphPoE indicates that the graph prior can improve soft calibration. However, the graph's global target preference can pull examples toward a few frequent clusters, reducing hard macro-F1 and especially recall for smaller clusters. Residual calibration subtracts the train-set target prior and keeps only the graph's relative residual signal, making it more suitable as a calibration term than as the main model.\n\n")
        handle.write("## Conclusions\n\n")
        handle.write("1. GraphPoE hard macro-F1 drops mainly because predictions concentrate too much on a few frequent listener clusters, suppressing recall/F1 for smaller clusters.\n")
        handle.write("2. The graph prior is better suited as a calibration or residual prior than as the main scoring model.\n")
        handle.write("3. A stronger text model plus graph calibration is worth pursuing, but dev search should constrain macro-F1 rather than optimizing only soft CE.\n")
        handle.write("4. B2A auxiliary calibration and a stronger text backbone should be tested before treating cluster-local continuous vector or operator variants as the main path.\n")
        handle.write("5. Recommendation: prioritize B2A auxiliary calibration plus a stronger text backbone before operator variants.\n\n")
        handle.write("## Outputs\n\n")
        for label, output_path in output_paths.items():
            handle.write(f"- {label}: `{output_path}`\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze graph calibration and lightweight stronger text baselines.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_analysis(args.root.resolve(), args.output_dir.resolve() if args.output_dir else None)
    print(f"Wrote cluster-wise breakdown: {paths['cluster_wise_error_breakdown']}")
    print(f"Wrote graph weight scan: {paths['graph_weight_scan']}")
    print(f"Wrote report: {paths['graph_calibration_report']}")


if __name__ == "__main__":
    main()
