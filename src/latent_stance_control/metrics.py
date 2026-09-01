from __future__ import annotations

from typing import Dict

import numpy as np


def _prob_matrix(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(-1, arr.shape[-1])
    return np.nan_to_num(arr, nan=eps, posinf=1.0 / eps, neginf=eps)


def normalize_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = _prob_matrix(x, eps=eps)
    arr = np.clip(arr, eps, None)
    arr /= np.maximum(arr.sum(axis=-1, keepdims=True), eps)
    return arr


def soft_cross_entropy(target: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    target = normalize_prob(target, eps)
    pred = normalize_prob(pred, eps)
    return float(-(target * np.log(pred)).sum(axis=-1).mean())


def kl_divergence(target: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    target = normalize_prob(target, eps)
    pred = normalize_prob(pred, eps)
    return float((target * (np.log(target) - np.log(pred))).sum(axis=-1).mean())


def _label_vector(prob: np.ndarray) -> np.ndarray:
    return np.argmax(normalize_prob(prob), axis=-1).reshape(-1)


def accuracy(target: np.ndarray, pred: np.ndarray) -> float:
    y_true = _label_vector(target)
    y_pred = _label_vector(pred)
    n = min(y_true.shape[0], y_pred.shape[0])
    if n == 0:
        return 0.0
    return float((y_true[:n] == y_pred[:n]).mean())


def macro_f1(target: np.ndarray, pred: np.ndarray) -> float:
    y_true = _label_vector(target)
    y_pred = _label_vector(pred)
    n = min(y_true.shape[0], y_pred.shape[0])
    if n == 0:
        return 0.0
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1s = []
    for label in labels:
        tp = float(((y_true == label) & (y_pred == label)).sum())
        fp = float(((y_true != label) & (y_pred == label)).sum())
        fn = float(((y_true == label) & (y_pred != label)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1s.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s)) if f1s else 0.0


def expected_calibration_error(target: np.ndarray, pred: np.ndarray, bins: int = 10) -> float:
    target = normalize_prob(target)
    pred = normalize_prob(pred)
    n = min(target.shape[0], pred.shape[0])
    if n == 0:
        return 0.0
    target = target[:n]
    pred = pred[:n]
    y_true = np.argmax(target, axis=-1).reshape(-1)
    y_pred = np.argmax(pred, axis=-1).reshape(-1)
    conf = np.max(pred, axis=-1).reshape(-1)
    ece = 0.0
    for i in range(bins):
        low = i / bins
        high = (i + 1) / bins
        mask = (conf >= low) & (conf < high if i < bins - 1 else conf <= high)
        if mask.any():
            ece += float(mask.mean() * abs((y_true[mask] == y_pred[mask]).mean() - conf[mask].mean()))
    return ece


def cosine_similarity(target: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    target = np.nan_to_num(np.asarray(target, dtype=np.float64), nan=0.0)
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0)
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    n = min(target.shape[0], pred.shape[0])
    if n == 0:
        return 0.0
    target = target[:n]
    pred = pred[:n]
    numerator = (target * pred).sum(axis=-1)
    denom = np.linalg.norm(target, axis=-1) * np.linalg.norm(pred, axis=-1)
    return float((numerator / np.maximum(denom, eps)).mean())


def evaluate_cluster_predictions(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "soft_ce": soft_cross_entropy(target, pred),
        "kl": kl_divergence(target, pred),
        "accuracy": accuracy(target, pred),
        "macro_f1": macro_f1(target, pred),
        "ece": expected_calibration_error(target, pred),
    }


def evaluate_vectors(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    target = np.nan_to_num(np.asarray(target), nan=0.0)
    pred = np.nan_to_num(np.asarray(pred), nan=0.0)
    return {
        "vector_mse": float(((target - pred) ** 2).mean()),
        "vector_cosine": cosine_similarity(target, pred),
    }
