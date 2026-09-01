from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN = ROOT / "latent_stance_control" / "runs" / "main"
OUT_DIR_DEFAULT = ROOT / "runs" / "emnlp_ablation_report"

DATASET_METADATA = {
    "dataset": "EMOJIDIALOGUE adjacent-turn test set",
    "train_examples": 58829,
    "dev_examples": 9263,
    "test_examples": 8397,
    "observed_emoji": 124,
    "stance_clusters": 9,
    "stance_vector_dim": 256,
}


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def read_jsonl(path: str | Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def read_text(path: str | Path, limit_chars: int = 200_000) -> str:
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        return f.read(limit_chars)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return sanitize(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def normalize_prob(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.nan_to_num(arr, nan=eps, posinf=1.0, neginf=eps)
    arr = np.clip(arr, eps, None)
    arr = arr / np.maximum(arr.sum(axis=-1, keepdims=True), eps)
    return arr


def soft_ce(gold: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    gold = normalize_prob(gold, eps)
    pred = normalize_prob(pred, eps)
    n = min(len(gold), len(pred))
    return float(-(gold[:n] * np.log(pred[:n])).sum(axis=-1).mean())


def soft_ce_per_example(gold: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    gold = normalize_prob(gold, eps)
    pred = normalize_prob(pred, eps)
    n = min(len(gold), len(pred))
    return -(gold[:n] * np.log(pred[:n])).sum(axis=-1)


def kl_div(gold: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    gold = normalize_prob(gold, eps)
    pred = normalize_prob(pred, eps)
    n = min(len(gold), len(pred))
    return float((gold[:n] * (np.log(gold[:n]) - np.log(pred[:n]))).sum(axis=-1).mean())


def jsd(gold: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    gold = normalize_prob(gold, eps)
    pred = normalize_prob(pred, eps)
    n = min(len(gold), len(pred))
    mid = normalize_prob(0.5 * (gold[:n] + pred[:n]), eps)
    left = (gold[:n] * (np.log(gold[:n]) - np.log(mid))).sum(axis=-1)
    right = (pred[:n] * (np.log(pred[:n]) - np.log(mid))).sum(axis=-1)
    return float(0.5 * (left + right).mean())


def jsd_per_example(gold: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    gold = normalize_prob(gold, eps)
    pred = normalize_prob(pred, eps)
    n = min(len(gold), len(pred))
    mid = normalize_prob(0.5 * (gold[:n] + pred[:n]), eps)
    left = (gold[:n] * (np.log(gold[:n]) - np.log(mid))).sum(axis=-1)
    right = (pred[:n] * (np.log(pred[:n]) - np.log(mid))).sum(axis=-1)
    return 0.5 * (left + right)


def brier(gold: np.ndarray, pred: np.ndarray) -> float:
    gold = normalize_prob(gold)
    pred = normalize_prob(pred)
    n = min(len(gold), len(pred))
    return float(((gold[:n] - pred[:n]) ** 2).sum(axis=-1).mean())


def labels(prob: np.ndarray) -> np.ndarray:
    return np.argmax(normalize_prob(prob), axis=-1)


def accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    y = labels(gold)
    p = labels(pred)
    n = min(len(y), len(p))
    return float((y[:n] == p[:n]).mean()) if n else 0.0


def macro_f1(gold: np.ndarray, pred: np.ndarray) -> float:
    y = labels(gold)
    p = labels(pred)
    n = min(len(y), len(p))
    if n == 0:
        return 0.0
    y = y[:n]
    p = p[:n]
    labs = sorted(set(y.tolist()) | set(p.tolist()))
    values = []
    for lab in labs:
        tp = float(((y == lab) & (p == lab)).sum())
        fp = float(((y != lab) & (p == lab)).sum())
        fn = float(((y == lab) & (p != lab)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        values.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(values)) if values else 0.0


def per_cluster_f1(gold: np.ndarray, pred: np.ndarray, num_classes: int | None = None) -> List[float]:
    y = labels(gold)
    p = labels(pred)
    n = min(len(y), len(p))
    if n == 0:
        return []
    y = y[:n]
    p = p[:n]
    if num_classes is None:
        num_classes = int(max(y.max(initial=0), p.max(initial=0)) + 1)
    values: List[float] = []
    for lab in range(num_classes):
        tp = float(((y == lab) & (p == lab)).sum())
        fp = float(((y != lab) & (p == lab)).sum())
        fn = float(((y == lab) & (p != lab)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        values.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return values


def entropy_confidence_breakdown(gold: np.ndarray, pred: np.ndarray) -> List[Dict[str, Any]]:
    gold = normalize_prob(gold)
    pred = normalize_prob(pred)
    n = min(len(gold), len(pred))
    if n == 0:
        return []
    gold = gold[:n]
    pred = pred[:n]
    entropy = -(gold * np.log(gold)).sum(axis=-1)
    threshold = float(np.median(entropy))
    y = labels(gold)
    p = labels(pred)
    conf = pred.max(axis=-1)
    ce = soft_ce_per_example(gold, pred)
    rows: List[Dict[str, Any]] = []
    for name, mask in [("low_target_entropy", entropy <= threshold), ("high_target_entropy", entropy > threshold)]:
        if mask.any():
            rows.append({
                "stratum": name,
                "n": int(mask.sum()),
                "target_entropy_mean": float(entropy[mask].mean()),
                "confidence_mean": float(conf[mask].mean()),
                "soft_ce": float(ce[mask].mean()),
                "accuracy": float((y[mask] == p[mask]).mean()),
            })
        else:
            rows.append({"stratum": name, "n": 0})
    return rows


def ece(gold: np.ndarray, pred: np.ndarray, bins: int = 10) -> float:
    gold = normalize_prob(gold)
    pred = normalize_prob(pred)
    n = min(len(gold), len(pred))
    if n == 0:
        return 0.0
    y = np.argmax(gold[:n], axis=-1)
    p = np.argmax(pred[:n], axis=-1)
    conf = np.max(pred[:n], axis=-1)
    total = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if mask.any():
            total += float(mask.mean() * abs((y[mask] == p[mask]).mean() - conf[mask].mean()))
    return total


def cluster_metrics(gold: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    return {
        "soft_ce": soft_ce(gold, pred),
        "jsd": jsd(gold, pred),
        "kl": kl_div(gold, pred),
        "accuracy": accuracy(gold, pred),
        "macro_f1": macro_f1(gold, pred),
        "ece": ece(gold, pred),
        "brier": brier(gold, pred),
        "per_cluster_f1": per_cluster_f1(gold, pred, DATASET_METADATA["stance_clusters"]),
        "entropy_confidence_breakdown": entropy_confidence_breakdown(gold, pred),
    }


def vector_cosine(gold: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    gold = np.nan_to_num(np.asarray(gold, dtype=np.float64), nan=0.0)
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0)
    if gold.ndim == 1:
        gold = gold.reshape(1, -1)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    n = min(len(gold), len(pred))
    num = (gold[:n] * pred[:n]).sum(axis=-1)
    den = np.linalg.norm(gold[:n], axis=-1) * np.linalg.norm(pred[:n], axis=-1)
    return float((num / np.maximum(den, eps)).mean()) if n else 0.0


def vector_cosine_per_example(gold: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    gold = np.nan_to_num(np.asarray(gold, dtype=np.float64), nan=0.0)
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0)
    n = min(len(gold), len(pred))
    num = (gold[:n] * pred[:n]).sum(axis=-1)
    den = np.linalg.norm(gold[:n], axis=-1) * np.linalg.norm(pred[:n], axis=-1)
    return num / np.maximum(den, eps)


def vector_mse(gold: np.ndarray, pred: np.ndarray) -> float:
    gold = np.nan_to_num(np.asarray(gold, dtype=np.float64), nan=0.0)
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0)
    n = min(len(gold), len(pred))
    return float(((gold[:n] - pred[:n]) ** 2).mean()) if n else 0.0


def npz_arrays(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def load_prepared_rows(prepared_dir: str | Path, split: str) -> List[Dict[str, Any]]:
    path = Path(prepared_dir) / f"{split}.jsonl"
    if not path.exists():
        return []
    return read_jsonl(path)


def graph_prediction(source_prob: np.ndarray, transitions: Sequence[str], matrices: Mapping[str, Any]) -> np.ndarray:
    source_prob = normalize_prob(source_prob)
    rows = []
    for i, transition in enumerate(transitions):
        matrix = np.asarray(matrices.get(transition) or matrices.get("A->B"), dtype=np.float64)
        gp = source_prob[i] @ matrix
        rows.append(gp / max(float(np.sum(gp)), 1e-12))
    return normalize_prob(np.asarray(rows))


def fuse_log_probs(text_prob: np.ndarray, graph_prob: np.ndarray, beta: float) -> np.ndarray:
    text_prob = normalize_prob(text_prob)
    graph_prob = normalize_prob(graph_prob)
    logits = np.log(text_prob) + beta * np.log(graph_prob)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return normalize_prob(exp)


def gated_fusion(text_prob: np.ndarray, graph_prob: np.ndarray, beta: float) -> np.ndarray:
    text_prob = normalize_prob(text_prob)
    entropy = -(text_prob * np.log(text_prob)).sum(axis=-1)
    gate = np.clip(entropy / np.log(text_prob.shape[-1]), 0.0, 1.0)[:, None]
    return fuse_log_probs(text_prob, graph_prob, beta * gate)


def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


GENERIC_PATTERNS = [
    "i understand",
    "i'm sorry",
    "im sorry",
    "i am sorry",
    "that sucks",
    "i know how you feel",
    "yeah",
    "okay",
]


def text_diagnostics(texts: Sequence[str]) -> Dict[str, float]:
    tokenized = [words(t) for t in texts]
    n = len(tokenized)
    lengths = np.asarray([len(toks) for toks in tokenized], dtype=np.float64)
    unigrams: List[str] = []
    bigrams: List[Tuple[str, str]] = []
    repeated = 0
    generic = 0
    too_short = 0
    for text, toks in zip(texts, tokenized):
        unigrams.extend(toks)
        bigrams.extend(list(zip(toks, toks[1:])))
        too_short += int(len(toks) < 3)
        lowered = (text or "").strip().lower()
        generic += int(any(lowered.startswith(pat) for pat in GENERIC_PATTERNS))
        counts = Counter(toks)
        repeated += int(any(v >= 3 for v in counts.values()) or any(a == b for a, b in zip(toks, toks[1:])))
    return {
        "mean_length": float(lengths.mean()) if n else None,
        "distinct_1": float(len(set(unigrams)) / max(len(unigrams), 1)),
        "distinct_2": float(len(set(bigrams)) / max(len(bigrams), 1)),
        "generic_rate": float(generic / max(n, 1)),
        "repetition_rate": float(repeated / max(n, 1)),
        "too_short_rate": float(too_short / max(n, 1)),
    }


def generation_rows_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    gold = np.asarray([r["gold_target_cluster"] for r in rows], dtype=np.float64)
    pred = np.asarray([r["generated_source_cluster_pred"] for r in rows], dtype=np.float64)
    metrics = {
        "vs_gold_ce": soft_ce(gold, pred),
        "vs_gold_jsd": jsd(gold, pred),
        "vs_gold_kl": kl_div(gold, pred),
        "vs_gold_acc": accuracy(gold, pred),
        "vs_gold_macro_f1": macro_f1(gold, pred),
        "vs_gold_ece": ece(gold, pred),
    }
    control_rows = [r for r in rows if r.get("control_target_cluster") is not None]
    if control_rows:
        control = np.asarray([r["control_target_cluster"] for r in control_rows], dtype=np.float64)
        control_pred = np.asarray([r["generated_source_cluster_pred"] for r in control_rows], dtype=np.float64)
        metrics.update(
            {
                "vs_control_ce": soft_ce(control, control_pred),
                "vs_control_jsd": jsd(control, control_pred),
                "vs_control_acc": accuracy(control, control_pred),
                "vs_control_macro_f1": macro_f1(control, control_pred),
            }
        )
    metrics.update(text_diagnostics([str(r.get("generated_response", "")) for r in rows]))
    return metrics


def mean_std(values: Sequence[float | None]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=0)),
        "values": vals,
    }


def bootstrap_ci(
    a_values: Sequence[float],
    b_values: Sequence[float] | None = None,
    n_boot: int = 2000,
    seed: int = 2026,
) -> Dict[str, Any]:
    a = np.asarray(a_values, dtype=np.float64)
    if b_values is None:
        delta = a
    else:
        b = np.asarray(b_values, dtype=np.float64)
        n = min(len(a), len(b))
        delta = a[:n] - b[:n]
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return {"mean_delta": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    n = len(delta)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(delta[idx].mean())
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n": int(n),
        "bootstrap_samples": int(n_boot),
    }


def load_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def environment_snapshot() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": load_git_commit(),
        "cwd": str(ROOT),
    }
    try:
        env["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import torch  # type: ignore

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env["torch_error"] = str(exc)
    return env


def append_command(out_dir: str | Path, command: str, note: str | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "commands.log").open("a", encoding="utf-8") as f:
        stamp = datetime.now().isoformat(timespec="seconds")
        f.write(f"[{stamp}] {command}\n")
        if note:
            f.write(f"  # {note}\n")


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def fmt_mean_std(metric: Mapping[str, Any] | None, digits: int = 4) -> str:
    if not metric:
        return "NA"
    mean = metric.get("mean")
    std = metric.get("std")
    if mean is None:
        return "NA"
    if std is None:
        return fmt_num(mean, digits)
    return f"{fmt_num(mean, digits)}±{fmt_num(std, digits)}"


def extract_seed(path: str | Path, obj: Mapping[str, Any] | None = None) -> int | None:
    if obj and isinstance(obj.get("seed"), int):
        return int(obj["seed"])
    match = re.search(r"seed[_-]?(\d+)", str(path))
    return int(match.group(1)) if match else None


def summarize_metric_object(obj: Any, max_keys: int = 80) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if len(flat) >= max_keys:
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (int, float, str)) or value is None:
            flat[prefix] = sanitize(value)

    visit("", obj)
    return flat


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) if x is not None else "NA" for x in row) + " |")
    return "\n".join(lines)


def latex_escape(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)
