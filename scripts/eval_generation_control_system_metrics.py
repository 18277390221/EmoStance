from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from ablation_utils import (
    accuracy,
    environment_snapshot,
    jsd_per_example,
    latex_escape,
    macro_f1,
    normalize_prob,
    read_jsonl,
    sanitize,
    soft_ce_per_example,
    write_csv,
    write_json,
)


SEEDS = [13, 21, 42]
DEFAULT_BOOTSTRAP_SEED = 20260515
EXPECTED_EXAMPLES_PER_SEED = 512


METHOD_SPECS = [
    {
        "method": "zero control",
        "status": "diagnostic",
        "source_kind": "control",
        "path": "runs/main/generator_control_eval_512_seed{seed}/generations_test.scored.jsonl",
        "group_key": "control_type",
        "group_value": "zero",
    },
    {
        "method": "shuffled control",
        "status": "diagnostic",
        "source_kind": "control",
        "path": "runs/main/generator_control_eval_512_seed{seed}/generations_test.scored.jsonl",
        "group_key": "control_type",
        "group_value": "shuffled",
    },
    {
        "method": "gold control",
        "status": "oracle",
        "source_kind": "control",
        "path": "runs/main/generator_control_eval_512_seed{seed}/generations_test.scored.jsonl",
        "group_key": "control_type",
        "group_value": "gold",
    },
    {
        "method": "predicted control baseline",
        "status": "deployable",
        "source_kind": "control",
        "path": "runs/main/generator_control_eval_512_seed{seed}/generations_test.scored.jsonl",
        "group_key": "control_type",
        "group_value": "predicted",
    },
    {
        "method": "role-aware predicted control",
        "status": "deployable",
        "source_kind": "role_aware_control",
        "path": "runs/main/generator_control_eval_role_weight025_c7mix050_512_seed{seed}/generations_test.scored.jsonl",
        "group_key": "control_type",
        "group_value": "predicted",
    },
    {
        "method": "role-aware predicted control + rerank",
        "status": "deployable",
        "source_kind": "rerank",
        "path": "runs/main/rerank_c7mix050_512_seed{seed}/selected_test.jsonl",
        "group_key": "selection_type",
        "group_value": "rerank_control",
    },
    {
        "method": "oracle gold selection",
        "status": "oracle",
        "source_kind": "rerank",
        "path": "runs/main/rerank_c7mix050_512_seed{seed}/selected_test.jsonl",
        "group_key": "selection_type",
        "group_value": "oracle_gold",
    },
]

METHOD_ORDER = [spec["method"] for spec in METHOD_SPECS]
STATUS_BY_METHOD = {spec["method"]: spec["status"] for spec in METHOD_SPECS}

REFERENCE_VALUES = {
    "zero control": {
        "vs_gold_ce": 1.7816,
        "vs_gold_acc": 0.4837,
        "vs_gold_macro_f1": 0.2584,
        "mean_words": 5.3034,
    },
    "shuffled control": {
        "vs_gold_ce": 2.0570,
        "vs_gold_acc": 0.4368,
        "vs_gold_macro_f1": 0.2332,
        "mean_words": 9.5111,
    },
    "gold control": {
        "vs_gold_ce": 1.2678,
        "vs_gold_acc": 0.6348,
        "vs_gold_macro_f1": 0.4162,
        "mean_words": 9.0742,
    },
    "predicted control baseline": {
        "vs_gold_ce": 1.7576,
        "vs_gold_acc": 0.5182,
        "vs_gold_macro_f1": 0.2920,
        "mean_words": 9.5495,
    },
    "role-aware predicted control": {
        "vs_gold_ce": 1.7367,
        "vs_gold_acc": 0.5169,
        "vs_gold_macro_f1": 0.3040,
        "mean_words": 9.7025,
    },
    "role-aware predicted control + rerank": {
        "vs_gold_ce": 1.5568,
        "vs_gold_acc": 0.5365,
        "vs_gold_macro_f1": 0.3412,
        "mean_words": 10.7240,
    },
    "oracle gold selection": {
        "vs_gold_ce": 1.2583,
        "vs_gold_acc": 0.6556,
        "vs_gold_macro_f1": 0.4416,
        "mean_words": 10.6191,
    },
}

GENERIC_REGEXES = [
    r"\bi'?m sorry\b",
    r"\bi am sorry\b",
    r"\bsorry to hear\b",
    r"\bi understand\b",
    r"\bi can understand\b",
    r"\bthat sounds (hard|tough|difficult|rough)\b",
    r"\bi'?m here for you\b",
    r"\bhope you feel better\b",
    r"\bi hope (you|it|everything)\b",
    r"\bthank you for sharing\b",
    r"\bthanks for sharing\b",
    r"\bthat'?s great\b",
    r"\bthat'?s wonderful\b",
]

GENERIC_EXACT = {
    "i'm sorry to hear that",
    "i understand how you feel",
    "that sounds really hard",
    "i'm here for you",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute generation-control ablation system diagnostics from existing scored generation artifacts."
    )
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument(
        "--ablation-root",
        default="runs/emnlp_ablation_report",
        help="Root directory for EMNLP ablation artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/emnlp_ablation_report/generation_control_system_metrics",
        help="Directory for extended generation-control diagnostics.",
    )
    parser.add_argument(
        "--reuse-existing-scores",
        action="store_true",
        help="Use existing generated-response stance distributions in scored JSONL files.",
    )
    parser.add_argument("--bootstrap", type=int, default=10000, help="Number of paired bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED, help="Random seed for bootstrap.")
    parser.add_argument(
        "--max-self-bleu-examples",
        type=int,
        default=512,
        help="Deterministic cap per method/seed for Self-BLEU computation.",
    )
    return parser.parse_args()


def split_word_count(text: str) -> int:
    return len(str(text or "").split())


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text or "").lower())


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def generic_flag(text: str, word_count: int) -> int:
    norm = normalized_text(text)
    exact = re.sub(r"[.!?]+$", "", norm)
    if exact in GENERIC_EXACT:
        return 1
    if word_count > 20:
        return 0
    return int(any(re.search(pattern, norm) for pattern in GENERIC_REGEXES))


def repetition_flag(text: str) -> int:
    toks = word_tokens(text)
    if len(toks) >= 3:
        if any(a == b == c for a, b, c in zip(toks, toks[1:], toks[2:])):
            return 1
        trigrams = list(zip(toks, toks[1:], toks[2:]))
        if any(count >= 2 for count in Counter(trigrams).values()):
            return 1
    sentences = [
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", s.lower())).strip()
        for s in re.split(r"[.!?]+", str(text or ""))
    ]
    sentences = [s for s in sentences if s]
    return int(any(count >= 2 for count in Counter(sentences).values()))


def ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_distinct_2(texts: Sequence[str]) -> float:
    bigrams: List[Tuple[str, str]] = []
    for text in texts:
        toks = word_tokens(text)
        bigrams.extend(zip(toks, toks[1:]))
    return float(len(set(bigrams)) / len(bigrams)) if bigrams else 0.0


def sentence_bleu4(hypothesis: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    if not hypothesis or not references:
        return 0.0
    hyp_len = len(hypothesis)
    ref_lens = [len(ref) for ref in references if ref]
    if not ref_lens:
        return 0.0
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
    bp = 1.0 if hyp_len > closest_ref_len else math.exp(1.0 - closest_ref_len / max(hyp_len, 1))
    precisions = []
    for n in range(1, 5):
        hyp_counts = ngram_counts(hypothesis, n)
        total = sum(hyp_counts.values())
        if total == 0:
            precisions.append(1.0)
            continue
        max_ref_counts: Counter = Counter()
        for ref in references:
            ref_counts = ngram_counts(ref, n)
            for gram, count in ref_counts.items():
                if count > max_ref_counts[gram]:
                    max_ref_counts[gram] = count
        clipped = sum(min(count, max_ref_counts[gram]) for gram, count in hyp_counts.items())
        precisions.append((clipped + 1.0) / (total + 1.0))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / 4.0))


def corpus_self_bleu(texts: Sequence[str], max_examples: int = 512) -> float:
    tokenized = [word_tokens(text) for text in texts]
    tokenized = tokenized[:max_examples]
    if len(tokenized) <= 1:
        return 0.0
    scores: List[float] = []
    for i, hyp in enumerate(tokenized):
        refs = tokenized[:i] + tokenized[i + 1 :]
        scores.append(sentence_bleu4(hyp, refs))
    return float(np.mean(scores)) if scores else 0.0


def row_key(seed: int, row: Mapping[str, Any]) -> str:
    if row.get("example_index") is not None:
        return str(row["example_index"])
    if row.get("dialogue_id") is not None and row.get("turn_id") is not None:
        return f"{row['dialogue_id']}::{row['turn_id']}"
    return f"row-{seed}"


def get_distribution(row: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return normalize_prob(np.asarray(value, dtype=np.float64))[0]
    except Exception:
        return None


def make_per_example_row(
    raw: Mapping[str, Any],
    method: str,
    status: str,
    seed: int,
    source_path: Path,
    source_group_key: str,
    source_group_value: str,
) -> Dict[str, Any]:
    response = str(raw.get("generated_response", "") or "")
    wc = split_word_count(response)
    gold = get_distribution(raw, "gold_target_cluster")
    pred = get_distribution(raw, "generated_source_cluster_pred")
    intended = get_distribution(raw, "control_target_cluster")
    intended_top1 = raw.get("control_target_top1")
    if method == "oracle gold selection" and gold is not None:
        intended = gold
        intended_top1 = raw.get("gold_target_top1")

    out: Dict[str, Any] = {
        "method": method,
        "status": status,
        "seed": seed,
        "example_id": row_key(seed, raw),
        "aligned_example_key": f"seed{seed}:{row_key(seed, raw)}",
        "dialogue_id": raw.get("dialogue_id"),
        "turn_id": raw.get("turn_id"),
        "source_path": str(source_path),
        "source_group_key": source_group_key,
        "source_group_value": source_group_value,
        "response": response,
        "word_count": wc,
        "too_short_flag": int(wc < 18),
        "too_long_flag": int(wc > 32),
        "length_pass_flag": int(18 <= wc <= 32),
        "generic_flag": generic_flag(response, wc),
        "repetition_flag": repetition_flag(response),
        "gold_target_distribution_available": gold is not None,
        "intended_distribution_available": intended is not None,
        "generated_stance_distribution_available": pred is not None,
        "gold_target_top1": raw.get("gold_target_top1"),
        "generated_source_top1": raw.get("generated_source_top1"),
        "control_target_top1": intended_top1,
    }

    if gold is not None and pred is not None:
        out["vs_gold_ce"] = float(soft_ce_per_example(gold, pred)[0])
        out["vs_gold_jsd"] = float(jsd_per_example(gold, pred)[0])
        out["vs_gold_acc_flag"] = int(int(np.argmax(gold)) == int(np.argmax(pred)))
    else:
        out["vs_gold_ce"] = None
        out["vs_gold_jsd"] = None
        out["vs_gold_acc_flag"] = None

    if intended is not None and pred is not None:
        out["vs_intended_ce"] = float(soft_ce_per_example(intended, pred)[0])
        out["vs_intended_jsd"] = float(jsd_per_example(intended, pred)[0])
        out["vs_intended_acc_flag"] = int(int(np.argmax(intended)) == int(np.argmax(pred)))
    else:
        out["vs_intended_ce"] = None
        out["vs_intended_jsd"] = None
        out["vs_intended_acc_flag"] = None
    return out


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for row in rows:
        val = row.get(key)
        if val is None:
            continue
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            vals.append(fval)
    return vals


def mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def sample_std_or_none(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else None)


def summarize_seed(
    method: str,
    status: str,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    max_self_bleu_examples: int = 512,
) -> Dict[str, Any]:
    responses = [str(row.get("response", "")) for row in rows]
    word_counts = [int(row.get("word_count", 0)) for row in rows]
    gold_rows = [row for row in rows if row.get("vs_gold_ce") is not None]
    intended_rows = [row for row in rows if row.get("vs_intended_ce") is not None]
    summary: Dict[str, Any] = {
        "method": method,
        "status": status,
        "seed": seed,
        "n": len(rows),
        "mean_words": mean_or_none(word_counts),
        "std_words": sample_std_or_none(word_counts),
        "median_words": float(np.median(word_counts)) if word_counts else None,
        "min_words": int(min(word_counts)) if word_counts else None,
        "max_words": int(max(word_counts)) if word_counts else None,
        "too_short_rate": mean_or_none(finite_values(rows, "too_short_flag")),
        "too_long_rate": mean_or_none(finite_values(rows, "too_long_flag")),
        "length_pass_rate": mean_or_none(finite_values(rows, "length_pass_flag")),
        "generic_rate": mean_or_none(finite_values(rows, "generic_flag")),
        "repetition_rate": mean_or_none(finite_values(rows, "repetition_flag")),
        "distinct_2": corpus_distinct_2(responses),
        "self_bleu": corpus_self_bleu(responses, max_self_bleu_examples),
    }
    for key in ["vs_gold_ce", "vs_gold_jsd", "vs_gold_acc_flag"]:
        summary[key.replace("_flag", "")] = mean_or_none(finite_values(rows, key))
    if gold_rows:
        gold_labels = []
        pred_labels = []
        for row in rows:
            if row.get("vs_gold_acc_flag") is None:
                continue
            gold_labels.append(int(row.get("gold_target_top1")) if row.get("gold_target_top1") is not None else None)
            pred_labels.append(int(row.get("generated_source_top1")) if row.get("generated_source_top1") is not None else None)
        if all(v is not None for v in gold_labels + pred_labels):
            y_arr = np.eye(9)[np.asarray(gold_labels, dtype=int)]
            p_arr = np.eye(9)[np.asarray(pred_labels, dtype=int)]
            summary["vs_gold_macro_f1"] = macro_f1(y_arr, p_arr)
        else:
            summary["vs_gold_macro_f1"] = None
    else:
        summary["vs_gold_macro_f1"] = None

    for key in ["vs_intended_ce", "vs_intended_jsd", "vs_intended_acc_flag"]:
        summary[key.replace("_flag", "")] = mean_or_none(finite_values(intended_rows, key))
    if intended_rows:
        intended_labels = []
        pred_labels = []
        for row in intended_rows:
            intended_labels.append(int(row.get("control_target_top1")) if row.get("control_target_top1") is not None else None)
            pred_labels.append(int(row.get("generated_source_top1")) if row.get("generated_source_top1") is not None else None)
        if all(v is not None for v in intended_labels + pred_labels):
            y_arr = np.eye(9)[np.asarray(intended_labels, dtype=int)]
            p_arr = np.eye(9)[np.asarray(pred_labels, dtype=int)]
            summary["vs_intended_macro_f1"] = macro_f1(y_arr, p_arr)
        else:
            summary["vs_intended_macro_f1"] = None
    else:
        summary["vs_intended_macro_f1"] = None
    return summary


def aggregate_mean_std(per_seed: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_seed:
        by_method[str(row["method"])].append(row)
    metrics = [
        "n",
        "vs_gold_ce",
        "vs_gold_jsd",
        "vs_gold_acc",
        "vs_gold_macro_f1",
        "vs_intended_ce",
        "vs_intended_jsd",
        "vs_intended_acc",
        "vs_intended_macro_f1",
        "generic_rate",
        "repetition_rate",
        "mean_words",
        "std_words",
        "median_words",
        "min_words",
        "max_words",
        "too_short_rate",
        "too_long_rate",
        "length_pass_rate",
        "distinct_2",
        "self_bleu",
    ]
    rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        items = by_method.get(method, [])
        out: Dict[str, Any] = {
            "method": method,
            "status": STATUS_BY_METHOD[method],
            "seeds": ",".join(str(item.get("seed")) for item in items),
        }
        for metric in metrics:
            vals = finite_values(items, metric)
            out[f"{metric}_mean"] = mean_or_none(vals)
            out[f"{metric}_std"] = sample_std_or_none(vals)
        rows.append(out)
    return rows


def load_records(
    project_root: Path,
    max_self_bleu_examples: int = 512,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    per_example: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    missing: List[str] = []
    for spec in METHOD_SPECS:
        for seed in SEEDS:
            source_path = project_root / str(spec["path"]).format(seed=seed)
            if not source_path.exists():
                missing.append(f"Missing source for {spec['method']} seed {seed}: {source_path}")
                continue
            raw_rows = read_jsonl(source_path)
            selected = [
                row
                for row in raw_rows
                if row.get(spec["group_key"]) == spec["group_value"] and row.get("split", "test") == "test"
            ]
            if len(selected) != EXPECTED_EXAMPLES_PER_SEED:
                missing.append(
                    f"Unexpected row count for {spec['method']} seed {seed}: "
                    f"{len(selected)} rows in {source_path}"
                )
            example_rows = [
                make_per_example_row(
                    raw=row,
                    method=spec["method"],
                    status=spec["status"],
                    seed=seed,
                    source_path=source_path,
                    source_group_key=spec["group_key"],
                    source_group_value=spec["group_value"],
                )
                for row in selected
            ]
            per_example.extend(example_rows)
            per_seed.append(
                summarize_seed(
                    spec["method"],
                    spec["status"],
                    seed,
                    example_rows,
                    max_self_bleu_examples=max_self_bleu_examples,
                )
            )
    return per_example, per_seed, missing


def paired_bootstrap(
    per_example: Sequence[Mapping[str, Any]],
    system_a: str,
    system_b: str,
    metric: str,
    direction: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in per_example:
        if row.get("method") not in {system_a, system_b}:
            continue
        val = row.get(metric)
        if val is None:
            continue
        try:
            fval = float(val)
        except Exception:
            continue
        if not np.isfinite(fval):
            continue
        grouped[str(row.get("aligned_example_key"))][str(row.get("method"))] = fval
    a_vals: List[float] = []
    b_vals: List[float] = []
    for values in grouped.values():
        if system_a in values and system_b in values:
            a_vals.append(values[system_a])
            b_vals.append(values[system_b])
    if not a_vals:
        return {
            "comparison": f"{system_a} vs {system_b}",
            "metric": metric,
            "direction": direction,
            "system_a": system_a,
            "system_b": system_b,
            "n": 0,
            "mean_a": None,
            "mean_b": None,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "bootstrap_samples": n_boot,
        }
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    delta = a - b
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=np.float64)
    n = len(delta)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(delta[idx].mean())
    p_value = min(1.0, 2.0 * min(float((samples <= 0).mean()), float((samples >= 0).mean())))
    return {
        "comparison": f"{system_a} vs {system_b}",
        "metric": metric,
        "direction": direction,
        "system_a": system_a,
        "system_b": system_b,
        "n": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": float(delta.mean()),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "p_value": p_value,
        "bootstrap_samples": n_boot,
    }


def run_targeted_bootstrap(
    per_example: Sequence[Mapping[str, Any]], n_boot: int, seed: int
) -> List[Dict[str, Any]]:
    comparisons = [
        ("role-aware predicted control + rerank", "role-aware predicted control"),
        ("predicted control baseline", "shuffled control"),
    ]
    metrics = [
        ("vs_gold_ce", "lower"),
        ("vs_gold_jsd", "lower"),
        ("generic_flag", "lower"),
        ("repetition_flag", "lower"),
        ("length_pass_flag", "higher"),
    ]
    rows: List[Dict[str, Any]] = []
    for comp_idx, (a, b) in enumerate(comparisons):
        for metric_idx, (metric, direction) in enumerate(metrics):
            rows.append(
                paired_bootstrap(
                    per_example,
                    a,
                    b,
                    metric,
                    direction,
                    n_boot,
                    seed + 101 * comp_idx + metric_idx,
                )
            )
    return rows


def metric_mean(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get(f"{metric}_mean")
    return None if value is None else float(value)


def metric_std(row: Mapping[str, Any], metric: str) -> float | None:
    value = row.get(f"{metric}_std")
    return None if value is None else float(value)


def fmt_plain(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        fval = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(fval):
        return "NA"
    return f"{fval:.{digits}f}"


def fmt_latex_mean_std(row: Mapping[str, Any], metric: str, bold: bool = False, digits: int = 4) -> str:
    mean = metric_mean(row, metric)
    std = metric_std(row, metric)
    if mean is None:
        return "NA"
    mean_s = fmt_plain(mean, digits)
    if bold:
        mean_s = r"\textbf{" + mean_s + "}"
    if std is None:
        return mean_s
    return mean_s + r"{\scriptsize $\pm$ " + fmt_plain(std, digits) + "}"


def deployable_best(summary_rows: Sequence[Mapping[str, Any]], metric: str, direction: str) -> str | None:
    candidates = [
        (str(row["method"]), metric_mean(row, metric))
        for row in summary_rows
        if row.get("status") == "deployable" and metric_mean(row, metric) is not None
    ]
    if not candidates:
        return None
    if direction == "lower":
        return min(candidates, key=lambda item: float(item[1]))[0]
    return max(candidates, key=lambda item: float(item[1]))[0]


def make_main_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    best = {
        "vs_gold_ce": deployable_best(summary_rows, "vs_gold_ce", "lower"),
        "vs_gold_jsd": deployable_best(summary_rows, "vs_gold_jsd", "lower"),
        "vs_gold_macro_f1": deployable_best(summary_rows, "vs_gold_macro_f1", "higher"),
        "generic_rate": deployable_best(summary_rows, "generic_rate", "lower"),
        "repetition_rate": deployable_best(summary_rows, "repetition_rate", "lower"),
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Status & CE $\downarrow$ & JSD $\downarrow$ & Macro-F1 $\uparrow$ & Generic $\downarrow$ & Repetition $\downarrow$ & Mean words \\",
        r"\midrule",
    ]
    for row in summary_rows:
        method = str(row["method"])
        cells = [
            latex_escape(method),
            latex_escape(row["status"]),
            fmt_latex_mean_std(row, "vs_gold_ce", best["vs_gold_ce"] == method),
            fmt_latex_mean_std(row, "vs_gold_jsd", best["vs_gold_jsd"] == method),
            fmt_latex_mean_std(row, "vs_gold_macro_f1", best["vs_gold_macro_f1"] == method),
            fmt_latex_mean_std(row, "generic_rate", best["generic_rate"] == method),
            fmt_latex_mean_std(row, "repetition_rate", best["repetition_rate"] == method),
            fmt_latex_mean_std(row, "mean_words", False),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{",
        r"Generation-control system diagnostics over three seeds on 512-example test subsets. ",
        r"The best deployable result is bolded. Gold control and oracle gold selection use ",
        r"gold information and are upper-reference conditions, not deployable systems. ",
        r"These automatic scores measure stance consistency and degeneration diagnostics, ",
        r"not human preference. Stance-consistency scores reuse the project stance scorer and ",
        r"therefore should be interpreted with evaluator-model coupling in mind.",
        r"}",
        r"\label{tab:generation-control-system}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def make_appendix_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    best = {
        "vs_gold_acc": deployable_best(summary_rows, "vs_gold_acc", "higher"),
        "distinct_2": deployable_best(summary_rows, "distinct_2", "higher"),
        "self_bleu": deployable_best(summary_rows, "self_bleu", "lower"),
        "too_short_rate": deployable_best(summary_rows, "too_short_rate", "lower"),
        "too_long_rate": deployable_best(summary_rows, "too_long_rate", "lower"),
        "length_pass_rate": deployable_best(summary_rows, "length_pass_rate", "higher"),
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Method & Status & Acc $\uparrow$ & Dist-2 $\uparrow$ & Self-BLEU $\downarrow$ & Std words & Median words & Too short $\downarrow$ & Too long $\downarrow$ & Len pass $\uparrow$ \\",
        r"\midrule",
    ]
    for row in summary_rows:
        method = str(row["method"])
        cells = [
            latex_escape(method),
            latex_escape(row["status"]),
            fmt_latex_mean_std(row, "vs_gold_acc", best["vs_gold_acc"] == method),
            fmt_latex_mean_std(row, "distinct_2", best["distinct_2"] == method),
            fmt_latex_mean_std(row, "self_bleu", best["self_bleu"] == method),
            fmt_latex_mean_std(row, "std_words", False),
            fmt_latex_mean_std(row, "median_words", False),
            fmt_latex_mean_std(row, "too_short_rate", best["too_short_rate"] == method),
            fmt_latex_mean_std(row, "too_long_rate", best["too_long_rate"] == method),
            fmt_latex_mean_std(row, "length_pass_rate", best["length_pass_rate"] == method),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{",
        r"Supplementary generation-control diagnostics. Distinct-2 and Self-BLEU are ",
        r"descriptive diversity metrics. Length pass uses the 18--32 word target range.",
        r"}",
        r"\label{tab:generation-control-system-supp}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def make_intended_table(summary_rows: Sequence[Mapping[str, Any]]) -> str:
    best = {
        "vs_intended_ce": deployable_best(summary_rows, "vs_intended_ce", "lower"),
        "vs_intended_jsd": deployable_best(summary_rows, "vs_intended_jsd", "lower"),
        "vs_intended_acc": deployable_best(summary_rows, "vs_intended_acc", "higher"),
        "vs_intended_macro_f1": deployable_best(summary_rows, "vs_intended_macro_f1", "higher"),
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & Status & Intended CE $\downarrow$ & Intended JSD $\downarrow$ & Intended Acc $\uparrow$ & Intended Macro-F1 $\uparrow$ \\",
        r"\midrule",
    ]
    for row in summary_rows:
        method = str(row["method"])
        cells = [
            latex_escape(method),
            latex_escape(row["status"]),
            fmt_latex_mean_std(row, "vs_intended_ce", best["vs_intended_ce"] == method),
            fmt_latex_mean_std(row, "vs_intended_jsd", best["vs_intended_jsd"] == method),
            fmt_latex_mean_std(row, "vs_intended_acc", best["vs_intended_acc"] == method),
            fmt_latex_mean_std(row, "vs_intended_macro_f1", best["vs_intended_macro_f1"] == method),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{",
        r"Control-realization diagnostics against the intended control distribution. ",
        r"These metrics evaluate whether the generator realizes the supplied control ",
        r"signal, not whether the response matches human preference.",
        r"}",
        r"\label{tab:generation-control-intended-supp}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def reproduction_rows(summary_rows: Sequence[Mapping[str, Any]], tolerance: float = 0.01) -> List[Dict[str, Any]]:
    by_method = {str(row["method"]): row for row in summary_rows}
    rows: List[Dict[str, Any]] = []
    for method, refs in REFERENCE_VALUES.items():
        row = by_method.get(method)
        for metric, ref in refs.items():
            observed = metric_mean(row, metric) if row else None
            diff = None if observed is None else float(observed - ref)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "reference": ref,
                    "observed": observed,
                    "abs_diff": None if diff is None else abs(diff),
                    "within_tolerance": bool(diff is not None and abs(diff) <= tolerance),
                }
            )
    return rows


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) if item is not None else "NA" for item in row) + " |")
    return "\n".join(lines)


def make_readme(
    output_dir: Path,
    ablation_root: Path,
    summary_rows: Sequence[Mapping[str, Any]],
    reproduction: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    missing: Sequence[str],
    command: str,
) -> str:
    source_rows = []
    for spec in METHOD_SPECS:
        source_rows.append(
            [
                spec["method"],
                spec["status"],
                spec["source_kind"],
                spec["path"],
                f"{spec['group_key']}={spec['group_value']}",
            ]
        )
    repro_rows = [
        [
            row["method"],
            row["metric"],
            fmt_plain(row["reference"], 4),
            fmt_plain(row["observed"], 4),
            fmt_plain(row["abs_diff"], 6),
            "yes" if row["within_tolerance"] else "no",
        ]
        for row in reproduction
    ]
    boot_rows = [
        [
            row["comparison"],
            row["metric"],
            row["direction"],
            row["n"],
            fmt_plain(row["delta"], 6),
            f"[{fmt_plain(row['ci_low'], 6)}, {fmt_plain(row['ci_high'], 6)}]",
        ]
        for row in bootstrap_rows
    ]
    summary_preview = [
        [
            row["method"],
            row["status"],
            fmt_plain(row.get("vs_gold_ce_mean"), 4),
            fmt_plain(row.get("vs_gold_jsd_mean"), 4),
            fmt_plain(row.get("generic_rate_mean"), 4),
            fmt_plain(row.get("repetition_rate_mean"), 4),
            fmt_plain(row.get("length_pass_rate_mean"), 4),
        ]
        for row in summary_rows
    ]
    lines = [
        "# Generation-Control System Diagnostics",
        "",
        f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "This directory extends the existing generation-control ablation with automatic system diagnostics.",
        "No model was retrained and no generation job was rerun; the script reused existing scored generation artifacts.",
        "",
        "These metrics are automatic system diagnostics, not human-preference metrics. The stance-consistency",
        "metrics reuse the project stance scorer, so they should be interpreted with evaluator-model coupling in mind.",
        "They should not be read as evidence that a response is more empathetic, more natural, or human-preferred.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Reused Artifacts",
        "",
        md_table(["method", "status", "source kind", "path template", "selector"], source_rows),
        "",
        "The three seeds are 13, 21, and 42. Each method/seed is expected to contain the same 512-example test subset.",
        "Rows are aligned by `(seed, example_index)` for paired bootstrap comparisons.",
        "",
        "## Metric Definitions",
        "",
        "- vs-gold CE/JSD/Acc/Macro-F1 compare the generated-response stance distribution to the gold target stance distribution.",
        "- vs-intended metrics are computed only when an intended control distribution is saved in the artifact.",
        "- Word counts follow the existing project generation scripts: `len(response.split())`, preserving the old Mean words table.",
        "- Length pass uses the 18--32 word target range; too-short is `<18`, too-long is `>32`.",
        "- Distinct-2 is corpus-level unique bigrams divided by total bigrams within each method/seed.",
        "- Self-BLEU is smoothed Self-BLEU-4, capped deterministically at 512 responses per method/seed.",
        "- Generic response rate uses a conservative fallback detector: listed template patterns only count when word count is <=20, plus exact template matches.",
        "- Repetition rate is a response-level flag for repeated normalized trigrams, three consecutive identical tokens, or duplicate normalized sentences.",
        "",
        "## Summary Preview",
        "",
        md_table(
            ["method", "status", "CE", "JSD", "Generic", "Repetition", "Len pass"],
            summary_preview,
        ),
        "",
        "## Reproduction Check",
        "",
        "The script checks whether the old CE / Acc / Macro-F1 / Mean words values are reproduced within tolerance 0.01.",
        "",
        md_table(["method", "metric", "reference", "observed", "abs diff", "ok"], repro_rows),
        "",
        "## Targeted Paired Bootstrap",
        "",
        md_table(["comparison", "metric", "direction", "n", "delta", "95% CI"], boot_rows),
        "",
        "Delta is `system_A - system_B`. For lower-is-better metrics, a negative delta favors system A; for length pass, a positive delta favors system A.",
        "",
        "## Missing Artifacts",
        "",
    ]
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("None.")
    lines += [
        "",
        "## Outputs",
        "",
        f"- `{output_dir / 'per_example_metrics.csv'}`",
        f"- `{output_dir / 'per_seed_summary.csv'}`",
        f"- `{output_dir / 'summary_mean_std.csv'}`",
        f"- `{output_dir / 'bootstrap_targeted_comparisons.csv'}`",
        f"- `{output_dir / 'generation_control_main_table.tex'}`",
        f"- `{output_dir / 'generation_control_appendix_table.tex'}`",
        f"- `{output_dir / 'generation_control_intended_table.tex'}`",
        f"- `{ablation_root / 'ablation_results_generation_control_extended.json'}`",
        f"- `{ablation_root / 'ablation_results_generation_control_extended.csv'}`",
        f"- `{ablation_root / 'ablation_tables_generation_control_extended.tex'}`",
    ]
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    ablation_root = (project_root / args.ablation_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_example, per_seed, missing = load_records(project_root, args.max_self_bleu_examples)
    summary_rows = aggregate_mean_std(per_seed)
    bootstrap_rows = run_targeted_bootstrap(per_example, args.bootstrap, args.seed)
    reproduction = reproduction_rows(summary_rows)

    per_example_fields = [
        "method",
        "status",
        "seed",
        "example_id",
        "aligned_example_key",
        "dialogue_id",
        "turn_id",
        "response",
        "word_count",
        "too_short_flag",
        "too_long_flag",
        "length_pass_flag",
        "generic_flag",
        "repetition_flag",
        "gold_target_distribution_available",
        "intended_distribution_available",
        "generated_stance_distribution_available",
        "vs_gold_ce",
        "vs_gold_jsd",
        "vs_gold_acc_flag",
        "vs_intended_ce",
        "vs_intended_jsd",
        "vs_intended_acc_flag",
        "source_path",
        "source_group_key",
        "source_group_value",
    ]
    write_csv(output_dir / "per_example_metrics.csv", per_example, per_example_fields)
    write_csv(output_dir / "per_seed_summary.csv", per_seed)
    write_csv(output_dir / "summary_mean_std.csv", summary_rows)
    write_csv(output_dir / "bootstrap_targeted_comparisons.csv", bootstrap_rows)

    main_table = make_main_table(summary_rows)
    appendix_table = make_appendix_table(summary_rows)
    intended_table = make_intended_table(summary_rows)
    write_text(output_dir / "generation_control_main_table.tex", main_table + "\n")
    write_text(output_dir / "generation_control_appendix_table.tex", appendix_table + "\n")
    write_text(output_dir / "generation_control_intended_table.tex", intended_table + "\n")

    combined_tables = "\n\n".join([main_table, appendix_table, intended_table]) + "\n"
    write_text(ablation_root / "ablation_tables_generation_control_extended.tex", combined_tables)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(__import__("sys").argv),
        "method_order": METHOD_ORDER,
        "seeds": SEEDS,
        "source_specs": METHOD_SPECS,
        "summary_mean_std": summary_rows,
        "per_seed_summary": per_seed,
        "bootstrap_targeted_comparisons": bootstrap_rows,
        "reproduction_check": reproduction,
        "missing": list(missing),
        "environment": environment_snapshot(),
    }
    write_json(ablation_root / "ablation_results_generation_control_extended.json", sanitize(payload))
    write_csv(ablation_root / "ablation_results_generation_control_extended.csv", summary_rows)

    command = (
        "python scripts/eval_generation_control_system_metrics.py "
        f"--project-root {args.project_root} "
        f"--ablation-root {args.ablation_root} "
        f"--output-dir {args.output_dir} "
        "--reuse-existing-scores "
        f"--bootstrap {args.bootstrap}"
    )
    readme = make_readme(output_dir, ablation_root, summary_rows, reproduction, bootstrap_rows, missing, command)
    write_text(output_dir / "README.md", readme)

    if missing:
        missing_lines = [
            "# Missing Generation-Control Artifacts",
            "",
            "The extended diagnostics were generated with available artifacts, but the following issues were detected:",
            "",
        ]
        missing_lines.extend(f"- {item}" for item in missing)
        missing_lines.append("")
        write_text(output_dir / "MISSING_ARTIFACTS.md", "\n".join(missing_lines))
    else:
        missing_path = output_dir / "MISSING_ARTIFACTS.md"
        if missing_path.exists():
            missing_path.unlink()

    print("Done.")
    print("")
    print("Generation-control system diagnostics completed.")
    print("")
    print("Outputs:")
    print(f"- {output_dir / 'per_example_metrics.csv'}")
    print(f"- {output_dir / 'per_seed_summary.csv'}")
    print(f"- {output_dir / 'summary_mean_std.csv'}")
    print(f"- {output_dir / 'bootstrap_targeted_comparisons.csv'}")
    print(f"- {output_dir / 'generation_control_main_table.tex'}")
    print(f"- {output_dir / 'generation_control_appendix_table.tex'}")
    print(f"- {output_dir / 'generation_control_intended_table.tex'}")
    print(f"- {output_dir / 'README.md'}")
    print(f"- {ablation_root / 'ablation_results_generation_control_extended.json'}")
    print(f"- {ablation_root / 'ablation_results_generation_control_extended.csv'}")
    print(f"- {ablation_root / 'ablation_tables_generation_control_extended.tex'}")


if __name__ == "__main__":
    main()
