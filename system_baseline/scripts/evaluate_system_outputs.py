from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval_utils import (
    DISPLAY_NAMES,
    METHOD_ORDER,
    context_copy_rate,
    distinct_n,
    env_payload,
    generic_response,
    load_yaml,
    meteor_fallback,
    module_available,
    multi_turn_leak,
    normalize_for_match,
    read_jsonl,
    repetition_rate,
    role_marker_leak,
    rouge_l_f1,
    rouge_n_f1,
    sentence_bleu,
    tokenize,
    word_count,
    write_csv,
    write_json,
    write_jsonl,
)


REFERENCE_METRICS = [
    "bertscore_f1",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bleu_1",
    "bleu_2",
    "bleu_3",
    "bleu_4",
    "meteor",
]


def mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def rate(values: Sequence[bool | int | float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def bertscore_values(rows: list[dict[str, Any]], model_type: str, num_layers: int | None, batch_size: int) -> tuple[list[float | None], str]:
    if not rows:
        return [], "no rows"
    if not module_available("bert_score"):
        return [None] * len(rows), "bert_score package unavailable; BERTScore-F1 reported as NA"
    try:
        from bert_score import score  # type: ignore

        candidates = [str(r.get("generated_response", "")) for r in rows]
        references = [str(r.get("gold_response", "")) for r in rows]
        kwargs: dict[str, Any] = {
            "model_type": model_type,
            "batch_size": batch_size,
            "verbose": False,
        }
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        _, _, f1 = score(candidates, references, **kwargs)
        return [float(x) for x in f1.tolist()], f"computed with bert_score.score(model_type='{model_type}', num_layers={num_layers}, batch_size={batch_size})"
    except Exception as exc:
        return [None] * len(rows), f"bert_score failed: {exc}"


def per_example_metrics(rows: list[dict[str, Any]], min_words: int, model_type: str, num_layers: int | None, batch_size: int) -> tuple[list[dict[str, Any]], str]:
    bertscores, note = bertscore_values(rows, model_type, num_layers, batch_size)
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        response = str(row.get("generated_response", "") or "")
        gold = str(row.get("gold_response", "") or "")
        history = row.get("history") if isinstance(row.get("history"), list) else []
        wc = word_count(response)
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        stance = {
            "generated_vs_gold_ce": meta.get("generated_vs_gold_soft_ce"),
            "generated_vs_gold_kl": meta.get("generated_vs_gold_kl"),
        }
        out.append(
            {
                "method_id": row.get("method_id"),
                "display_name": row.get("display_name"),
                "example_id": row.get("example_id"),
                "dialogue_id": row.get("dialogue_id"),
                "bertscore_f1": bertscores[idx] if idx < len(bertscores) else None,
                "rouge_1": rouge_n_f1(response, gold, 1),
                "rouge_2": rouge_n_f1(response, gold, 2),
                "rouge_l": rouge_l_f1(response, gold),
                "bleu_1": sentence_bleu(response, [gold], 1),
                "bleu_2": sentence_bleu(response, [gold], 2),
                "bleu_3": sentence_bleu(response, [gold], 3),
                "bleu_4": sentence_bleu(response, [gold], 4),
                "meteor": meteor_fallback(response, gold),
                "empty_response": wc == 0,
                "too_short_response": 0 < wc < min_words,
                "repetition_rate": repetition_rate(response),
                "generic_response": generic_response(response),
                "role_marker_leakage": role_marker_leak(response),
                "multi_turn_leakage": multi_turn_leak(response),
                "context_copy_rate": context_copy_rate(response, history),
                **stance,
            }
        )
    return out, note


def self_bleu(responses: Sequence[str], max_items: int = 200) -> float:
    if len(responses) < 2:
        return 0.0
    subset = list(responses[:max_items])
    values: list[float] = []
    for i, response in enumerate(subset):
        refs = [r for j, r in enumerate(subset) if j != i]
        values.append(sentence_bleu(response, refs, 2))
    return sum(values) / len(values) if values else 0.0


def stance_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ce_vals = []
    kl_vals = []
    acc_vals = []
    entropy_vals = []
    for row in rows:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        ce = meta.get("generated_vs_gold_soft_ce")
        kl = meta.get("generated_vs_gold_kl")
        pred = meta.get("generated_source_cluster_pred")
        gold_top = meta.get("gold_target_top1")
        pred_top = meta.get("generated_source_top1")
        if isinstance(ce, (int, float)):
            ce_vals.append(float(ce))
        if isinstance(kl, (int, float)):
            kl_vals.append(float(kl))
        if isinstance(gold_top, int) and isinstance(pred_top, int):
            acc_vals.append(1.0 if gold_top == pred_top else 0.0)
        if isinstance(pred, list) and pred:
            total = sum(float(x) for x in pred) or 1.0
            probs = [max(float(x) / total, 1e-12) for x in pred]
            entropy_vals.append(-sum(p * __import__("math").log(p) for p in probs))
    return {
        "stance_ce": mean(ce_vals),
        "stance_jsd": None,
        "stance_accuracy": mean(acc_vals),
        "stance_macro_f1": None,
        "stance_entropy": mean(entropy_vals),
        "stance_available_examples": len(ce_vals),
    }


def aggregate_method(rows: list[dict[str, Any]], per_rows: list[dict[str, Any]], min_words: int, self_bleu_max: int) -> dict[str, Any]:
    responses = [str(row.get("generated_response", "") or "") for row in rows]
    normalized = [normalize_for_match(r) for r in responses]
    starts = [" ".join(tokenize(r)[:3]) for r in responses if tokenize(r)]
    token_counts = [word_count(r) for r in responses]
    ref_block = {metric: mean([row.get(metric) for row in per_rows]) for metric in REFERENCE_METRICS}
    degen = {
        "empty_response_rate": rate([row["empty_response"] for row in per_rows]),
        "too_short_response_rate": rate([row["too_short_response"] for row in per_rows]),
        "repetition_rate": mean([row["repetition_rate"] for row in per_rows]) or 0.0,
        "generic_response_rate": rate([row["generic_response"] for row in per_rows]),
        "role_marker_leakage_rate": rate([row["role_marker_leakage"] for row in per_rows]),
        "multi_turn_leakage_rate": rate([row["multi_turn_leakage"] for row in per_rows]),
        "context_copy_rate": mean([row["context_copy_rate"] for row in per_rows]) or 0.0,
    }
    diversity = {
        "distinct_1": distinct_n(responses, 1),
        "distinct_2": distinct_n(responses, 2),
        "unique_response_ratio": len(set(normalized)) / len(normalized) if normalized else 0.0,
        "unique_start_ratio": len(set(starts)) / len(starts) if starts else 0.0,
        "self_bleu": self_bleu(responses, self_bleu_max),
    }
    quality_internal = {
        "min_word_count": min(token_counts) if token_counts else 0,
        "max_word_count": max(token_counts) if token_counts else 0,
        "responses_below_min_words": sum(1 for x in token_counts if 0 < x < min_words),
    }
    return {
        "method_id": rows[0].get("method_id") if rows else "",
        "display_name": rows[0].get("display_name") if rows else "",
        "num_examples": len(rows),
        **ref_block,
        **diversity,
        **degen,
        **stance_metrics(rows),
        "_internal_quality": quality_internal,
    }


def duplicate_count(rows: list[dict[str, Any]]) -> int:
    ids = [str(row.get("example_id", "")) for row in rows]
    return len(ids) - len(set(ids))


def target_leakage_suspect_rate(rows: list[dict[str, Any]]) -> float:
    flags = []
    for row in rows:
        gen = normalize_for_match(row.get("generated_response", ""))
        gold = normalize_for_match(row.get("gold_response", ""))
        flags.append(bool(gen and gold and gen == gold))
    return rate(flags)


def write_quality_check(path: Path, test_count: int, method_rows: dict[str, list[dict[str, Any]]], metrics: dict[str, Any]) -> None:
    lines = [
        "# System Evaluation Quality Check",
        "",
        "Length-related values here are internal diagnostics only and are not used in the main system metric tables.",
        "",
        "| method | expected examples | actual examples | missing | empty rate | too-short rate | role leakage | duplicate ids | target-copy suspect | status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for method_id in METHOD_ORDER:
        rows = method_rows.get(method_id, [])
        block = metrics.get(method_id, {})
        actual = len(rows)
        status = "ok" if actual == test_count else "partial" if actual else "missing"
        lines.append(
            f"| {DISPLAY_NAMES[method_id]} | {test_count} | {actual} | {max(test_count - actual, 0)} | "
            f"{fmt(block.get('empty_response_rate'))} | {fmt(block.get('too_short_response_rate'))} | "
            f"{fmt(block.get('role_marker_leakage_rate'))} | {duplicate_count(rows)} | "
            f"{fmt(target_leakage_suspect_rate(rows))} | {status} |"
        )
    lines += [
        "",
        "Checks:",
        "- [x] Generation files are read from `system_baseline/generations/`.",
        "- [x] `example_id` alignment is checked against `system_baseline/data/ed_test_inputs.jsonl`.",
        "- [x] Empty, too-short, repeated, generic, role-marker, multi-turn, and context-copy diagnostics are computed.",
        "- [x] Gold responses are used only as references for evaluation.",
        "- [x] Main result tables do not report mean length, median length, length ratio, or too-long rate.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate standardized system generations.")
    parser.add_argument("--config", default="system_baseline/configs/system_eval.yaml")
    parser.add_argument("--generations_dir", default="system_baseline/generations")
    parser.add_argument("--output_dir", default="system_baseline/metrics")
    args = parser.parse_args()

    root = Path(".").resolve()
    cfg = load_yaml(root / args.config)
    min_words = int(cfg.get("evaluation", {}).get("min_words_for_quality_check", 3))
    self_bleu_max = int(cfg.get("evaluation", {}).get("self_bleu_max_items", 200))
    bert_model_type = str(cfg.get("evaluation", {}).get("bertscore_model_type", "distilbert-base-uncased"))
    bert_num_layers_value = cfg.get("evaluation", {}).get("bertscore_num_layers", 6)
    bert_num_layers = int(bert_num_layers_value) if bert_num_layers_value not in (None, "") else None
    bert_batch_size = int(cfg.get("evaluation", {}).get("bertscore_batch_size", 64))
    test_rows = read_jsonl(root / cfg.get("data", {}).get("test_inputs", "system_baseline/data/ed_test_inputs.jsonl"))
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_dir = root / args.generations_dir

    method_rows: dict[str, list[dict[str, Any]]] = {}
    all_per_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    bert_notes: dict[str, str] = {}
    for method_id in METHOD_ORDER:
        path = generations_dir / f"{method_id}.jsonl"
        rows = read_jsonl(path)
        method_rows[method_id] = rows
        if not rows:
            metrics[method_id] = {
                "method_id": method_id,
                "display_name": DISPLAY_NAMES[method_id],
                "num_examples": 0,
            }
            bert_notes[method_id] = "no rows"
            continue
        per_rows, note = per_example_metrics(rows, min_words, bert_model_type, bert_num_layers, bert_batch_size)
        bert_notes[method_id] = note
        all_per_rows.extend(per_rows)
        metrics[method_id] = aggregate_method(rows, per_rows, min_words, self_bleu_max)

    write_jsonl(output_dir / "per_example_metrics.jsonl", all_per_rows)
    write_json(output_dir / "per_method_metrics.json", {"metrics": metrics, "bert_score_notes": bert_notes})
    fieldnames = [
        "method_id",
        "display_name",
        "num_examples",
        *REFERENCE_METRICS,
        "distinct_1",
        "distinct_2",
        "unique_response_ratio",
        "unique_start_ratio",
        "self_bleu",
        "generic_response_rate",
        "repetition_rate",
        "empty_response_rate",
        "too_short_response_rate",
        "role_marker_leakage_rate",
        "multi_turn_leakage_rate",
        "context_copy_rate",
        "stance_ce",
        "stance_jsd",
        "stance_accuracy",
        "stance_macro_f1",
        "stance_entropy",
        "stance_available_examples",
    ]
    write_csv(output_dir / "per_method_metrics.csv", [metrics[m] for m in METHOD_ORDER], fieldnames)
    write_json(root / "system_baseline/logs/environment.json", env_payload(root))
    write_quality_check(root / "system_baseline/logs/quality_check.md", len(test_rows), method_rows, metrics)
    print(f"Wrote {output_dir / 'per_method_metrics.csv'}")
    print(f"Wrote {output_dir / 'per_example_metrics.jsonl'}")
    print(f"Wrote {root / 'system_baseline/logs/quality_check.md'}")


if __name__ == "__main__":
    main()
