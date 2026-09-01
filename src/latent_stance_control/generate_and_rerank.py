from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from .data import load_prepared_split, write_json, write_jsonl
from .generate_and_score_controls import (
    aligned_predicted_rows,
    generate_one,
    generation_prompt_text,
    load_generator,
    load_stance_scorer,
    mean_text_stats,
    parse_splits,
    prompt_text,
    score_records,
    select_rows,
    unload_generator,
)
from .metrics import evaluate_cluster_predictions, kl_divergence, soft_cross_entropy


SELECTION_TYPES = ("raw_first", "rerank_control", "oracle_gold")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_seed(seed: int, split: str, split_index: int) -> int:
    return seed + split_index * 1009 + sum(ord(ch) for ch in split)


def as_float_list(value: Any) -> List[float]:
    return np.asarray(value, dtype=np.float32).astype(float).tolist()


def top1(prob: Any) -> int:
    return int(np.argmax(np.asarray(prob, dtype=np.float64)))


def response_word_count(record: dict) -> int:
    return len(str(record.get("generated_response", "")).split())


def generate_candidates_for_split(
    lm,
    tokenizer,
    projector,
    split: str,
    gold_rows: List[dict],
    predicted_rows: List[dict],
    args,
    prefix_tokens: int,
) -> List[dict]:
    records: List[dict] = []
    total = len(gold_rows) * args.num_candidates
    done = 0
    for example_index, (gold_row, pred_row) in enumerate(zip(gold_rows, predicted_rows)):
        if "target_vector" not in pred_row:
            raise KeyError(f"{split} example {example_index} is missing predicted target_vector")
        if "target_cluster" not in pred_row:
            raise KeyError(f"{split} example {example_index} is missing predicted target_cluster")

        control_vector = np.asarray(pred_row["target_vector"], dtype=np.float32)
        control_cluster = as_float_list(pred_row["target_cluster"])
        gold_cluster = as_float_list(gold_row["target_cluster"])

        for candidate_index in range(args.num_candidates):
            generated = generate_one(lm, tokenizer, projector, gold_row, control_vector, args, prefix_tokens)
            records.append(
                {
                    "split": split,
                    "example_index": example_index,
                    "candidate_index": candidate_index,
                    "dialogue_id": gold_row.get("dialogue_id"),
                    "turn_id": gold_row.get("turn_id"),
                    "role": gold_row.get("role"),
                    "next_role": gold_row.get("next_role"),
                    "transition": gold_row.get("transition"),
                    "situation": gold_row.get("situation", ""),
                    "context": gold_row.get("context", ""),
                    "prompt": generation_prompt_text(gold_row, args),
                    "reference_response": gold_row.get("response", ""),
                    "generated_response": generated,
                    "gold_target_cluster": gold_cluster,
                    "gold_target_top1": top1(gold_cluster),
                    "control_target_cluster": control_cluster,
                    "control_target_top1": top1(control_cluster),
                    "control_source": "predicted",
                    "control_source_dialogue_id": pred_row.get("dialogue_id"),
                    "control_source_turn_id": pred_row.get("turn_id"),
                }
            )
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"generated {split}: {done}/{total}", flush=True)
    return records


def attach_candidate_scores(records: List[dict], pred_cluster: np.ndarray, args) -> None:
    for record, q in zip(records, pred_cluster):
        q = np.asarray(q, dtype=np.float64)
        gold = np.asarray(record["gold_target_cluster"], dtype=np.float64)
        control = np.asarray(record["control_target_cluster"], dtype=np.float64)
        control_ce = soft_cross_entropy(control, q)
        gold_ce = soft_cross_entropy(gold, q)

        record["generated_source_cluster_pred"] = q.astype(float).tolist()
        record["generated_source_top1"] = int(np.argmax(q))
        record["generated_words"] = response_word_count(record)
        record["generated_vs_gold_soft_ce"] = gold_ce
        record["generated_vs_gold_kl"] = kl_divergence(gold, q)
        record["generated_vs_control_soft_ce"] = control_ce
        record["generated_vs_control_kl"] = kl_divergence(control, q)
        record["rerank_control_score"] = -control_ce
        record["oracle_gold_score"] = -gold_ce


def group_by_example(records: Iterable[dict]) -> Dict[int, List[dict]]:
    groups: Dict[int, List[dict]] = defaultdict(list)
    for record in records:
        groups[int(record["example_index"])].append(record)
    for candidates in groups.values():
        candidates.sort(key=lambda item: int(item["candidate_index"]))
    return dict(groups)


def pick_max(candidates: List[dict], score_key: str) -> dict:
    return max(candidates, key=lambda item: (float(item[score_key]), -int(item["candidate_index"])))


def select_reranked(records: List[dict]) -> List[dict]:
    selected: List[dict] = []
    for example_index in sorted(group_by_example(records)):
        candidates = group_by_example(records)[example_index]
        picks = {
            "raw_first": candidates[0],
            "rerank_control": pick_max(candidates, "rerank_control_score"),
            "oracle_gold": pick_max(candidates, "oracle_gold_score"),
        }
        for selection_type, candidate in picks.items():
            row = dict(candidate)
            row["selection_type"] = selection_type
            row["selected_candidate_index"] = int(candidate["candidate_index"])
            row["candidate_count"] = len(candidates)
            selected.append(row)
    return selected


def safe_eval(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if len(target) == 0 or len(pred) == 0:
        return {"soft_ce": 0.0, "kl": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "ece": 0.0}
    return evaluate_cluster_predictions(target, pred)


def metric_block(records: List[dict]) -> Dict[str, Any]:
    if not records:
        return {
            "vs_gold": safe_eval(np.zeros((0, 0)), np.zeros((0, 0))),
            "vs_control": safe_eval(np.zeros((0, 0)), np.zeros((0, 0))),
            "mean_generated_words": 0.0,
            "empty_rate": 0.0,
            "mean_selected_candidate_index": 0.0,
            "generated_top1_distribution": {},
        }

    pred = np.asarray([record["generated_source_cluster_pred"] for record in records], dtype=np.float64)
    gold = np.asarray([record["gold_target_cluster"] for record in records], dtype=np.float64)
    control = np.asarray([record["control_target_cluster"] for record in records], dtype=np.float64)
    selected_indices = [int(record.get("selected_candidate_index", record.get("candidate_index", 0))) for record in records]
    top1_values, top1_counts = np.unique(np.argmax(pred, axis=-1), return_counts=True)
    out: Dict[str, Any] = {
        "vs_gold": safe_eval(gold, pred),
        "vs_control": safe_eval(control, pred),
        "mean_selected_candidate_index": float(np.mean(selected_indices)) if selected_indices else 0.0,
        "generated_top1_distribution": {str(k): int(v) for k, v in zip(top1_values, top1_counts)},
    }
    out.update(mean_text_stats(records))
    return out


def mean_score(records: List[dict], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records])) if records else 0.0


def candidate_pool_metrics(records: List[dict]) -> Dict[str, float]:
    groups = group_by_example(records)
    raw_rows: List[dict] = []
    best_control_rows: List[dict] = []
    best_gold_rows: List[dict] = []
    candidate_counts: List[int] = []
    for candidates in groups.values():
        raw_rows.append(candidates[0])
        best_control_rows.append(min(candidates, key=lambda item: (float(item["generated_vs_control_soft_ce"]), int(item["candidate_index"]))))
        best_gold_rows.append(min(candidates, key=lambda item: (float(item["generated_vs_gold_soft_ce"]), int(item["candidate_index"]))))
        candidate_counts.append(len(candidates))

    raw_control_ce = mean_score(raw_rows, "generated_vs_control_soft_ce")
    best_control_ce = mean_score(best_control_rows, "generated_vs_control_soft_ce")
    raw_gold_ce = mean_score(raw_rows, "generated_vs_gold_soft_ce")
    best_gold_ce = mean_score(best_gold_rows, "generated_vs_gold_soft_ce")
    return {
        "examples": float(len(groups)),
        "mean_candidates_per_example": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "raw_control_ce": raw_control_ce,
        "best_control_ce": best_control_ce,
        "control_ce_oracle_gain": raw_control_ce - best_control_ce,
        "raw_gold_ce": raw_gold_ce,
        "best_gold_ce": best_gold_ce,
        "gold_ce_oracle_gain": raw_gold_ce - best_gold_ce,
    }


def aggregate_metrics(candidates_by_split: Dict[str, List[dict]], selected_by_split: Dict[str, List[dict]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for split, candidates in candidates_by_split.items():
        split_selected = selected_by_split[split]
        metrics[split] = {
            "candidate_pool": candidate_pool_metrics(candidates),
            "selections": {},
        }
        for selection_type in SELECTION_TYPES:
            subset = [record for record in split_selected if record["selection_type"] == selection_type]
            metrics[split]["selections"][selection_type] = metric_block(subset)
    return metrics


def write_markdown_report(path: Path, metrics: Dict[str, Any], args) -> None:
    lines = [
        "# Generate And Rerank",
        "",
        f"Generator: `{args.generator_dir}`",
        f"Stance scorer: `{args.stance_dir}`",
        f"Predicted control prepared: `{args.predicted_prepared}`",
        f"Candidates per example: `{args.num_candidates}`",
        f"Max examples per split: `{args.max_examples}`",
        f"Sampling: `{'false' if args.no_sample else 'true'}`",
        "",
    ]
    for split, split_metrics in metrics.items():
        pool = split_metrics["candidate_pool"]
        lines += [
            f"## {split}",
            "",
            (
                "Candidate pool: "
                f"examples={int(pool.get('examples', 0))}, "
                f"mean candidates={pool.get('mean_candidates_per_example', 0):.2f}, "
                f"control CE oracle gain={pool.get('control_ce_oracle_gain', 0):.4f}, "
                f"gold CE oracle gain={pool.get('gold_ce_oracle_gain', 0):.4f}"
            ),
            "",
            "| selection | vs_gold CE | vs_gold acc | vs_gold macro-F1 | vs_control CE | vs_control acc | selected idx | mean words | empty rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for selection_type in SELECTION_TYPES:
            m = split_metrics["selections"][selection_type]
            vg = m["vs_gold"]
            vc = m["vs_control"]
            lines.append(
                f"| {selection_type} | {vg.get('soft_ce', 0):.4f} | {vg.get('accuracy', 0):.4f} | "
                f"{vg.get('macro_f1', 0):.4f} | {vc.get('soft_ce', 0):.4f} | {vc.get('accuracy', 0):.4f} | "
                f"{m.get('mean_selected_candidate_index', 0):.2f} | {m.get('mean_generated_words', 0):.2f} | "
                f"{m.get('empty_rate', 0):.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple predicted-control replies per context, score each reply with the stance predictor, "
            "and rerank candidates by predicted-control stance consistency."
        )
    )
    parser.add_argument("--prepared", required=True, help="Gold prepared directory.")
    parser.add_argument("--predicted-prepared", required=True, help="Prepared directory containing predicted target vectors.")
    parser.add_argument("--generator-dir", required=True)
    parser.add_argument("--stance-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="Override base generator model. Defaults to generator config model.")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--max-examples", type=int, default=128, help="Per split. 0 means full split.")
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--response-length-words", type=int, default=0, help="Optional evaluation-time target reply length in words. 0 disables the prompt hint.")
    parser.add_argument("--no-sample", action="store_true", help="Disable sampling. With num-candidates > 1 this usually gives duplicates.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--score-max-length", type=int, default=384)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    args.do_sample = not args.no_sample

    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be >= 1")

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(args.prepared)
    predicted_prepared = Path(args.predicted_prepared)

    if args.no_sample and args.num_candidates > 1:
        print("warning: --no-sample with --num-candidates > 1 usually produces duplicate candidates.", flush=True)

    lm, gen_tokenizer, projector, prefix_tokens, vector_dim, base_model = load_generator(
        Path(args.generator_dir), args.model, args.bf16, args.device
    )
    candidates_by_split: Dict[str, List[dict]] = {}
    selection_summary: Dict[str, Any] = {}
    for split_index, split in enumerate(parse_splits(args.splits)):
        set_seed(split_seed(args.seed, split, split_index))
        gold_all = load_prepared_split(prepared, split)
        pred_all = load_prepared_split(predicted_prepared, split)
        gold_rows, indices = select_rows(gold_all, args.max_examples, args.seed, args.sample_strategy)
        pred_rows = aligned_predicted_rows(gold_rows, pred_all, indices)
        selection_summary[split] = {"available": len(gold_all), "selected": len(gold_rows), "indices": indices[:20]}
        records = generate_candidates_for_split(lm, gen_tokenizer, projector, split, gold_rows, pred_rows, args, prefix_tokens)
        candidates_by_split[split] = records
        write_jsonl(out / f"candidates_{split}.jsonl", records)

    unload_generator(lm, projector)

    stance_model, stance_tokenizer = load_stance_scorer(Path(args.stance_dir), prepared, args.device)
    selected_by_split: Dict[str, List[dict]] = {}
    for split, records in candidates_by_split.items():
        pred_cluster = score_records(records, stance_model, stance_tokenizer, args)
        attach_candidate_scores(records, pred_cluster, args)
        selected = select_reranked(records)
        selected_by_split[split] = selected
        write_jsonl(out / f"candidates_{split}.scored.jsonl", records)
        write_jsonl(out / f"selected_{split}.jsonl", selected)

    metrics = aggregate_metrics(candidates_by_split, selected_by_split)
    summary = {
        "generator_dir": args.generator_dir,
        "base_model": base_model,
        "stance_dir": args.stance_dir,
        "prepared": args.prepared,
        "predicted_prepared": args.predicted_prepared,
        "splits": parse_splits(args.splits),
        "max_examples": args.max_examples,
        "num_candidates": args.num_candidates,
        "response_length_words": args.response_length_words,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "selection": selection_summary,
        "metrics": metrics,
    }
    write_json(out / "metrics.json", summary)
    write_markdown_report(out / "report.md", metrics, args)


if __name__ == "__main__":
    main()
