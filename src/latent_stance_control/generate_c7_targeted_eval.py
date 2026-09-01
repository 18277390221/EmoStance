from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .data import load_prepared_split, read_json, write_json, write_jsonl
from .generate_and_score_controls import (
    generate_one,
    generation_prompt_text,
    load_generator,
    load_stance_scorer,
    mean_text_stats,
    parse_splits,
    prompt_text,
    score_records,
    unload_generator,
)
from .metrics import evaluate_cluster_predictions, kl_divergence, normalize_prob, soft_cross_entropy


CONTROL_TYPES = ("gold", "ungated_predicted", "c7_gated_predicted", "forced_c7")


def parse_groups(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def top1(prob: Sequence[float]) -> int:
    return int(np.argmax(np.asarray(prob, dtype=np.float64)))


def one_hot(size: int, index: int) -> List[float]:
    arr = np.zeros(size, dtype=np.float64)
    arr[index] = 1.0
    return arr.astype(float).tolist()


def assert_aligned(split: str, gold: List[dict], ungated: List[dict], gated: List[dict]) -> None:
    if len(gold) != len(ungated) or len(gold) != len(gated):
        raise ValueError(f"{split}: split size mismatch: gold={len(gold)} ungated={len(ungated)} gated={len(gated)}")
    for i, (g, u, c) in enumerate(zip(gold, ungated, gated)):
        key_g = (g.get("dialogue_id"), int(g.get("turn_id", -1)))
        key_u = (u.get("dialogue_id"), int(u.get("turn_id", -2)))
        key_c = (c.get("dialogue_id"), int(c.get("turn_id", -3)))
        if key_g != key_u or key_g != key_c:
            raise ValueError(f"{split}: row alignment mismatch at {i}: gold={key_g} ungated={key_u} gated={key_c}")


def gate_active(row: dict, min_gate_prob: float | None = None) -> bool:
    if min_gate_prob is not None:
        return float(row.get("c7_gate_prob", 0.0)) >= min_gate_prob
    return int(row.get("c7_gate_pred", 0)) == 1


def reason_flags(gold_row: dict, ungated_row: dict, gated_row: dict, c7_cluster: int, min_gate_prob: float | None) -> Dict[str, bool]:
    return {
        "gold_c7": top1(gold_row["target_cluster"]) == c7_cluster,
        "gate_active": gate_active(gated_row, min_gate_prob=min_gate_prob),
        "gated_c7": top1(gated_row["target_cluster"]) == c7_cluster,
        "ungated_c7": top1(
            gated_row.get("ungated_target_cluster", ungated_row.get("target_cluster"))
        )
        == c7_cluster,
    }


def selected_by(flags: Dict[str, bool], groups: Sequence[str], mode: str) -> bool:
    values = [flags.get(group, False) for group in groups]
    if not values:
        return False
    if mode == "intersection":
        return all(values)
    return any(values)


def select_indices(
    split: str,
    gold_rows: List[dict],
    ungated_rows: List[dict],
    gated_rows: List[dict],
    args: argparse.Namespace,
    split_seed: int,
) -> Tuple[List[int], Dict[str, Any]]:
    groups = parse_groups(args.target_groups)
    candidates: List[int] = []
    reason_counts = {group: 0 for group in ["gold_c7", "gate_active", "gated_c7", "ungated_c7"]}
    candidate_reason_counts = {group: 0 for group in reason_counts}
    transition_counts: Dict[str, int] = {}
    candidate_transition_counts: Dict[str, int] = {}

    allowed_transitions = None if args.transitions.lower() in {"all", "*"} else set(parse_groups(args.transitions))
    for i, (gold_row, ungated_row, gated_row) in enumerate(zip(gold_rows, ungated_rows, gated_rows)):
        transition = str(gold_row.get("transition", ""))
        if allowed_transitions is not None and transition not in allowed_transitions:
            continue
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        flags = reason_flags(gold_row, ungated_row, gated_row, args.c7_cluster, args.min_gate_prob)
        for key, value in flags.items():
            reason_counts[key] += int(value)
        if selected_by(flags, groups, args.selection_mode):
            candidates.append(i)
            candidate_transition_counts[transition] = candidate_transition_counts.get(transition, 0) + 1
            for key, value in flags.items():
                candidate_reason_counts[key] += int(value)

    indices = list(candidates)
    if args.max_examples and args.max_examples > 0 and len(indices) > args.max_examples:
        if args.sample_strategy == "random":
            rng = random.Random(split_seed)
            indices = sorted(rng.sample(indices, args.max_examples))
        else:
            indices = indices[: args.max_examples]

    selected_reason_counts = {group: 0 for group in reason_counts}
    selected_transition_counts: Dict[str, int] = {}
    for i in indices:
        transition = str(gold_rows[i].get("transition", ""))
        selected_transition_counts[transition] = selected_transition_counts.get(transition, 0) + 1
        flags = reason_flags(gold_rows[i], ungated_rows[i], gated_rows[i], args.c7_cluster, args.min_gate_prob)
        for key, value in flags.items():
            selected_reason_counts[key] += int(value)

    summary = {
        "split": split,
        "available": len(gold_rows),
        "candidate_count": len(candidates),
        "selected": len(indices),
        "target_groups": groups,
        "selection_mode": args.selection_mode,
        "reason_counts": reason_counts,
        "candidate_reason_counts": candidate_reason_counts,
        "selected_reason_counts": selected_reason_counts,
        "transition_counts": transition_counts,
        "candidate_transition_counts": candidate_transition_counts,
        "selected_transition_counts": selected_transition_counts,
        "indices_preview": indices[:20],
    }
    return indices, summary


def ungated_control(row_from_ungated_prepared: dict, row_from_gated_prepared: dict, source: str) -> Tuple[np.ndarray, List[float], str]:
    if source == "gated_fields" and "ungated_target_vector" in row_from_gated_prepared and "ungated_target_cluster" in row_from_gated_prepared:
        return (
            np.asarray(row_from_gated_prepared["ungated_target_vector"], dtype=np.float32),
            list(row_from_gated_prepared["ungated_target_cluster"]),
            "gated_prepared.ungated_fields",
        )
    return (
        np.asarray(row_from_ungated_prepared["target_vector"], dtype=np.float32),
        list(row_from_ungated_prepared["target_cluster"]),
        "ungated_prepared.target",
    )


def make_control_specs(
    gold_row: dict,
    ungated_row: dict,
    gated_row: dict,
    forced_cluster: List[float],
    forced_vector: np.ndarray,
    ungated_source: str,
) -> Dict[str, Dict[str, Any]]:
    ungated_vector, ungated_cluster, resolved_ungated_source = ungated_control(ungated_row, gated_row, ungated_source)
    return {
        "gold": {
            "vector": np.asarray(gold_row["target_vector"], dtype=np.float32),
            "cluster": list(gold_row["target_cluster"]),
            "source": "gold.target",
        },
        "ungated_predicted": {
            "vector": ungated_vector,
            "cluster": ungated_cluster,
            "source": resolved_ungated_source,
        },
        "c7_gated_predicted": {
            "vector": np.asarray(gated_row["target_vector"], dtype=np.float32),
            "cluster": list(gated_row["target_cluster"]),
            "source": "gated_prepared.target",
        },
        "forced_c7": {
            "vector": forced_vector.astype(np.float32),
            "cluster": forced_cluster,
            "source": "cluster_prototype.c7",
        },
    }


def generate_split(
    split: str,
    indices: List[int],
    gold_rows: List[dict],
    ungated_rows: List[dict],
    gated_rows: List[dict],
    forced_cluster: List[float],
    forced_vector: np.ndarray,
    lm,
    tokenizer,
    projector,
    args: argparse.Namespace,
    prefix_tokens: int,
) -> List[dict]:
    records: List[dict] = []
    total = len(indices) * len(CONTROL_TYPES)
    done = 0
    for example_index, row_index in enumerate(indices):
        gold_row = gold_rows[row_index]
        ungated_row = ungated_rows[row_index]
        gated_row = gated_rows[row_index]
        flags = reason_flags(gold_row, ungated_row, gated_row, args.c7_cluster, args.min_gate_prob)
        specs = make_control_specs(gold_row, ungated_row, gated_row, forced_cluster, forced_vector, args.ungated_source)
        for control_type in CONTROL_TYPES:
            spec = specs[control_type]
            generated = generate_one(lm, tokenizer, projector, gold_row, spec["vector"], args, prefix_tokens)
            records.append(
                {
                    "split": split,
                    "example_index": example_index,
                    "dataset_index": row_index,
                    "dialogue_id": gold_row.get("dialogue_id"),
                    "turn_id": gold_row.get("turn_id"),
                    "role": gold_row.get("role"),
                    "next_role": gold_row.get("next_role"),
                    "transition": gold_row.get("transition"),
                    "selection_flags": flags,
                    "c7_gate_prob": gated_row.get("c7_gate_prob"),
                    "c7_gate_pred": gated_row.get("c7_gate_pred"),
                    "control_type": control_type,
                    "control_source": spec["source"],
                    "situation": gold_row.get("situation", ""),
                    "context": gold_row.get("context", ""),
                    "prompt": generation_prompt_text(gold_row, args),
                    "reference_response": gold_row.get("response", ""),
                    "generated_response": generated,
                    "gold_target_cluster": gold_row.get("target_cluster"),
                    "gold_target_top1": top1(gold_row["target_cluster"]),
                    "ungated_target_cluster": specs["ungated_predicted"]["cluster"],
                    "ungated_target_top1": top1(specs["ungated_predicted"]["cluster"]),
                    "gated_target_cluster": gated_row.get("target_cluster"),
                    "gated_target_top1": top1(gated_row["target_cluster"]),
                    "control_target_cluster": spec["cluster"],
                    "control_target_top1": top1(spec["cluster"]),
                }
            )
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"generated {split}: {done}/{total}", flush=True)
    return records


def c7_prf(reference_top: np.ndarray, generated_top: np.ndarray, c7_cluster: int) -> Dict[str, float]:
    ref = reference_top == c7_cluster
    pred = generated_top == c7_cluster
    tp = float((ref & pred).sum())
    fp = float((~ref & pred).sum())
    fn = float((ref & ~pred).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "c7_precision": float(precision),
        "c7_recall": float(recall),
        "c7_f1": float(f1),
    }


def subset_rate(mask: np.ndarray, generated_top: np.ndarray, c7_cluster: int) -> float:
    if not mask.any():
        return 0.0
    return float((generated_top[mask] == c7_cluster).mean())


def metric_block(records: List[dict], pred_cluster: np.ndarray, c7_cluster: int) -> Dict[str, Any]:
    gold = normalize_prob(np.asarray([record["gold_target_cluster"] for record in records], dtype=np.float64))
    control = normalize_prob(np.asarray([record["control_target_cluster"] for record in records], dtype=np.float64))
    generated_top = np.argmax(normalize_prob(pred_cluster), axis=-1)
    gold_top = np.asarray([int(record["gold_target_top1"]) for record in records], dtype=np.int64)
    control_top = np.asarray([int(record["control_target_top1"]) for record in records], dtype=np.int64)
    generated_c7 = generated_top == c7_cluster
    gold_c7 = gold_top == c7_cluster
    control_c7 = control_top == c7_cluster
    out: Dict[str, Any] = {
        "examples": len(records),
        "vs_gold": evaluate_cluster_predictions(gold, pred_cluster),
        "vs_control": evaluate_cluster_predictions(control, pred_cluster),
        "mean_generated_c7_prob": float(normalize_prob(pred_cluster)[:, c7_cluster].mean()) if len(records) else 0.0,
        "generated_c7_count": int(generated_c7.sum()),
        "generated_c7_rate": float(generated_c7.mean()) if len(records) else 0.0,
        "gold_c7_count": int(gold_c7.sum()),
        "gold_c7_rate": float(gold_c7.mean()) if len(records) else 0.0,
        "control_c7_count": int(control_c7.sum()),
        "control_c7_rate": float(control_c7.mean()) if len(records) else 0.0,
        "generated_c7_rate_on_gold_c7": subset_rate(gold_c7, generated_top, c7_cluster),
        "generated_c7_rate_on_control_c7": subset_rate(control_c7, generated_top, c7_cluster),
        "top1_acc_vs_gold_on_gold_c7": float((generated_top[gold_c7] == gold_top[gold_c7]).mean()) if gold_c7.any() else 0.0,
        "top1_acc_vs_control_on_control_c7": float((generated_top[control_c7] == control_top[control_c7]).mean()) if control_c7.any() else 0.0,
        "c7_vs_gold": c7_prf(gold_top, generated_top, c7_cluster),
        "c7_vs_control": c7_prf(control_top, generated_top, c7_cluster),
        "generated_top1_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(generated_top, return_counts=True))
        },
    }
    out.update(mean_text_stats(records))
    return out


def attach_scores_and_metrics(records_by_split: Dict[str, List[dict]], stance_model, stance_tokenizer, args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    for split, records in records_by_split.items():
        pred = score_records(records, stance_model, stance_tokenizer, args)
        for record, q in zip(records, pred):
            q = normalize_prob(q)[0]
            record["generated_source_cluster_pred"] = q.astype(float).tolist()
            record["generated_source_top1"] = int(np.argmax(q))
            gold = np.asarray(record["gold_target_cluster"], dtype=np.float64)
            control = np.asarray(record["control_target_cluster"], dtype=np.float64)
            record["generated_vs_gold_soft_ce"] = soft_cross_entropy(gold, q)
            record["generated_vs_gold_kl"] = kl_divergence(gold, q)
            record["generated_vs_control_soft_ce"] = soft_cross_entropy(control, q)
            record["generated_vs_control_kl"] = kl_divergence(control, q)

        metrics[split] = {}
        ungated_rate = 0.0
        for control_type in CONTROL_TYPES:
            idx = [i for i, record in enumerate(records) if record["control_type"] == control_type]
            subset = [records[i] for i in idx]
            block = metric_block(subset, pred[idx], args.c7_cluster)
            metrics[split][control_type] = block
            if control_type == "ungated_predicted":
                ungated_rate = float(block["generated_c7_rate"])
        for control_type in CONTROL_TYPES:
            metrics[split][control_type]["c7_lift_over_ungated"] = float(
                metrics[split][control_type]["generated_c7_rate"] - ungated_rate
            )
    return metrics


def write_report(path: Path, metrics: Dict[str, Dict[str, Any]], selection: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# C7 Targeted Generation Evaluation",
        "",
        f"Generator: `{args.generator_dir}`",
        f"Stance scorer: `{args.stance_dir}`",
        f"Gold prepared: `{args.prepared}`",
        f"Ungated prepared: `{args.ungated_prepared}`",
        f"Gated prepared: `{args.gated_prepared}`",
        f"Selection groups: `{args.target_groups}`",
        f"Selection mode: `{args.selection_mode}`",
        f"Ungated source: `{args.ungated_source}`",
        f"Max examples per split: `{args.max_examples}`",
        "",
    ]
    for split, summary in selection.items():
        lines += [
            f"## {split}",
            "",
            f"- available: `{summary.get('available', 0)}`",
            f"- candidates: `{summary.get('candidate_count', 0)}`",
            f"- selected: `{summary.get('selected', 0)}`",
            f"- selected reason counts: `{summary.get('selected_reason_counts', {})}`",
            "",
            "| control | vs_gold CE | vs_gold acc | vs_gold macro-F1 | vs_control CE | vs_control acc | generated c7 rate | c7 lift | control c7 rate | c7 on control-c7 | mean words |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for control_type in CONTROL_TYPES:
            m = metrics.get(split, {}).get(control_type, {})
            vg = m.get("vs_gold", {})
            vc = m.get("vs_control", {})
            lines.append(
                f"| {control_type} | {vg.get('soft_ce', 0.0):.4f} | {vg.get('accuracy', 0.0):.4f} | "
                f"{vg.get('macro_f1', 0.0):.4f} | {vc.get('soft_ce', 0.0):.4f} | {vc.get('accuracy', 0.0):.4f} | "
                f"{m.get('generated_c7_rate', 0.0):.4f} | {m.get('c7_lift_over_ungated', 0.0):+.4f} | "
                f"{m.get('control_c7_rate', 0.0):.4f} | {m.get('generated_c7_rate_on_control_c7', 0.0):.4f} | "
                f"{m.get('mean_generated_words', 0.0):.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted generation evaluation for c7 controls.")
    parser.add_argument("--prepared", required=True, help="Gold-control prepared directory.")
    parser.add_argument("--ungated-prepared", required=True, help="Prepared directory with ungated predicted controls.")
    parser.add_argument("--gated-prepared", required=True, help="Prepared directory with c7-gated predicted controls.")
    parser.add_argument("--generator-dir", required=True)
    parser.add_argument("--stance-dir", required=True)
    parser.add_argument("--cluster-prototypes", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="Override base generator model. Defaults to generator config model.")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--c7-cluster", type=int, default=7)
    parser.add_argument("--target-groups", default="gold_c7,gate_active,gated_c7")
    parser.add_argument("--selection-mode", choices=["union", "intersection"], default="union")
    parser.add_argument("--transitions", default="all", help="Comma-separated transitions or all. Quote values like 'A->B'.")
    parser.add_argument("--min-gate-prob", type=float, default=None, help="Override c7 gate active definition with a probability threshold.")
    parser.add_argument("--ungated-source", choices=["gated_fields", "ungated_prepared"], default="gated_fields")
    parser.add_argument("--max-examples", type=int, default=128, help="Per split. 0 means all targeted examples.")
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--response-length-words", type=int, default=0, help="Optional evaluation-time target reply length in words. 0 disables the prompt hint.")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--score-max-length", type=int, default=384)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Only select targeted examples and write selection_summary.json.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(args.prepared)
    ungated_prepared = Path(args.ungated_prepared)
    gated_prepared = Path(args.gated_prepared)
    prototypes = np.asarray(read_json(args.cluster_prototypes), dtype=np.float64)
    if args.c7_cluster < 0 or args.c7_cluster >= prototypes.shape[0]:
        raise ValueError(f"c7 cluster {args.c7_cluster} is outside prototype matrix with {prototypes.shape[0]} rows")
    forced_cluster = one_hot(prototypes.shape[0], args.c7_cluster)
    forced_vector = prototypes[args.c7_cluster]

    selection_summary: Dict[str, Any] = {}
    split_payloads: Dict[str, Tuple[List[int], List[dict], List[dict], List[dict]]] = {}
    for split_index, split in enumerate(parse_splits(args.splits)):
        gold_rows = load_prepared_split(prepared, split)
        ungated_rows = load_prepared_split(ungated_prepared, split)
        gated_rows = load_prepared_split(gated_prepared, split)
        if not gold_rows:
            continue
        assert_aligned(split, gold_rows, ungated_rows, gated_rows)
        indices, summary = select_indices(split, gold_rows, ungated_rows, gated_rows, args, args.seed + split_index)
        selection_summary[split] = summary
        split_payloads[split] = (indices, gold_rows, ungated_rows, gated_rows)

    if args.dry_run:
        write_json(
            out / "selection_summary.json",
            {
                "prepared": args.prepared,
                "ungated_prepared": args.ungated_prepared,
                "gated_prepared": args.gated_prepared,
                "c7_cluster": args.c7_cluster,
                "response_length_words": args.response_length_words,
                "selection": selection_summary,
            },
        )
        return

    lm, gen_tokenizer, projector, prefix_tokens, vector_dim, base_model = load_generator(Path(args.generator_dir), args.model, args.bf16, args.device)
    if int(vector_dim) != int(prototypes.shape[1]):
        raise ValueError(f"Generator vector dim {vector_dim} does not match prototype dim {prototypes.shape[1]}")

    records_by_split: Dict[str, List[dict]] = {}
    for split, (indices, gold_rows, ungated_rows, gated_rows) in split_payloads.items():
        records = generate_split(
            split,
            indices,
            gold_rows,
            ungated_rows,
            gated_rows,
            forced_cluster,
            forced_vector,
            lm,
            gen_tokenizer,
            projector,
            args,
            prefix_tokens,
        )
        records_by_split[split] = records
        write_jsonl(out / f"generations_{split}.jsonl", records)

    unload_generator(lm, projector)

    stance_model, stance_tokenizer = load_stance_scorer(Path(args.stance_dir), prepared, args.device)
    metrics = attach_scores_and_metrics(records_by_split, stance_model, stance_tokenizer, args)
    for split, records in records_by_split.items():
        write_jsonl(out / f"generations_{split}.scored.jsonl", records)

    summary = {
        "generator_dir": args.generator_dir,
        "base_model": base_model,
        "stance_dir": args.stance_dir,
        "prepared": args.prepared,
        "ungated_prepared": args.ungated_prepared,
        "gated_prepared": args.gated_prepared,
        "cluster_prototypes": args.cluster_prototypes,
        "c7_cluster": args.c7_cluster,
        "response_length_words": args.response_length_words,
        "controls": list(CONTROL_TYPES),
        "splits": parse_splits(args.splits),
        "selection": selection_summary,
        "metrics": metrics,
    }
    write_json(out / "metrics.json", summary)
    write_report(out / "report.md", metrics, selection_summary, args)


if __name__ == "__main__":
    main()
