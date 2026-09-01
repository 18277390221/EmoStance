from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .data import load_prepared_split, read_json, write_json, write_jsonl
from .generate_and_score_controls import generate_one, load_generator, load_stance_scorer, score_records, unload_generator
from .metrics import kl_divergence, soft_cross_entropy
from .run_ablations import cluster_prototypes as compute_cluster_prototypes


DEFAULT_CONTROL_CLUSTERS = "1,2,7"
DEFAULT_CONTEXT_CLUSTERS = "7"


def parse_int_list(value: str, *, allow_all: bool = False) -> List[int] | None:
    value = value.strip().lower()
    if allow_all and value in {"all", "*"}:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_splits(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def onehot(cluster: int, num_clusters: int) -> List[float]:
    arr = [0.0] * num_clusters
    arr[int(cluster)] = 1.0
    return arr


def load_cluster_prototypes(prepared: Path, path: str | None) -> np.ndarray:
    if path:
        return np.asarray(read_json(path), dtype=np.float32)
    default_paths = [
        prepared.parent / "ablations" / "cluster_prototypes.json",
        prepared.parent / "ablations_role_aware" / "cluster_prototypes.json",
        prepared.parent / "ablations_role_aware_no_focal" / "cluster_prototypes.json",
    ]
    for candidate in default_paths:
        if candidate.exists():
            return np.asarray(read_json(candidate), dtype=np.float32)
    meta = read_json(prepared / "meta.json")
    train_rows = load_prepared_split(prepared, "train")
    return compute_cluster_prototypes(train_rows, int(meta["num_clusters"]), int(meta["vector_dim"]))


def top_cluster(row: dict) -> int:
    return int(np.argmax(np.asarray(row["target_cluster"], dtype=np.float64)))


def sorted_probs(row: dict) -> np.ndarray:
    return np.sort(np.asarray(row["target_cluster"], dtype=np.float64))[::-1]


def top_prob(row: dict) -> float:
    return float(sorted_probs(row)[0])


def margin(row: dict) -> float:
    probs = sorted_probs(row)
    return float(probs[0] - probs[1]) if len(probs) > 1 else float(probs[0])


def transition_allowed(row: dict, transitions: Sequence[str] | None) -> bool:
    if transitions is None:
        return True
    return str(row.get("transition", "A->B")) in set(transitions)


def select_contexts(rows: List[dict], args) -> Tuple[List[dict], List[int]]:
    context_clusters = parse_int_list(args.context_clusters, allow_all=True)
    transitions = None if args.transitions.strip().lower() in {"all", "*"} else [x.strip() for x in args.transitions.split(",") if x.strip()]
    candidates: List[Tuple[int, dict]] = []
    for idx, row in enumerate(rows):
        c = top_cluster(row)
        if context_clusters is not None and c not in context_clusters:
            continue
        if top_prob(row) < args.min_target_prob:
            continue
        if margin(row) < args.min_target_margin:
            continue
        if not transition_allowed(row, transitions):
            continue
        candidates.append((idx, row))

    if args.sample_strategy == "random":
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
    if args.max_examples and args.max_examples > 0:
        candidates = candidates[: args.max_examples]
    if args.sample_strategy != "random":
        candidates = sorted(candidates, key=lambda x: x[0])
    indices = [idx for idx, _ in candidates]
    return [row for _, row in candidates], indices


def word_jaccard(a: str, b: str) -> float:
    sa = {token.strip(".,!?;:'\"()[]{} ").lower() for token in a.split()}
    sb = {token.strip(".,!?;:'\"()[]{} ").lower() for token in b.split()}
    sa.discard("")
    sb.discard("")
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def generate_split(lm, tokenizer, projector, split: str, rows: List[dict], prototypes: np.ndarray, args, prefix_tokens: int) -> List[dict]:
    control_clusters = parse_int_list(args.control_clusters) or []
    records: List[dict] = []
    total = len(rows) * len(control_clusters)
    done = 0
    for i, row in enumerate(rows):
        gold_cluster = top_cluster(row)
        for control_cluster in control_clusters:
            vector = prototypes[int(control_cluster)]
            generated = generate_one(lm, tokenizer, projector, row, vector, args, prefix_tokens)
            records.append(
                {
                    "split": split,
                    "example_index": i,
                    "dialogue_id": row.get("dialogue_id"),
                    "turn_id": row.get("turn_id"),
                    "role": row.get("role"),
                    "next_role": row.get("next_role"),
                    "transition": row.get("transition"),
                    "control_type": f"cluster_{int(control_cluster):02d}",
                    "control_cluster_id": int(control_cluster),
                    "situation": row.get("situation", ""),
                    "context": row.get("context", ""),
                    "reference_response": row.get("response", ""),
                    "generated_response": generated,
                    "gold_target_cluster": row.get("target_cluster"),
                    "gold_target_top1": gold_cluster,
                    "gold_target_top1_prob": top_prob(row),
                    "gold_target_margin": margin(row),
                    "control_target_cluster": onehot(int(control_cluster), prototypes.shape[0]),
                    "control_target_top1": int(control_cluster),
                }
            )
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"generated {split}: {done}/{total}", flush=True)
    return records


def attach_scores(records_by_split: Dict[str, List[dict]], stance_model, stance_tokenizer, args) -> None:
    for records in records_by_split.values():
        pred = score_records(records, stance_model, stance_tokenizer, args)
        for record, q in zip(records, pred):
            control_cluster = int(record["control_cluster_id"])
            gold = np.asarray(record["gold_target_cluster"], dtype=np.float64)
            control = np.asarray(record["control_target_cluster"], dtype=np.float64)
            record["generated_source_cluster_pred"] = q.astype(float).tolist()
            record["generated_source_top1"] = int(np.argmax(q))
            record["generated_control_prob"] = float(q[control_cluster])
            record["generated_gold_prob"] = float(q[int(record["gold_target_top1"])])
            record["generated_vs_control_soft_ce"] = soft_cross_entropy(control, q)
            record["generated_vs_control_kl"] = kl_divergence(control, q)
            record["generated_vs_gold_soft_ce"] = soft_cross_entropy(gold, q)
            record["generated_vs_gold_kl"] = kl_divergence(gold, q)
            record["generated_word_count"] = len(str(record.get("generated_response", "")).split())


def summarize_records(records_by_split: Dict[str, List[dict]], control_clusters: Sequence[int]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for split, records in records_by_split.items():
        split_summary: Dict[str, Any] = {"by_control": {}, "pairwise": {}}
        for cluster in control_clusters:
            subset = [r for r in records if int(r["control_cluster_id"]) == int(cluster)]
            if not subset:
                continue
            top_counter = Counter(int(r["generated_source_top1"]) for r in subset)
            split_summary["by_control"][str(cluster)] = {
                "n": len(subset),
                "vs_control_acc": float(np.mean([int(r["generated_source_top1"]) == int(cluster) for r in subset])),
                "mean_control_prob": float(np.mean([r["generated_control_prob"] for r in subset])),
                "mean_vs_control_ce": float(np.mean([r["generated_vs_control_soft_ce"] for r in subset])),
                "vs_gold_acc": float(np.mean([int(r["generated_source_top1"]) == int(r["gold_target_top1"]) for r in subset])),
                "mean_vs_gold_ce": float(np.mean([r["generated_vs_gold_soft_ce"] for r in subset])),
                "mean_words": float(np.mean([r["generated_word_count"] for r in subset])),
                "generated_top1_distribution": {str(k): int(v) for k, v in top_counter.most_common()},
            }

        by_example: Dict[int, Dict[int, dict]] = defaultdict(dict)
        for record in records:
            by_example[int(record["example_index"])][int(record["control_cluster_id"])] = record
        pair_rows = []
        for example_idx, items in by_example.items():
            if not all(cluster in items for cluster in control_clusters):
                continue
            top1s = {cluster: int(items[cluster]["generated_source_top1"]) for cluster in control_clusters}
            responses = {cluster: str(items[cluster].get("generated_response", "")) for cluster in control_clusters}
            row = {
                "example_index": example_idx,
                "all_top1_same": len(set(top1s.values())) == 1,
                "num_distinct_top1": len(set(top1s.values())),
                "mean_pairwise_jaccard": float(np.mean([word_jaccard(responses[a], responses[b]) for i, a in enumerate(control_clusters) for b in control_clusters[i + 1 :]])),
            }
            if 7 in control_clusters:
                row["c7_top1_is_7"] = top1s[7] == 7
                row["c7_prob_lift_over_others"] = float(
                    items[7]["generated_source_cluster_pred"][7]
                    - max(items[c]["generated_source_cluster_pred"][7] for c in control_clusters if c != 7)
                )
            pair_rows.append(row)
        if pair_rows:
            split_summary["pairwise"] = {
                "n_contexts": len(pair_rows),
                "top1_changes_rate": float(np.mean([not r["all_top1_same"] for r in pair_rows])),
                "mean_num_distinct_top1": float(np.mean([r["num_distinct_top1"] for r in pair_rows])),
                "mean_pairwise_jaccard": float(np.mean([r["mean_pairwise_jaccard"] for r in pair_rows])),
            }
            if 7 in control_clusters:
                split_summary["pairwise"].update(
                    {
                        "c7_top1_is_7_rate": float(np.mean([r.get("c7_top1_is_7", False) for r in pair_rows])),
                        "mean_c7_prob_lift_over_others": float(np.mean([r.get("c7_prob_lift_over_others", 0.0) for r in pair_rows])),
                    }
                )
        summary[split] = split_summary
    return summary


def write_annotation_sheet(out: Path, split: str, records: List[dict], control_clusters: Sequence[int], max_rows: int) -> None:
    by_example: Dict[int, Dict[int, dict]] = defaultdict(dict)
    base_rows: Dict[int, dict] = {}
    for record in records:
        idx = int(record["example_index"])
        by_example[idx][int(record["control_cluster_id"])] = record
        base_rows[idx] = record
    fieldnames = [
        "example_index",
        "gold_target_top1",
        "gold_target_top1_prob",
        "transition",
        "context",
        "reference_response",
    ]
    for cluster in control_clusters:
        fieldnames += [f"c{cluster}_response", f"c{cluster}_scorer_top1", f"c{cluster}_control_prob", f"c{cluster}_vs_control_ce"]
    fieldnames += ["human_c7_distinct", "human_c7_appropriate", "preferred_control", "notes"]

    rows = []
    for idx in sorted(by_example)[:max_rows if max_rows and max_rows > 0 else None]:
        item = base_rows[idx]
        row = {
            "example_index": idx,
            "gold_target_top1": item.get("gold_target_top1"),
            "gold_target_top1_prob": f'{float(item.get("gold_target_top1_prob", 0.0)):.3f}',
            "transition": item.get("transition"),
            "context": item.get("context", ""),
            "reference_response": item.get("reference_response", ""),
            "human_c7_distinct": "",
            "human_c7_appropriate": "",
            "preferred_control": "",
            "notes": "",
        }
        for cluster in control_clusters:
            rec = by_example[idx].get(int(cluster), {})
            row[f"c{cluster}_response"] = rec.get("generated_response", "")
            row[f"c{cluster}_scorer_top1"] = rec.get("generated_source_top1", "")
            row[f"c{cluster}_control_prob"] = f'{float(rec.get("generated_control_prob", 0.0)):.3f}' if rec else ""
            row[f"c{cluster}_vs_control_ce"] = f'{float(rec.get("generated_vs_control_soft_ce", 0.0)):.3f}' if rec else ""
        rows.append(row)

    csv_path = out / f"annotation_sheet_{split}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [f"# Counterfactual Annotation Sheet: {split}", "", "| idx | gold | context | c1 | c2 | c7 | c7 distinct? | c7 appropriate? |", "|---:|---:|---|---|---|---|---|---|"]
    for row in rows[: min(len(rows), 40)]:
        def short(value: str, n: int = 140) -> str:
            value = str(value).replace("\n", "<br>").replace("|", "\\|")
            return value[:n] + ("..." if len(value) > n else "")
        md_lines.append(
            f"| {row['example_index']} | {row['gold_target_top1']} | {short(row['context'], 180)} | "
            f"{short(row.get('c1_response', ''))} | {short(row.get('c2_response', ''))} | {short(row.get('c7_response', ''))} |  |  |"
        )
    (out / f"annotation_sheet_{split}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_report(out: Path, summary: Dict[str, Any], args) -> None:
    lines = [
        "# Counterfactual Cluster Control Report",
        "",
        f"Generator: `{args.generator_dir}`",
        f"Stance scorer: `{args.stance_dir}`",
        f"Control clusters: `{args.control_clusters}`",
        f"Context clusters: `{args.context_clusters}`",
        f"Min target prob: `{args.min_target_prob}`",
        f"Max examples per split: `{args.max_examples}`",
        "",
    ]
    for split, split_summary in summary.items():
        lines += [f"## {split}", "", "### By Control", "", "| control | n | vs-control acc | control prob | CE to control | vs-gold acc | CE to gold | words | generated top1 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for cluster, metrics in split_summary.get("by_control", {}).items():
            lines.append(
                f"| {cluster} | {metrics['n']} | {metrics['vs_control_acc']:.4f} | {metrics['mean_control_prob']:.4f} | "
                f"{metrics['mean_vs_control_ce']:.4f} | {metrics['vs_gold_acc']:.4f} | {metrics['mean_vs_gold_ce']:.4f} | "
                f"{metrics['mean_words']:.2f} | {metrics['generated_top1_distribution']} |"
            )
        p = split_summary.get("pairwise", {})
        if p:
            lines += [
                "",
                "### Same-Context Distinctiveness",
                "",
                f"- contexts: {p.get('n_contexts')}",
                f"- top1 changes rate: {p.get('top1_changes_rate', 0):.4f}",
                f"- mean distinct generated top1 count: {p.get('mean_num_distinct_top1', 0):.4f}",
                f"- mean pairwise lexical Jaccard: {p.get('mean_pairwise_jaccard', 0):.4f}",
            ]
            if "c7_top1_is_7_rate" in p:
                lines += [
                    f"- c7 control -> scorer top1 7 rate: {p.get('c7_top1_is_7_rate', 0):.4f}",
                    f"- mean c7 prob lift over c1/c2 controls: {p.get('mean_c7_prob_lift_over_others', 0):.4f}",
                ]
        lines.append("")
    lines += [
        "## Human Review Columns",
        "",
        "The CSV annotation sheet includes blank columns for:",
        "",
        "- `human_c7_distinct`: whether c7 is visibly different from c1/c2.",
        "- `human_c7_appropriate`: whether the playful/teasing style is appropriate for the context.",
        "- `preferred_control`: which control gives the best response for this context.",
        "",
        "Use this report as an automatic screen, then manually inspect the annotation sheet before claiming generation value.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate same-context counterfactual replies using selected cluster prototype controls and score them.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--generator-dir", required=True)
    parser.add_argument("--stance-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cluster-prototypes", default=None, help="cluster_prototypes.json. Defaults to known ablation outputs or computed from train.")
    parser.add_argument("--model", default=None, help="Override base generator model. Defaults to generator config model.")
    parser.add_argument("--splits", default="dev")
    parser.add_argument("--control-clusters", default=DEFAULT_CONTROL_CLUSTERS)
    parser.add_argument("--context-clusters", default=DEFAULT_CONTEXT_CLUSTERS, help="Gold target clusters to sample contexts from, or 'all'.")
    parser.add_argument("--transitions", default="A->B", help="Comma-separated transitions to include, or 'all'.")
    parser.add_argument("--min-target-prob", type=float, default=0.8)
    parser.add_argument("--min-target-margin", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=40, help="Per split. 0 means all matching contexts.")
    parser.add_argument("--annotation-max-rows", type=int, default=80)
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="random")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--score-max-length", type=int, default=384)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(args.prepared)
    prototypes = load_cluster_prototypes(prepared, args.cluster_prototypes)
    control_clusters = parse_int_list(args.control_clusters) or []
    for cluster in control_clusters:
        if cluster < 0 or cluster >= prototypes.shape[0]:
            raise ValueError(f"Control cluster {cluster} outside prototype shape {prototypes.shape}.")

    lm, gen_tokenizer, projector, prefix_tokens, vector_dim, base_model = load_generator(Path(args.generator_dir), args.model, args.bf16, args.device)
    if vector_dim != prototypes.shape[1]:
        raise ValueError(f"Generator vector dim {vector_dim} does not match prototypes {prototypes.shape}.")

    records_by_split: Dict[str, List[dict]] = {}
    selection: Dict[str, Any] = {}
    for split in parse_splits(args.splits):
        all_rows = load_prepared_split(prepared, split)
        rows, indices = select_contexts(all_rows, args)
        selection[split] = {"available": len(all_rows), "selected": len(rows), "indices": indices[:50]}
        if not rows:
            records_by_split[split] = []
            continue
        records = generate_split(lm, gen_tokenizer, projector, split, rows, prototypes, args, prefix_tokens)
        records_by_split[split] = records
        write_jsonl(out / f"generations_{split}.jsonl", records)

    unload_generator(lm, projector)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stance_model, stance_tokenizer = load_stance_scorer(Path(args.stance_dir), prepared, args.device)
    attach_scores(records_by_split, stance_model, stance_tokenizer, args)
    for split, records in records_by_split.items():
        write_jsonl(out / f"generations_{split}.scored.jsonl", records)
        write_annotation_sheet(out, split, records, control_clusters, args.annotation_max_rows)

    summary = summarize_records(records_by_split, control_clusters)
    write_json(
        out / "metrics.json",
        {
            "generator_dir": args.generator_dir,
            "base_model": base_model,
            "stance_dir": args.stance_dir,
            "prepared": args.prepared,
            "cluster_prototypes": args.cluster_prototypes,
            "control_clusters": control_clusters,
            "context_clusters": args.context_clusters,
            "selection": selection,
            "metrics": summary,
        },
    )
    write_report(out, summary, args)


if __name__ == "__main__":
    main()
