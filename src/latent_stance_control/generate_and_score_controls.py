from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .data import build_model_text, load_prepared_split, read_json, write_json, write_jsonl
from .metrics import evaluate_cluster_predictions, normalize_prob, soft_cross_entropy, kl_divergence
from .models import DebertaStancePredictor, StancePrefixProjector


CONTROL_TYPES = ("gold", "predicted", "zero", "shuffled")


def parse_splits(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_rows(rows: List[dict], max_examples: int, seed: int, strategy: str) -> Tuple[List[dict], List[int]]:
    indices = list(range(len(rows)))
    if max_examples and max_examples > 0 and max_examples < len(rows):
        if strategy == "random":
            rng = random.Random(seed)
            indices = sorted(rng.sample(indices, max_examples))
        else:
            indices = indices[:max_examples]
    return [rows[i] for i in indices], indices


def aligned_predicted_rows(gold_rows: List[dict], predicted_rows: List[dict], indices: List[int]) -> List[dict]:
    selected = [predicted_rows[i] for i in indices]
    for pos, (gold, pred) in enumerate(zip(gold_rows, selected)):
        if gold.get("dialogue_id") != pred.get("dialogue_id") or int(gold.get("turn_id", -1)) != int(pred.get("turn_id", -2)):
            raise ValueError(
                "Gold/predicted prepared rows are not aligned at selected position "
                f"{pos}: gold=({gold.get('dialogue_id')}, {gold.get('turn_id')}), "
                f"pred=({pred.get('dialogue_id')}, {pred.get('turn_id')})"
            )
    return selected


def shifted_indices(n: int, seed: int) -> List[int]:
    if n <= 1:
        return [0] * n
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    if any(i == j for i, j in enumerate(perm)):
        perm = perm[1:] + perm[:1]
    return perm


def prompt_text(row: dict) -> str:
    return build_model_text(row) + f'\n{row.get("next_role", "B")}:'


def response_length_instruction(args) -> str:
    target = int(getattr(args, "response_length_words", 0) or 0)
    if target <= 0:
        return ""
    return f"Instruction: Aim for about {target} words in the next reply.\n"


def generation_prompt_text(row: dict, args) -> str:
    return response_length_instruction(args) + prompt_text(row)


def clean_generation(decoded: str, prompt: str) -> str:
    text = decoded.strip()
    prompt = prompt.strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    # Mistral sometimes continues with a speaker tag or a new turn. Keep the first answer span.
    for marker in ("\nA:", "\nB:", "\nUser:", "\nAssistant:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


def decode_generated(output_ids: torch.Tensor, prompt_ids: torch.Tensor, tokenizer: Any, prompt: str, max_new_tokens: int) -> str:
    ids = output_ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    prompt_list = prompt_ids.detach().cpu().tolist()
    if prompt_list and isinstance(prompt_list[0], list):
        prompt_list = prompt_list[0]

    if len(ids) >= len(prompt_list) and ids[: len(prompt_list)] == prompt_list:
        new_ids = ids[len(prompt_list):]
    elif len(ids) > max_new_tokens:
        new_ids = ids[-max_new_tokens:]
    else:
        new_ids = ids

    while new_ids and new_ids[0] in {tokenizer.bos_token_id, tokenizer.pad_token_id}:
        new_ids = new_ids[1:]
    if tokenizer.eos_token_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.eos_token_id)]
    return clean_generation(tokenizer.decode(new_ids, skip_special_tokens=True), prompt)


def load_generator(generator_dir: Path, model_name: Optional[str], bf16: bool, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = read_json(generator_dir / "config.json")
    base_model = model_name or config["model"]
    prefix_tokens = int(config.get("prefix_tokens", 8))
    vector_dim = int(config["vector_dim"])

    tokenizer_path = generator_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.bfloat16} if bf16 else {}
    lm = AutoModelForCausalLM.from_pretrained(base_model, **kwargs).to(device)
    lm.eval()
    for param in lm.parameters():
        param.requires_grad_(False)

    projector = StancePrefixProjector(vector_dim, lm.config.hidden_size, prefix_tokens).to(device)
    projector.load_state_dict(torch.load(generator_dir / "stance_prefix_projector.pt", map_location=device))
    projector.eval()
    return lm, tokenizer, projector, prefix_tokens, vector_dim, base_model


@torch.no_grad()
def generate_one(lm, tokenizer, projector, row: dict, stance_vector: np.ndarray, args, prefix_tokens: int) -> str:
    prompt = generation_prompt_text(row, args)
    encoded = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    token_embeds = lm.get_input_embeddings()(input_ids)
    stance = torch.tensor(stance_vector, dtype=torch.float32, device=args.device).view(1, -1)
    prefix = projector(stance).to(token_embeds.dtype)
    inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=args.device)

    gen_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        gen_kwargs.update({"do_sample": False})
    output_ids = lm.generate(**gen_kwargs)
    return decode_generated(output_ids[0], input_ids[0], tokenizer, prompt, args.max_new_tokens)


def control_specs(rows: List[dict], predicted_rows: List[dict], vector_dim: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    shuffled = shifted_indices(len(rows), seed)
    specs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in CONTROL_TYPES}
    for i, row in enumerate(rows):
        pred_row = predicted_rows[i]
        shuf_row = rows[shuffled[i]]
        specs["gold"].append(
            {
                "vector": np.asarray(row["target_vector"], dtype=np.float32),
                "control_cluster": row["target_cluster"],
                "control_source_dialogue_id": row.get("dialogue_id"),
                "control_source_turn_id": row.get("turn_id"),
            }
        )
        specs["predicted"].append(
            {
                "vector": np.asarray(pred_row["target_vector"], dtype=np.float32),
                "control_cluster": pred_row["target_cluster"],
                "control_source_dialogue_id": pred_row.get("dialogue_id"),
                "control_source_turn_id": pred_row.get("turn_id"),
            }
        )
        specs["zero"].append(
            {
                "vector": np.zeros(vector_dim, dtype=np.float32),
                "control_cluster": None,
                "control_source_dialogue_id": None,
                "control_source_turn_id": None,
            }
        )
        specs["shuffled"].append(
            {
                "vector": np.asarray(shuf_row["target_vector"], dtype=np.float32),
                "control_cluster": shuf_row["target_cluster"],
                "control_source_dialogue_id": shuf_row.get("dialogue_id"),
                "control_source_turn_id": shuf_row.get("turn_id"),
            }
        )
    return specs


def generate_split(lm, tokenizer, projector, split: str, rows: List[dict], predicted_rows: List[dict], vector_dim: int, args, prefix_tokens: int) -> List[dict]:
    specs = control_specs(rows, predicted_rows, vector_dim, args.seed + hash(split) % 100000)
    records: List[dict] = []
    total = len(rows) * len(CONTROL_TYPES)
    done = 0
    for i, row in enumerate(rows):
        for control_type in CONTROL_TYPES:
            spec = specs[control_type][i]
            generated = generate_one(lm, tokenizer, projector, row, spec["vector"], args, prefix_tokens)
            records.append(
                {
                    "split": split,
                    "example_index": i,
                    "dialogue_id": row.get("dialogue_id"),
                    "turn_id": row.get("turn_id"),
                    "role": row.get("role"),
                    "next_role": row.get("next_role"),
                    "control_type": control_type,
                    "control_source_dialogue_id": spec["control_source_dialogue_id"],
                    "control_source_turn_id": spec["control_source_turn_id"],
                    "situation": row.get("situation", ""),
                    "context": row.get("context", ""),
                    "prompt": generation_prompt_text(row, args),
                    "reference_response": row.get("response", ""),
                    "generated_response": generated,
                    "gold_target_cluster": row.get("target_cluster"),
                    "control_target_cluster": spec["control_cluster"],
                    "gold_target_top1": int(np.argmax(row.get("target_cluster"))),
                    "control_target_top1": None if spec["control_cluster"] is None else int(np.argmax(spec["control_cluster"])),
                }
            )
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"generated {split}: {done}/{total}", flush=True)
    return records


def unload_generator(lm, projector) -> None:
    del lm
    del projector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_stance_scorer(stance_dir: Path, prepared: Path, device: str):
    from transformers import AutoTokenizer

    meta = read_json(prepared / "meta.json")
    model = DebertaStancePredictor(str(stance_dir / "encoder"), meta["num_clusters"], meta["vector_dim"]).float().to(device)
    state = torch.load(stance_dir / "stance_predictor.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(stance_dir / "tokenizer", use_fast=True)
    return model, tokenizer


def scoring_text(record: dict) -> str:
    generated = str(record.get("generated_response", "")).strip()
    role = record.get("next_role", "B")
    context = str(record.get("context", ""))
    if generated:
        context = context + f"\n{role}: {generated}"
    else:
        context = context + f"\n{role}:"
    row = {"situation": record.get("situation", ""), "context": context}
    return build_model_text(row)


@torch.no_grad()
def score_records(records: List[dict], stance_model, stance_tokenizer, args) -> np.ndarray:
    preds: List[np.ndarray] = []
    for start in range(0, len(records), args.score_batch_size):
        batch = records[start : start + args.score_batch_size]
        texts = [scoring_text(record) for record in batch]
        encoded = stance_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.score_max_length,
            return_tensors="pt",
        )
        output = stance_model(encoded["input_ids"].to(args.device), encoded["attention_mask"].to(args.device))
        pred = torch.softmax(output.source_logits, dim=-1).detach().cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 0), dtype=np.float32)


def mean_text_stats(records: List[dict]) -> Dict[str, float]:
    lengths = [len(str(record.get("generated_response", "")).split()) for record in records]
    empty = [1 if length == 0 else 0 for length in lengths]
    return {
        "mean_generated_words": float(np.mean(lengths)) if lengths else 0.0,
        "empty_rate": float(np.mean(empty)) if empty else 0.0,
    }


def metric_block(records: List[dict], pred_cluster: np.ndarray) -> Dict[str, Any]:
    gold = np.asarray([record["gold_target_cluster"] for record in records], dtype=np.float64)
    out: Dict[str, Any] = {"vs_gold": evaluate_cluster_predictions(gold, pred_cluster)}
    control_rows = [record for record in records if record.get("control_target_cluster") is not None]
    if control_rows:
        idx = [i for i, record in enumerate(records) if record.get("control_target_cluster") is not None]
        control = np.asarray([records[i]["control_target_cluster"] for i in idx], dtype=np.float64)
        out["vs_control"] = evaluate_cluster_predictions(control, pred_cluster[idx])
    out.update(mean_text_stats(records))
    out["generated_top1_distribution"] = {
        str(k): int(v) for k, v in zip(*np.unique(np.argmax(pred_cluster, axis=-1), return_counts=True))
    }
    return out


def attach_scores_and_metrics(records_by_split: Dict[str, List[dict]], stance_model, stance_tokenizer, args) -> Dict[str, Dict[str, Any]]:
    metrics: Dict[str, Dict[str, Any]] = {}
    for split, records in records_by_split.items():
        pred = score_records(records, stance_model, stance_tokenizer, args)
        for record, q in zip(records, pred):
            record["generated_source_cluster_pred"] = q.astype(float).tolist()
            record["generated_source_top1"] = int(np.argmax(q))
            gold = np.asarray(record["gold_target_cluster"], dtype=np.float64)
            record["generated_vs_gold_soft_ce"] = soft_cross_entropy(gold, q)
            record["generated_vs_gold_kl"] = kl_divergence(gold, q)
            if record.get("control_target_cluster") is not None:
                control = np.asarray(record["control_target_cluster"], dtype=np.float64)
                record["generated_vs_control_soft_ce"] = soft_cross_entropy(control, q)
                record["generated_vs_control_kl"] = kl_divergence(control, q)

        metrics[split] = {}
        for control_type in CONTROL_TYPES:
            subset_idx = [i for i, record in enumerate(records) if record["control_type"] == control_type]
            subset = [records[i] for i in subset_idx]
            metrics[split][control_type] = metric_block(subset, pred[subset_idx])
    return metrics


def write_markdown_report(path: Path, metrics: Dict[str, Dict[str, Any]], args) -> None:
    lines = [
        "# Generate And Score Controls",
        "",
        f"Generator: `{args.generator_dir}`",
        f"Stance scorer: `{args.stance_dir}`",
        f"Max examples per split: `{args.max_examples}`",
        "",
    ]
    for split, split_metrics in metrics.items():
        lines += [f"## {split}", "", "| control | vs_gold CE | vs_gold acc | vs_gold macro-F1 | vs_control CE | vs_control acc | mean words | empty rate |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for control_type in CONTROL_TYPES:
            m = split_metrics[control_type]
            vg = m["vs_gold"]
            vc = m.get("vs_control", {})
            lines.append(
                f"| {control_type} | {vg.get('soft_ce', 0):.4f} | {vg.get('accuracy', 0):.4f} | {vg.get('macro_f1', 0):.4f} | "
                f"{vc.get('soft_ce', float('nan')):.4f} | {vc.get('accuracy', float('nan')):.4f} | "
                f"{m.get('mean_generated_words', 0):.2f} | {m.get('empty_rate', 0):.4f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gold/predicted/zero/shuffled controls and score generated replies with the stance predictor.")
    parser.add_argument("--prepared", required=True, help="Gold-control prepared directory.")
    parser.add_argument("--predicted-prepared", required=True, help="Prepared directory with predicted cluster-prototype target vectors.")
    parser.add_argument("--generator-dir", required=True)
    parser.add_argument("--stance-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="Override base generator model. Defaults to generator config model.")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--max-examples", type=int, default=64, help="Per split. 0 means full split.")
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
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prepared = Path(args.prepared)
    predicted_prepared = Path(args.predicted_prepared)

    lm, gen_tokenizer, projector, prefix_tokens, vector_dim, base_model = load_generator(Path(args.generator_dir), args.model, args.bf16, args.device)
    records_by_split: Dict[str, List[dict]] = {}
    selection_summary: Dict[str, Any] = {}
    for split in parse_splits(args.splits):
        gold_all = load_prepared_split(prepared, split)
        pred_all = load_prepared_split(predicted_prepared, split)
        gold_rows, indices = select_rows(gold_all, args.max_examples, args.seed, args.sample_strategy)
        pred_rows = aligned_predicted_rows(gold_rows, pred_all, indices)
        selection_summary[split] = {"available": len(gold_all), "selected": len(gold_rows), "indices": indices[:20]}
        records = generate_split(lm, gen_tokenizer, projector, split, gold_rows, pred_rows, vector_dim, args, prefix_tokens)
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
        "predicted_prepared": args.predicted_prepared,
        "controls": list(CONTROL_TYPES),
        "splits": parse_splits(args.splits),
        "max_examples": args.max_examples,
        "response_length_words": args.response_length_words,
        "selection": selection_summary,
        "metrics": metrics,
    }
    write_json(out / "metrics.json", summary)
    write_markdown_report(out / "report.md", metrics, args)


if __name__ == "__main__":
    main()
