from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import build_model_text, load_prepared_split, write_json, write_jsonl
from .generate_and_score_controls import (
    aligned_predicted_rows,
    decode_generated,
    load_generator,
    load_stance_scorer,
    prompt_text,
    response_length_instruction,
    score_records,
    select_rows,
    unload_generator,
)
from .metrics import kl_divergence, soft_cross_entropy


METHODS = (
    ("ours", "Ours: latent stance soft-prefix + rerank"),
    ("llm_only", "Mistral only"),
    ("llm_prompt", "Mistral + emotion-rich prompt"),
    ("gold_reference", "Gold reference"),
)

CHOICE_TASKS = (
    (
        "emotion_rich",
        "Task A: Choose the response with the richest and most natural emotional expression.",
        "positive",
    ),
    (
        "passive_ai",
        "Task B: Choose the response that feels most passive or most like a generic AI assistant reply.",
        "negative",
    ),
    (
        "continue_interest",
        "Task C: Choose the response that would make you most interested in continuing the conversation.",
        "positive",
    ),
    (
        "context_fit",
        "Task D: Choose the response that best fits the context and sounds most like a natural next turn.",
        "positive",
    ),
    (
        "felt_understood",
        "Task E: Choose the response that would make the speaker feel most understood.",
        "positive",
    ),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_distinct_dialogues(rows: List[dict], n: int, seed: int) -> Tuple[List[dict], List[int]]:
    by_dialogue: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        by_dialogue.setdefault(str(row.get("dialogue_id", idx)), []).append(idx)
    rng = random.Random(seed)
    dialogue_ids = list(by_dialogue)
    rng.shuffle(dialogue_ids)
    chosen: List[int] = []
    for dialogue_id in dialogue_ids[:n]:
        chosen.append(rng.choice(by_dialogue[dialogue_id]))
    chosen.sort()
    return [rows[i] for i in chosen], chosen


def word_count(text: Any) -> int:
    return len(str(text or "").split())


def clean_eval_generation(text: Any) -> str:
    value = str(text or "").strip()
    for prefix in ("Rewritten reply:", "Expanded reply:", "Shortened reply:", "Reply:", "A:", "B:"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    for marker in (
        "\n\nContext:",
        "\nContext:",
        "\n\nSituation:",
        "\nSituation:",
        "\nDialogue:",
        "\nCandidate reply:",
        "\nRewritten reply:",
        "\nExpanded reply:",
        "\nShortened reply:",
        "\nInstruction:",
        "\nRecommended reply:",
        "\nShared content plan:",
        "\nContent plan:",
        "\n---",
    ):
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    for marker in (
        " Recommended reply:",
        " Shared content plan:",
        " Content plan:",
        " ---",
    ):
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    value = re.sub(r"\s*\(\s*\d+\s+words?\s*\)\s*$", "", value, flags=re.IGNORECASE)
    return value.strip()


def plan_text(row: dict) -> str:
    return str(row.get("response_plan", "") or "").strip()


def plan_block(row: dict) -> str:
    plan = plan_text(row)
    if not plan:
        return ""
    return f"Shared content plan: {plan}\n"


def prompt_text_for_eval(row: dict, args, emotion_rich: bool = False) -> str:
    next_role = row.get("next_role", "B")
    shared_plan = plan_block(row)
    length_hint = response_length_instruction(args).strip()
    if emotion_rich:
        instruction = (
            "Instruction: Continue the dialogue as the next speaker. Write one natural, emotionally rich, "
            "empathetic reply that fits the context and the shared content plan. Do not mention labels, clusters, plans, or this instruction."
        )
    elif shared_plan or length_hint:
        instruction = (
            "Instruction: Continue the dialogue as the next speaker. Write one natural reply that fits the context "
            "and the shared content plan if provided. Do not mention labels, clusters, plans, or this instruction."
        )
    else:
        return prompt_text(row)
    if length_hint:
        instruction = instruction + " " + length_hint.removeprefix("Instruction: ").strip()
    plan_section = f"\n{shared_plan}" if shared_plan else ""
    return f"{instruction}{plan_section}\n{build_model_text(row)}\n{next_role}:"


def prompt_with_emotion_instruction(row: dict, args) -> str:
    return prompt_text_for_eval(row, args, emotion_rich=True)


def context_plan_prompt(row: dict, args) -> str:
    next_role = row.get("next_role", "B")
    return (
        "Instruction: Create a concise content plan for the next dialogue reply. Do not write the reply. "
        "Do not mention emotion labels, clusters, model names, or scoring. Keep it content-only: dialogue act, content focus, "
        "and whether to ask a follow-up question. Do not include any sample reply text. Return one short abstract plan sentence.\n\n"
        f"Context:\n{build_model_text(row)}\n{next_role}:\n\nContent plan:"
    )


def gold_plan_prompt(row: dict, args) -> str:
    next_role = row.get("next_role", "B")
    return (
        "Instruction: Extract the high-level content direction of the gold next reply. Do not copy the gold wording. "
        "Do not write the reply. Do not mention emotion labels, clusters, model names, or scoring. "
        "Return one short abstract plan sentence describing only the dialogue act, content focus, and follow-up question if any. Do not include any sample reply text.\n\n"
        f"Context:\n{build_model_text(row)}\n{next_role}:\n\n"
        f"Gold next reply:\n{row.get('response', '')}\n\nAbstract content plan:"
    )


def clean_response_plan(text: Any) -> str:
    value = clean_eval_generation(text)
    for prefix in ("Content plan:", "Abstract content plan:", "Plan:"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
    for marker in (" Response:", " Reply:", " Sample reply:", " Example reply:", " Final reply:"):
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    for marker in ("\nResponse:", "\nReply:", "\nSample reply:", "\nExample reply:", "\nFinal reply:"):
        if marker in value:
            value = value.split(marker, 1)[0].strip()
    value = re.sub(r"\s+", " ", value).strip()
    return value[:500]


@torch.no_grad()
def generate_response_plan(lm, tokenizer, row: dict, args) -> Tuple[str, str]:
    if args.direction_control == "none":
        return "", ""
    prompt = context_plan_prompt(row, args) if args.direction_control == "context_plan" else gold_plan_prompt(row, args)
    encoded = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.plan_max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.plan_do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": args.plan_temperature, "top_p": args.plan_top_p})
    else:
        gen_kwargs.update({"do_sample": False})
    output_ids = lm.generate(**gen_kwargs)
    generated = decode_generated(output_ids[0], input_ids[0], tokenizer, prompt, args.plan_max_new_tokens)
    return clean_response_plan(generated), prompt


def attach_response_plans(lm, tokenizer, split: str, rows: List[dict], args) -> Tuple[List[dict], List[dict]]:
    planned_rows: List[dict] = []
    plan_records: List[dict] = []
    total = len(rows)
    for idx, row in enumerate(rows):
        row_copy = dict(row)
        plan, prompt = generate_response_plan(lm, tokenizer, row_copy, args)
        row_copy["direction_control"] = args.direction_control
        row_copy["response_plan"] = plan
        planned_rows.append(row_copy)
        plan_records.append(
            {
                "split": split,
                "example_index": idx,
                "dialogue_id": row.get("dialogue_id"),
                "turn_id": row.get("turn_id"),
                "direction_control": args.direction_control,
                "response_plan": plan,
                "plan_prompt": prompt,
            }
        )
        if args.direction_control != "none" and args.progress_every and (idx + 1) % args.progress_every == 0:
            print(f"generated response plans: {idx + 1}/{total}", flush=True)
    return planned_rows, plan_records


@torch.no_grad()
def generate_plain_once(lm, tokenizer, prompt: str, args, force_sample: bool = False) -> str:
    encoded = tokenizer(prompt, truncation=True, max_length=args.max_prompt_length, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample or force_sample:
        gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        gen_kwargs.update({"do_sample": False})
    output_ids = lm.generate(**gen_kwargs)
    return clean_eval_generation(decode_generated(output_ids[0], input_ids[0], tokenizer, prompt, args.max_new_tokens))


def generate_plain(lm, tokenizer, prompt: str, args) -> Tuple[str, List[str], int]:
    generated = generate_plain_once(lm, tokenizer, prompt, args)
    return generated, [generated], 0


@torch.no_grad()
def generate_prefix_with_prompt(lm, tokenizer, projector, prompt: str, stance_vector: np.ndarray, args, force_sample: bool = False) -> str:
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
    if args.do_sample or force_sample:
        gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        gen_kwargs.update({"do_sample": False})
    output_ids = lm.generate(**gen_kwargs)
    return clean_eval_generation(decode_generated(output_ids[0], input_ids[0], tokenizer, prompt, args.max_new_tokens))


def base_record(split: str, example_index: int, row: dict, method: str, method_label: str, generated: str, prompt_value: Optional[str] = None) -> dict:
    return {
        "split": split,
        "example_index": example_index,
        "dialogue_id": row.get("dialogue_id"),
        "turn_id": row.get("turn_id"),
        "role": row.get("role"),
        "next_role": row.get("next_role"),
        "transition": row.get("transition"),
        "method": method,
        "method_label": method_label,
        "situation": row.get("situation", ""),
        "context": row.get("context", ""),
        "prompt": prompt_value if prompt_value is not None else prompt_text(row),
        "reference_response": row.get("response", ""),
        "direction_control": row.get("direction_control", "none"),
        "response_plan": row.get("response_plan", ""),
        "generated_response": generated,
        "generated_words": word_count(generated),
        "gold_target_cluster": row.get("target_cluster"),
        "gold_target_top1": int(np.argmax(row.get("target_cluster"))),
    }


def attach_stance_scores(records: List[dict], stance_model, stance_tokenizer, args) -> None:
    pred = score_records(records, stance_model, stance_tokenizer, args)
    for record, q in zip(records, pred):
        gold = np.asarray(record["gold_target_cluster"], dtype=np.float64)
        record["generated_source_cluster_pred"] = q.astype(float).tolist()
        record["generated_source_top1"] = int(np.argmax(q))
        record["generated_vs_gold_soft_ce"] = soft_cross_entropy(gold, q)
        record["generated_vs_gold_kl"] = kl_divergence(gold, q)
        if record.get("control_target_cluster") is not None:
            control = np.asarray(record["control_target_cluster"], dtype=np.float64)
            record["generated_vs_control_soft_ce"] = soft_cross_entropy(control, q)
            record["generated_vs_control_kl"] = kl_divergence(control, q)


def generate_ours_candidates(lm, tokenizer, projector, split: str, rows: List[dict], pred_rows: List[dict], args, prefix_tokens: int) -> List[dict]:
    candidates: List[dict] = []
    total = len(rows) * args.num_candidates
    done = 0
    for example_index, (row, pred_row) in enumerate(zip(rows, pred_rows)):
        vector = np.asarray(pred_row["target_vector"], dtype=np.float32)
        control_cluster = np.asarray(pred_row["target_cluster"], dtype=np.float64).astype(float).tolist()
        prompt = prompt_text_for_eval(row, args, emotion_rich=False)
        for candidate_index in range(args.num_candidates):
            raw_generated = generate_prefix_with_prompt(
                lm,
                tokenizer,
                projector,
                prompt,
                vector,
                args,
                force_sample=(candidate_index > 0 and args.retry_with_sampling),
            )
            generated = raw_generated
            record = base_record(split, example_index, row, "ours_candidate", "Ours candidate", generated, prompt_value=prompt)
            record.update(
                {
                    "candidate_index": candidate_index,
                    "raw_generated_response": raw_generated,
                    "raw_generated_words": word_count(raw_generated),
                    "control_target_cluster": control_cluster,
                    "control_target_top1": int(np.argmax(control_cluster)),
                    "control_source_dialogue_id": pred_row.get("dialogue_id"),
                    "control_source_turn_id": pred_row.get("turn_id"),
                }
            )
            candidates.append(record)
            done += 1
            if args.progress_every and done % args.progress_every == 0:
                print(f"generated ours candidates: {done}/{total}", flush=True)
    return candidates


def select_ours(candidates: List[dict], args) -> List[dict]:
    by_example: Dict[int, List[dict]] = {}
    for row in candidates:
        by_example.setdefault(int(row["example_index"]), []).append(row)
    selected: List[dict] = []
    for example_index in sorted(by_example):
        group = sorted(by_example[example_index], key=lambda item: int(item["candidate_index"]))
        best = min(
            group,
            key=lambda item: (
                float(item["generated_vs_control_soft_ce"]),
                int(item["candidate_index"]),
            ),
        )
        row = dict(best)
        row["method"] = "ours"
        row["method_label"] = "Ours: latent stance soft-prefix + rerank"
        row["selected_candidate_index"] = int(best["candidate_index"])
        row["candidate_count"] = len(group)
        selected.append(row)
    return selected


def generate_baselines(lm, tokenizer, split: str, rows: List[dict], args) -> List[dict]:
    records: List[dict] = []
    total = len(rows) * 2
    done = 0
    for example_index, row in enumerate(rows):
        plain_prompt = prompt_text_for_eval(row, args, emotion_rich=False)
        raw_generated, attempts, chosen_idx = generate_plain(lm, tokenizer, plain_prompt, args)
        generated = raw_generated
        record = base_record(split, example_index, row, "llm_only", "Mistral only", generated, prompt_value=plain_prompt)
        record.update(
            {
                "raw_generated_response": raw_generated,
                "raw_generated_words": word_count(raw_generated),
                "generation_attempts": attempts,
                "selected_generation_attempt": chosen_idx,
            }
        )
        records.append(record)
        done += 1
        if args.progress_every and done % args.progress_every == 0:
            print(f"generated baselines: {done}/{total}", flush=True)

        emotion_prompt = prompt_with_emotion_instruction(row, args)
        raw_generated, attempts, chosen_idx = generate_plain(lm, tokenizer, emotion_prompt, args)
        generated = raw_generated
        record = base_record(split, example_index, row, "llm_prompt", "Mistral + emotion-rich prompt", generated, prompt_value=emotion_prompt)
        record.update(
            {
                "raw_generated_response": raw_generated,
                "raw_generated_words": word_count(raw_generated),
                "generation_attempts": attempts,
                "selected_generation_attempt": chosen_idx,
            }
        )
        records.append(record)
        done += 1
        if args.progress_every and done % args.progress_every == 0:
            print(f"generated baselines: {done}/{total}", flush=True)
    return records


def make_gold_records(split: str, rows: List[dict]) -> List[dict]:
    records: List[dict] = []
    for example_index, row in enumerate(rows):
        record = base_record(
            split,
            example_index,
            row,
            "gold_reference",
            "Gold reference",
            str(row.get("response", "")),
            prompt_value="",
        )
        record["is_gold_reference"] = True
        records.append(record)
    return records


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def option_label(index: int) -> str:
    return chr(ord("A") + index)


def response_card(record: dict, label: str, reveal_method: bool = False) -> str:
    title = f"Response {label}"
    if reveal_method:
        title += f" · {record.get('method_label', record.get('method'))}"
    response = esc(record.get("generated_response", ""))
    meta_parts = [f"words: {word_count(record.get('generated_response', ''))}"]
    return f"""
      <article class="option-card" data-option="{esc(label)}">
        <h4>{esc(title)}</h4>
        <p class="response">{response}</p>
        <p class="meta">{esc(" | ".join(meta_parts))}</p>
      </article>
    """


def choice_task_block(example_index: int, options: List[dict]) -> str:
    blocks: List[str] = []
    for task_id, question, polarity in CHOICE_TASKS:
        option_radios = "\n".join(
            f'<label><input type="radio" name="choice-{example_index}-{esc(task_id)}" value="{esc(option["label"])}"> Response {esc(option["label"])}</label>'
            for option in options
        )
        blocks.append(
            f"""
            <div class="choice-task" data-task-id="{esc(task_id)}" data-polarity="{esc(polarity)}">
              <p>{esc(question)}</p>
              <div class="choice-row">{option_radios}</div>
            </div>
            """
        )
    return "\n".join(blocks)


def build_blind_options(example_index: int, method_records: Dict[str, dict], args) -> List[dict]:
    ordered = [(key, method_records[key]) for key, _label in METHODS if key in method_records]
    if getattr(args, "blind_labels", True):
        rng = random.Random(f"{args.seed}:blind:{example_index}")
        rng.shuffle(ordered)
    options: List[dict] = []
    for index, (method, record) in enumerate(ordered):
        options.append(
            {
                "label": option_label(index),
                "method": method,
                "method_label": record.get("method_label", method),
                "record": record,
            }
        )
    return options


def write_html(path: Path, examples: List[dict], records: List[dict], args) -> None:
    by_example: Dict[int, List[dict]] = {}
    for record in records:
        by_example.setdefault(int(record["example_index"]), []).append(record)

    sections: List[str] = []
    blind_key_rows: List[dict] = []
    for idx, row in enumerate(examples):
        method_records = {record["method"]: record for record in by_example.get(idx, [])}
        options = build_blind_options(idx, method_records, args)
        for option in options:
            blind_key_rows.append(
                {
                    "example_index": idx,
                    "dialogue_id": row.get("dialogue_id"),
                    "turn_id": row.get("turn_id"),
                    "option": option["label"],
                    "method": option["method"],
                    "method_label": option["method_label"],
                    "direction_control": option["record"].get("direction_control", "none"),
                    "response_plan": option["record"].get("response_plan", ""),
                }
            )
        cards = "\n".join(
            response_card(option["record"], option["label"], reveal_method=not getattr(args, "blind_labels", True))
            for option in options
        )
        tasks = choice_task_block(idx, options)
        sections.append(
            f"""
            <section class="example" id="example-{idx + 1}" data-example-index="{idx}">
              <header>
                <div>
                  <h3>Example {idx + 1}</h3>
                  <p class="meta">dialogue_id: {esc(row.get('dialogue_id'))} | turn: {esc(row.get('turn_id'))} | transition: {esc(row.get('transition'))}</p>
                </div>
              </header>
              <div class="context-block">
                <h4>Context</h4>
                <pre>{esc(row.get('context', ''))}</pre>
              </div>
              {f'<details class="plan-block"><summary>Shared content plan</summary><p>{esc(row.get("response_plan", ""))}</p></details>' if getattr(args, "show_plan", False) and row.get("response_plan") else ''}
              <div class="grid">
                {cards}
              </div>
              <div class="tasks">
                <h4>Single-choice Tasks</h4>
                {tasks}
                <textarea class="example-note" placeholder="Optional note: which responses were hard to distinguish, sounded generic, or made you want to continue"></textarea>
              </div>
            </section>
            """
        )

    task_payload = [
        {"id": task_id, "question": question, "polarity": polarity}
        for task_id, question, polarity in CHOICE_TASKS
    ]
    payload = json.dumps(
        {
            "created_by": "latent_stance_control.make_human_eval_html",
            "seed": args.seed,
            "num_examples": len(examples),
            "blind_labels": args.blind_labels,
            "methods": [{"id": key, "label": label} for key, label in METHODS],
            "tasks": task_payload,
        },
        ensure_ascii=False,
        indent=2,
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Latent Stance Choice Evaluation</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #666b73;
      --line: #d9d9d2;
      --accent: #245f73;
      --soft: #edf4f5;
      --choice: #f6f1e6;
    }}
    body {{
      margin: 0;
      font: 15px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 22px 48px; }}
    h1, h2, h3, h4 {{ margin: 0; line-height: 1.2; }}
    .intro, .example {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .intro {{ padding: 18px 20px; margin-bottom: 20px; }}
    .intro p {{ margin: 8px 0 0; color: var(--muted); }}
    .example {{ padding: 18px; margin: 18px 0; }}
    .example header {{ display: flex; justify-content: space-between; gap: 16px; margin-bottom: 14px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin: 6px 0 0; }}
    .context-block {{ border-left: 4px solid var(--accent); background: var(--soft); padding: 12px 14px; margin-bottom: 12px; }}
    pre {{ white-space: pre-wrap; margin: 8px 0 0; font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .option-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; min-height: 210px; display: flex; flex-direction: column; gap: 10px; }}
    .option-card h4 {{ color: var(--accent); font-size: 15px; }}
    .response {{ font-size: 16px; margin: 0; flex: 1; }}
    .tasks {{ margin-top: 14px; border-top: 1px solid var(--line); padding-top: 14px; }}
    .choice-task {{ background: var(--choice); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; margin: 10px 0; }}
    .choice-task p {{ margin: 0 0 8px; font-weight: 600; }}
    .choice-row {{ display: flex; flex-wrap: wrap; gap: 14px; }}
    .choice-row label {{ white-space: nowrap; }}
    textarea {{ width: 100%; min-height: 58px; box-sizing: border-box; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font: inherit; }}
    .toolbar {{ position: sticky; top: 0; z-index: 2; background: rgba(247,247,244,.96); border-bottom: 1px solid var(--line); padding: 10px 0; margin-bottom: 16px; }}
    button {{ border: 1px solid var(--accent); background: var(--accent); color: #fff; border-radius: 6px; padding: 8px 12px; cursor: pointer; font: inherit; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} .option-card {{ min-height: auto; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="intro">
      <h1>Latent Stance Control Four-choice Human Evaluation</h1>
      <p>Each example contains four anonymous responses to the same context: Response A/B/C/D. Do not rate every response separately; for each task, choose the single response that best matches the prompt.</p>
      <p>The page shows anonymous options only; the true method mapping is saved next to the page as blind_key.json and blind_key.jsonl.</p>
      <p>Direction-control setting: {esc(args.direction_control)}. context_plan means the shared response plan is generated from context only; gold_plan means the high-level direction is extracted from the gold response and should be used for diagnostic experiments only.</p>
      <p>Task B is a negative metric: more selections mean the method is more likely to sound passive or generic. The other tasks are positive metrics.</p>
    </section>
    <div class="toolbar">
      <button onclick="exportChoices()">Export Choices JSON</button>
    </div>
    {"".join(sections)}
  </div>
  <script type="application/json" id="metadata">{html.escape(payload)}</script>
  <script>
    function exportChoices() {{
      const data = [];
      document.querySelectorAll('.example').forEach((example, idx) => {{
        const item = {{ example_index: Number(example.dataset.exampleIndex), tasks: [], note: '' }};
        example.querySelectorAll('.choice-task').forEach(task => {{
          const checked = task.querySelector('input[type="radio"]:checked');
          item.tasks.push({{
            task_id: task.dataset.taskId,
            polarity: task.dataset.polarity,
            selected_option: checked ? checked.value : null
          }});
        }});
        const note = example.querySelector('.example-note');
        item.note = note ? note.value : '';
        data.push(item);
      }});
      const blob = new Blob([JSON.stringify(data, null, 2)], {{ type: 'application/json' }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'human_eval_choices.json';
      a.click();
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    write_json(path.parent / "blind_key.json", {"tasks": task_payload, "options": blind_key_rows})
    write_jsonl(path.parent / "blind_key.jsonl", blind_key_rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Sample test examples, generate three systems, and write an offline HTML page for human evaluation.")
    parser.add_argument("--prepared", default="runs/main/prepared")
    parser.add_argument("--predicted-prepared", default="runs/main/prepared_predicted_control")
    parser.add_argument("--generator-dir", default="runs/main/generator")
    parser.add_argument("--stance-dir", default="runs/main/stance")
    parser.add_argument("--out", default="runs/main/human_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-examples", type=int, default=20)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-strategy", choices=["distinct_dialogues", "random", "first"], default="distinct_dialogues")
    parser.add_argument("--model", default=None, help="Override base generator model.")
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--response-length-words", type=int, default=0, help="Optional evaluation-time target reply length in words. 0 disables the prompt hint.")
    parser.add_argument("--retry-with-sampling", action=argparse.BooleanOptionalAction, default=True, help="Use sampling for additional generated candidates.")
    parser.add_argument("--direction-control", choices=["none", "context_plan", "gold_plan"], default="none", help="Use a shared content plan to align reply direction. context_plan is the main non-gold setting; gold_plan is diagnostic.")
    parser.add_argument("--plan-max-new-tokens", type=int, default=96)
    parser.add_argument("--plan-do-sample", action="store_true", help="Sample response plans. Default is greedy for stable direction control.")
    parser.add_argument("--plan-temperature", type=float, default=0.2)
    parser.add_argument("--plan-top-p", type=float, default=0.9)
    parser.add_argument("--show-plan", action=argparse.BooleanOptionalAction, default=False, help="Show the shared content plan in the HTML page. Default hides it from raters.")
    parser.add_argument("--blind-labels", action=argparse.BooleanOptionalAction, default=True, help="Show anonymous Response A/B/C/D labels and write the method mapping to blind_key files.")
    parser.add_argument("--do-sample", action="store_true", help="Use sampling. Default is greedy for reproducible review pages.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--score-max-length", type=int, default=384)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = load_prepared_split(args.prepared, args.split)
    pred_all = load_prepared_split(args.predicted_prepared, args.split)
    if args.sample_strategy == "distinct_dialogues":
        rows, indices = sample_distinct_dialogues(all_rows, args.num_examples, args.seed)
    else:
        rows, indices = select_rows(all_rows, args.num_examples, args.seed, "random" if args.sample_strategy == "random" else "first")
    pred_rows = aligned_predicted_rows(rows, pred_all, indices)

    lm, tokenizer, projector, prefix_tokens, _vector_dim, base_model = load_generator(Path(args.generator_dir), args.model, args.bf16, args.device)
    rows, response_plan_records = attach_response_plans(lm, tokenizer, args.split, rows, args)
    ours_candidates = generate_ours_candidates(lm, tokenizer, projector, args.split, rows, pred_rows, args, prefix_tokens)
    baseline_records = generate_baselines(lm, tokenizer, args.split, rows, args)
    unload_generator(lm, projector)

    stance_model, stance_tokenizer = load_stance_scorer(Path(args.stance_dir), Path(args.prepared), args.device)
    attach_stance_scores(ours_candidates, stance_model, stance_tokenizer, args)
    ours_selected = select_ours(ours_candidates, args)
    attach_stance_scores(baseline_records, stance_model, stance_tokenizer, args)
    gold_records = make_gold_records(args.split, rows)
    attach_stance_scores(gold_records, stance_model, stance_tokenizer, args)

    records = []
    for example_index in range(len(rows)):
        records.extend([row for row in ours_selected if int(row["example_index"]) == example_index])
        records.extend([row for row in baseline_records if int(row["example_index"]) == example_index])
        records.extend([row for row in gold_records if int(row["example_index"]) == example_index])

    write_jsonl(out / "sampled_examples.jsonl", rows)
    write_jsonl(out / "response_plans.jsonl", response_plan_records)
    write_jsonl(out / "generations.jsonl", records)
    write_jsonl(out / "ours_candidates.scored.jsonl", ours_candidates)
    write_json(
        out / "config.json",
        {
            "prepared": args.prepared,
            "predicted_prepared": args.predicted_prepared,
            "generator_dir": args.generator_dir,
            "base_model": base_model,
            "stance_dir": args.stance_dir,
            "split": args.split,
            "num_examples": len(rows),
            "num_candidates": args.num_candidates,
            "response_length_words": args.response_length_words,
            "retry_with_sampling": args.retry_with_sampling,
            "blind_labels": args.blind_labels,
            "direction_control": args.direction_control,
            "plan_max_new_tokens": args.plan_max_new_tokens,
            "plan_do_sample": args.plan_do_sample,
            "plan_temperature": args.plan_temperature,
            "plan_top_p": args.plan_top_p,
            "show_plan": args.show_plan,
            "choice_tasks": [
                {"id": task_id, "question": question, "polarity": polarity}
                for task_id, question, polarity in CHOICE_TASKS
            ],
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "indices": indices,
            "methods": [{"id": key, "label": label} for key, label in METHODS],
        },
    )
    write_html(out / "index.html", rows, records, args)
    print(f"Wrote {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
