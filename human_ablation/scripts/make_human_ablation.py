#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


QUESTION = (
    "哪个回复更符合当前对话语境，并且更能让上一轮发言者觉得自己的内容、感受或处境被认真回应和理解？"
)

COMPARISONS = [
    {
        "comparison": "final_vs_no_rerank",
        "ablation_name": "no_rerank",
        "label": "final_emoh vs no_rerank",
    },
    {
        "comparison": "final_vs_no_role_aware",
        "ablation_name": "no_role_aware",
        "label": "final_emoh vs no_role_aware",
    },
    {
        "comparison": "final_vs_zero_control",
        "ablation_name": "zero_control",
        "label": "final_emoh vs zero_control",
    },
]

FORBIDDEN_BLIND_PATTERNS = [
    "final_emoh",
    "no_rerank",
    "no_role_aware",
    "zero_control",
    "role-aware",
    "rerank",
    "zero control",
    "stance vector",
    "stance cluster",
    "emoji label",
    "gold target",
    "oracle",
]

STYLE = """
:root {
  color-scheme: light;
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #20242c;
  --muted: #657082;
  --line: #d8dee8;
  --soft: #f0f4f8;
  --accent: #1f5f7a;
  --accent-2: #22543d;
  --danger: #9a3412;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.55 Arial, Helvetica, sans-serif;
}
.app { max-width: 980px; margin: 0 auto; padding: 24px; }
.top, .card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
h1 { font-size: 23px; margin: 0 0 10px; }
h2 { font-size: 18px; margin: 0 0 10px; }
h3 { color: #41506a; font-size: 14px; margin: 18px 0 8px; }
p { margin: 8px 0; }
.muted { color: var(--muted); }
.small { font-size: 13px; }
.question {
  font-size: 20px;
  font-weight: 700;
  line-height: 1.45;
  margin-bottom: 14px;
}
.box {
  background: var(--soft);
  border: 1px solid #e1e7ef;
  border-radius: 8px;
  padding: 12px 14px;
  white-space: pre-wrap;
}
.responses {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
.response-title { font-weight: 700; margin-bottom: 8px; }
.choices {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 18px;
}
.choice {
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid #bac7d8;
  background: #fff;
  border-radius: 8px;
  padding: 10px 12px;
  cursor: pointer;
}
.choice:hover { background: #edf4fb; }
.choice input { margin: 0; }
button {
  border: 1px solid #bac7d8;
  background: #fff;
  color: var(--ink);
  border-radius: 8px;
  padding: 10px 12px;
  font: inherit;
  cursor: pointer;
}
button:hover { background: #edf4fb; }
button.primary { background: var(--accent); border-color: var(--accent); color: white; }
button.secondary { background: #f8fafc; }
button:disabled { opacity: .5; cursor: not-allowed; }
textarea {
  width: 100%;
  min-height: 72px;
  resize: vertical;
  border: 1px solid #bac7d8;
  border-radius: 8px;
  padding: 10px 12px;
  font: inherit;
  background: #fff;
}
.nav { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-top: 16px; }
.nav-group { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.progressbar { height: 10px; background: #e5ebf2; border-radius: 999px; overflow: hidden; margin-top: 12px; }
.progressbar > div { height: 100%; width: 0%; background: var(--accent-2); }
.warning { color: var(--danger); font-weight: 700; }
.ok { color: var(--accent-2); font-weight: 700; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid var(--line); padding: 8px 10px; vertical-align: top; }
th { background: var(--soft); text-align: left; }
@media (max-width: 760px) {
  .app { padding: 12px; }
  .responses, .choices { grid-template-columns: 1fr; }
  .nav { align-items: stretch; flex-direction: column; }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build blind human-evaluation HTML files for generation-control ablations."
    )
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--output-dir", default="human_ablation", help="Output directory.")
    parser.add_argument("--num-contexts", type=int, default=50)
    parser.add_argument("--num-annotators", type=int, default=10)
    parser.add_argument("--annotators-per-item", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--reuse-existing-template", action="store_true")
    parser.add_argument("--prefer-existing-outputs", action="store_true")
    parser.add_argument("--final-output-path", default=None)
    parser.add_argument("--no-rerank-output-path", default=None)
    parser.add_argument("--no-role-aware-output-path", default=None)
    parser.add_argument("--zero-control-output-path", default=None)
    parser.add_argument("--existing-human-template-dir", default=None)
    parser.add_argument("--exclude-contexts-path", default=None)
    parser.add_argument(
        "--min-final-words",
        type=int,
        default=15,
        help="Minimum word count required for final_emoh responses used in the annotation set.",
    )
    parser.add_argument(
        "--require-final-complete-sentence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require final_emoh responses to end like a complete sentence.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_json_or_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("items", "records", "data", "examples", "generations"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            return [obj]
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    raise ValueError(f"Unsupported input format: {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_context(value: Any) -> str:
    text = str(value or "").replace("<s>", "").replace("</s>", "").replace("[INST]", "").replace("[/INST]", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def example_key(row: dict[str, Any]) -> str:
    for key in ("example_index", "example_id", "id", "sample_id", "turn_id"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    dialogue_id = row.get("dialogue_id") or row.get("context_id") or ""
    turn_id = row.get("turn_id") or ""
    return f"{dialogue_id}_{turn_id}"


def response_text(row: dict[str, Any]) -> str:
    for key in (
        "generated_response",
        "selected_response",
        "response",
        "generation",
        "output",
        "text",
        "final_response",
    ):
        if key in row and norm_text(row[key]):
            return norm_text(row[key])
    return ""


def row_dialogue_id(row: dict[str, Any]) -> str:
    if row.get("dialogue_id"):
        return str(row["dialogue_id"])
    context_id = str(row.get("context_id") or "")
    match = re.search(r"(hit:[^_]+_conv:[^_]+)", context_id)
    if match:
        return match.group(1)
    return context_id


def context_uid(row: dict[str, Any]) -> str:
    dialogue_id = row_dialogue_id(row)
    turn_id = row.get("turn_id")
    if dialogue_id and turn_id not in (None, ""):
        return f"{dialogue_id}_turn_{turn_id}"
    if dialogue_id:
        return dialogue_id
    return f"example_{example_key(row)}"


def default_paths(root: Path, args: argparse.Namespace) -> dict[str, Path]:
    rerank_path = root / "runs/main/rerank_c7mix050_512_seed13/selected_test.jsonl"
    control_path = root / "runs/main/generator_control_eval_512_seed13/generations_test.scored.jsonl"
    exclude_path = root / "runs/human_eval_other_methods_main/eval_items_master.jsonl"
    return {
        "final": Path(args.final_output_path) if args.final_output_path else rerank_path,
        "no_rerank": Path(args.no_rerank_output_path) if args.no_rerank_output_path else rerank_path,
        "no_role_aware": Path(args.no_role_aware_output_path) if args.no_role_aware_output_path else control_path,
        "zero_control": Path(args.zero_control_output_path) if args.zero_control_output_path else control_path,
        "exclude": Path(args.exclude_contexts_path) if args.exclude_contexts_path else exclude_path,
    }


def load_excluded_dialogues(path: Path) -> tuple[set[str], str]:
    if not path.exists():
        return set(), f"No exclusion file found at {path}; no main-human-eval contexts were excluded."
    excluded: set[str] = set()
    for row in read_json_or_table(path):
        did = row_dialogue_id(row)
        if did:
            excluded.add(did)
    return excluded, f"Excluded {len(excluded)} dialogue ids from {path} when possible."


def load_project_outputs(paths: dict[str, Path]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    missing: list[str] = []
    for name in ("final", "no_rerank", "no_role_aware", "zero_control"):
        if not paths[name].exists():
            missing.append(f"{name}: missing file {paths[name]}")
    if missing:
        return {}, missing

    final_rows = read_json_or_table(paths["final"])
    no_rerank_rows = read_json_or_table(paths["no_rerank"])
    no_role_rows = read_json_or_table(paths["no_role_aware"])
    zero_rows = read_json_or_table(paths["zero_control"])

    final_by_key: dict[str, dict[str, Any]] = {}
    no_rerank_by_key: dict[str, dict[str, Any]] = {}
    no_role_by_key: dict[str, dict[str, Any]] = {}
    zero_by_key: dict[str, dict[str, Any]] = {}

    for row in final_rows:
        if str(row.get("selection_type", "")).lower() == "rerank_control":
            final_by_key[example_key(row)] = row
    for row in no_rerank_rows:
        if str(row.get("selection_type", "")).lower() == "raw_first":
            no_rerank_by_key[example_key(row)] = row
    for row in no_role_rows:
        if str(row.get("control_type", "")).lower() == "predicted":
            no_role_by_key[example_key(row)] = row
    for row in zero_rows:
        if str(row.get("control_type", "")).lower() == "zero":
            zero_by_key[example_key(row)] = row

    for label, rows in (
        ("final_emoh/rerank_control", final_by_key),
        ("no_rerank/raw_first", no_rerank_by_key),
        ("no_role_aware/predicted", no_role_by_key),
        ("zero_control/zero", zero_by_key),
    ):
        if not rows:
            missing.append(f"{label}: no matching records found in configured file")

    records: dict[str, dict[str, Any]] = {}
    common = sorted(set(final_by_key) & set(no_rerank_by_key) & set(no_role_by_key) & set(zero_by_key), key=int_or_text)
    for key in common:
        final_row = final_by_key[key]
        record = {
            "example_id": key,
            "context_uid": context_uid(final_row),
            "dialogue_id": row_dialogue_id(final_row),
            "turn_id": final_row.get("turn_id"),
            "situation": clean_context(final_row.get("situation", "")),
            "dialogue_history": clean_context(final_row.get("context", "")),
            "final_emoh": response_text(final_row),
            "no_rerank": response_text(no_rerank_by_key[key]),
            "no_role_aware": response_text(no_role_by_key[key]),
            "zero_control": response_text(zero_by_key[key]),
            "metadata": {
                "split": final_row.get("split"),
                "source_role": final_row.get("role"),
                "target_role": final_row.get("next_role"),
                "transition": final_row.get("transition"),
                "gold_target_top1": final_row.get("gold_target_top1"),
                "final_candidate_index": final_row.get("selected_candidate_index", final_row.get("candidate_index")),
                "candidate_count": final_row.get("candidate_count"),
                "final_source_path": str(paths["final"]),
                "no_rerank_source_path": str(paths["no_rerank"]),
                "no_role_aware_source_path": str(paths["no_role_aware"]),
                "zero_control_source_path": str(paths["zero_control"]),
            },
        }
        records[key] = record
    return records, missing


def int_or_text(value: Any) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(text):012d}")
    except ValueError:
        return (1, text)


def looks_complete_sentence(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(re.search(r"[.!?][\"')\]]*\s*$", stripped))


def has_valid_pairs(
    record: dict[str, Any],
    min_final_words: int = 0,
    require_final_complete_sentence: bool = False,
) -> bool:
    if not norm_text(record.get("dialogue_history")):
        return False
    final = norm_text(record.get("final_emoh"))
    if not final:
        return False
    if word_count(final) < min_final_words:
        return False
    if require_final_complete_sentence and not looks_complete_sentence(final):
        return False
    for name in ("no_rerank", "no_role_aware", "zero_control"):
        ablation = norm_text(record.get(name))
        if not ablation or ablation == final:
            return False
    return True


def select_contexts(
    records: dict[str, dict[str, Any]],
    excluded_dialogues: set[str],
    num_contexts: int,
    seed: int,
    min_final_words: int,
    require_final_complete_sentence: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    all_valid = [
        r
        for r in records.values()
        if has_valid_pairs(r, min_final_words, require_final_complete_sentence)
    ]
    no_overlap = [r for r in all_valid if r.get("dialogue_id") not in excluded_dialogues]
    pool = no_overlap if len(no_overlap) >= num_contexts else all_valid
    pool_mode = "excluded_main_eval_dialogues" if pool is no_overlap else "used_overlap_because_pool_was_too_small"
    if len(pool) < num_contexts:
        raise RuntimeError(f"Only {len(pool)} valid contexts available; need {num_contexts}.")

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        key = str(row.get("metadata", {}).get("gold_target_top1", "unknown"))
        strata[key].append(row)
    for rows in strata.values():
        rng.shuffle(rows)
    stratum_keys = list(strata.keys())
    rng.shuffle(stratum_keys)

    selected: list[dict[str, Any]] = []
    while len(selected) < num_contexts:
        progressed = False
        for key in list(stratum_keys):
            if strata[key]:
                selected.append(strata[key].pop())
                progressed = True
                if len(selected) == num_contexts:
                    break
        if not progressed:
            break
    rng.shuffle(selected)
    summary = {
        "available_common_records": len(records),
        "valid_nonidentical_records": len(all_valid),
        "valid_after_main_eval_exclusion": len(no_overlap),
        "selection_pool_mode": pool_mode,
        "selected_contexts": len(selected),
        "strata_counts": dict(Counter(str(r.get("metadata", {}).get("gold_target_top1", "unknown")) for r in selected)),
        "min_final_words": min_final_words,
        "require_final_complete_sentence": require_final_complete_sentence,
    }
    return selected, summary


def build_unassigned_items(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counter = 1
    for cidx, record in enumerate(selected, 1):
        for comp in COMPARISONS:
            item_uid = f"habl_item_{counter:04d}"
            ablation_name = comp["ablation_name"]
            rows.append(
                {
                    "item_uid": item_uid,
                    "context_uid": record["context_uid"],
                    "example_id": record["example_id"],
                    "dialogue_id": record.get("dialogue_id"),
                    "turn_id": record.get("turn_id"),
                    "comparison": comp["comparison"],
                    "question": QUESTION,
                    "situation": record.get("situation", ""),
                    "dialogue_history": record.get("dialogue_history", ""),
                    "response_final_emoh": record["final_emoh"],
                    "response_ablation": record[ablation_name],
                    "ablation_name": ablation_name,
                    "metadata": {
                        **record.get("metadata", {}),
                        "context_order": cidx,
                        "final_word_count": word_count(record["final_emoh"]),
                        "ablation_word_count": word_count(record[ablation_name]),
                    },
                }
            )
            counter += 1
    return rows


def assign_items(
    items: list[dict[str, Any]],
    num_annotators: int,
    annotators_per_item: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 17)
    annotators = [f"annotator_{i:02d}" for i in range(1, num_annotators + 1)]
    items_by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        items_by_context[item["context_uid"]].append(item)

    contexts = sorted(items_by_context.keys())
    annotator_comp_counts: dict[str, Counter[str]] = {aid: Counter() for aid in annotators}
    annotator_total: Counter[str] = Counter()
    assignments: list[dict[str, Any]] = []

    for cidx, context in enumerate(contexts):
        held_out = annotators[cidx % num_annotators]
        available = [aid for aid in annotators if aid != held_out]
        rng.shuffle(available)
        remaining = set(available)
        context_items = sorted(items_by_context[context], key=lambda x: x["comparison"])
        rng.shuffle(context_items)
        for item in context_items:
            comparison = item["comparison"]
            ranked = sorted(
                remaining,
                key=lambda aid: (
                    annotator_comp_counts[aid][comparison],
                    annotator_total[aid],
                    rng.random(),
                ),
            )
            chosen = ranked[:annotators_per_item]
            for aid in chosen:
                remaining.remove(aid)
                annotator_comp_counts[aid][comparison] += 1
                annotator_total[aid] += 1
                assignments.append(
                    {
                        "assignment_id": f"habl_assign_{len(assignments) + 1:04d}",
                        "annotator_id": aid,
                        "item_uid": item["item_uid"],
                        "context_uid": item["context_uid"],
                        "example_id": item["example_id"],
                        "dialogue_id": item.get("dialogue_id"),
                        "turn_id": item.get("turn_id"),
                        "comparison": comparison,
                        "question": item["question"],
                        "situation": item.get("situation", ""),
                        "dialogue_history": item.get("dialogue_history", ""),
                        "response_final_emoh": item["response_final_emoh"],
                        "response_ablation": item["response_ablation"],
                        "ablation_name": item["ablation_name"],
                        "held_out_annotator_for_context": held_out,
                        "seed": seed,
                    }
                )

    balance_ab(assignments, seed + 29)
    return sorted(assignments, key=lambda x: x["assignment_id"])


def balance_ab(assignments: list[dict[str, Any]], seed: int) -> None:
    rng = random.Random(seed)
    by_comp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_comp[row["comparison"]].append(row)
    for comparison, rows in by_comp.items():
        rng.shuffle(rows)
        target_a = len(rows) // 2
        for idx, row in enumerate(rows):
            final_in_a = idx < target_a
            if final_in_a:
                row["response_a"] = row["response_final_emoh"]
                row["response_b"] = row["response_ablation"]
                row["a_is"] = "final_emoh"
                row["b_is"] = row["ablation_name"]
                row["final_position"] = "A"
                row["ablation_position"] = "B"
            else:
                row["response_a"] = row["response_ablation"]
                row["response_b"] = row["response_final_emoh"]
                row["a_is"] = row["ablation_name"]
                row["b_is"] = "final_emoh"
                row["final_position"] = "B"
                row["ablation_position"] = "A"


def blind_assignment(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "assignment_id": row["assignment_id"],
        "item_uid": row["item_uid"],
        "question": row["question"],
        "situation": row.get("situation", ""),
        "dialogue_history": row.get("dialogue_history", ""),
        "response_a": row["response_a"],
        "response_b": row["response_b"],
    }


def render_annotator_html(path: Path, annotator_id: str, assignments: list[dict[str, Any]]) -> None:
    blind_items = [blind_assignment(row) for row in assignments]
    data_json = json.dumps(blind_items, ensure_ascii=False)
    storage_key = f"human_ablation_{annotator_id}_answers"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human Evaluation: Dialogue Response Comparison</title>
  <style>{STYLE}</style>
</head>
<body>
<main class="app">
  <section class="top">
    <h1>Human Evaluation: Dialogue Response Comparison</h1>
    <p><strong>Instructions:</strong></p>
    <p>You will see a dialogue context and two anonymized responses.</p>
    <p>Please choose which response better fits the dialogue context and would make the previous speaker feel more seriously responded to or understood.</p>
    <p>Do not try to infer which system produced each response.</p>
    <p>If both responses are equally good, choose Tie.</p>
    <p>If both responses are bad or neither fits the context, choose Neither / both are bad.</p>
    <p class="small muted">Please evaluate the responses only for research annotation. Do not share the examples outside the study.</p>
    <p class="small muted">Annotator ID: <span id="annotator-id"></span></p>
    <div class="progressbar"><div id="progress-bar"></div></div>
    <p class="small" id="progress-text"></p>
  </section>
  <section class="card" id="item-card"></section>
  <section class="card">
    <div class="nav">
      <div class="nav-group">
        <button type="button" id="prev-btn" class="secondary">Previous</button>
        <button type="button" id="next-btn" class="primary">Next</button>
      </div>
      <div class="nav-group">
        <button type="button" id="export-jsonl">Export answers as JSONL</button>
        <button type="button" id="export-csv">Export answers as CSV</button>
      </div>
    </div>
    <p class="small muted" id="status-text"></p>
  </section>
</main>
<script>
const ANNOTATOR_ID = {json.dumps(annotator_id)};
const STORAGE_KEY = {json.dumps(storage_key)};
const ITEMS = {data_json};
let current = 0;
let answers = loadAnswers();
let enteredAt = Date.now();

document.getElementById('annotator-id').textContent = ANNOTATOR_ID;

function escapeHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }}[ch]));
}}

function loadAnswers() {{
  try {{
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {{}};
  }} catch (err) {{
    return {{}};
  }}
}}

function saveAnswers() {{
  localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
}}

function recordTime(itemId) {{
  if (!answers[itemId]) return;
  const now = Date.now();
  const delta = Math.max(0, now - enteredAt);
  answers[itemId].time_spent_ms = (answers[itemId].time_spent_ms || 0) + delta;
  enteredAt = now;
  saveAnswers();
}}

function render() {{
  enteredAt = Date.now();
  const item = ITEMS[current];
  const saved = answers[item.assignment_id] || {{}};
  const checked = value => saved.selected_option === value ? 'checked' : '';
  document.getElementById('item-card').innerHTML = `
    <p class="small muted">Item ${{current + 1}} / ${{ITEMS.length}}</p>
    <h2>Question</h2>
    <div class="question">${{escapeHtml(item.question)}}</div>
    <h2>Dialogue Context</h2>
    ${{item.situation ? `<h3>Situation</h3><div class="box">${{escapeHtml(item.situation)}}</div>` : ''}}
    <h3>Dialogue</h3>
    <div class="box">${{escapeHtml(item.dialogue_history)}}</div>
    <div class="responses">
      <div>
        <div class="response-title">Response A</div>
        <div class="box">${{escapeHtml(item.response_a)}}</div>
      </div>
      <div>
        <div class="response-title">Response B</div>
        <div class="box">${{escapeHtml(item.response_b)}}</div>
      </div>
    </div>
    <h3>Choose one answer</h3>
    <div class="choices">
      <label class="choice"><input type="radio" name="choice_${{escapeHtml(item.assignment_id)}}" value="A" ${{checked('A')}}> Response A</label>
      <label class="choice"><input type="radio" name="choice_${{escapeHtml(item.assignment_id)}}" value="B" ${{checked('B')}}> Response B</label>
      <label class="choice"><input type="radio" name="choice_${{escapeHtml(item.assignment_id)}}" value="TIE" ${{checked('TIE')}}> Tie / both are equally good</label>
      <label class="choice"><input type="radio" name="choice_${{escapeHtml(item.assignment_id)}}" value="NEITHER" ${{checked('NEITHER')}}> Neither / both are bad</label>
    </div>
    <h3>Comment (optional)</h3>
    <textarea id="comment-box" placeholder="Optional comment">${{escapeHtml(saved.comment || '')}}</textarea>
  `;
  document.querySelectorAll(`input[name="choice_${{item.assignment_id}}"]`).forEach(input => {{
    input.addEventListener('change', () => {{
      answers[item.assignment_id] = answers[item.assignment_id] || {{
        annotator_id: ANNOTATOR_ID,
        assignment_id: item.assignment_id,
        item_uid: item.item_uid,
        first_answered_at: new Date().toISOString(),
        time_spent_ms: 0
      }};
      answers[item.assignment_id].selected_option = input.value;
      answers[item.assignment_id].timestamp = new Date().toISOString();
      saveAnswers();
      updateProgress();
    }});
  }});
  document.getElementById('comment-box').addEventListener('input', event => {{
    answers[item.assignment_id] = answers[item.assignment_id] || {{
      annotator_id: ANNOTATOR_ID,
      assignment_id: item.assignment_id,
      item_uid: item.item_uid,
      time_spent_ms: 0
    }};
    answers[item.assignment_id].comment = event.target.value;
    answers[item.assignment_id].timestamp = new Date().toISOString();
    saveAnswers();
  }});
  document.getElementById('prev-btn').disabled = current === 0;
  document.getElementById('next-btn').textContent = current === ITEMS.length - 1 ? 'Finish' : 'Next';
  updateProgress();
}}

function updateProgress() {{
  const answered = ITEMS.filter(item => answers[item.assignment_id] && answers[item.assignment_id].selected_option).length;
  const pct = ITEMS.length ? (answered / ITEMS.length) * 100 : 0;
  document.getElementById('progress-bar').style.width = `${{pct}}%`;
  document.getElementById('progress-text').textContent = `${{answered}} / ${{ITEMS.length}} answered`;
  document.getElementById('status-text').textContent = answered === ITEMS.length
    ? 'All items are answered. You can export now.'
    : 'Your answers are saved automatically in this browser.';
}}

function goNext() {{
  const item = ITEMS[current];
  if (!answers[item.assignment_id] || !answers[item.assignment_id].selected_option) {{
    alert('Please answer this item before going to the next one.');
    return;
  }}
  recordTime(item.assignment_id);
  if (current < ITEMS.length - 1) {{
    current += 1;
    render();
  }} else {{
    updateProgress();
    alert('All done. Please export your answers.');
  }}
}}

function goPrev() {{
  const item = ITEMS[current];
  recordTime(item.assignment_id);
  if (current > 0) {{
    current -= 1;
    render();
  }}
}}

function answerRows() {{
  const now = new Date().toISOString();
  return ITEMS.map(item => {{
    const ans = answers[item.assignment_id] || {{}};
    return {{
      annotator_id: ANNOTATOR_ID,
      assignment_id: item.assignment_id,
      item_uid: item.item_uid,
      selected_option: ans.selected_option || '',
      timestamp: ans.timestamp || now,
      time_spent_ms: ans.time_spent_ms || 0,
      comment: ans.comment || ''
    }};
  }});
}}

function download(filename, text, type) {{
  const blob = new Blob([text], {{type}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}

function exportJsonl() {{
  if (ITEMS[current]) recordTime(ITEMS[current].assignment_id);
  const rows = answerRows();
  const missing = rows.filter(row => !row.selected_option).length;
  if (missing && !confirm(`${{missing}} items are unanswered. Export partial answers anyway?`)) return;
  download(`${{ANNOTATOR_ID}}_human_ablation_answers.jsonl`, rows.map(row => JSON.stringify(row)).join('\\n') + '\\n', 'application/jsonl');
}}

function csvEscape(value) {{
  const text = String(value ?? '');
  return /[",\\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}}

function exportCsv() {{
  if (ITEMS[current]) recordTime(ITEMS[current].assignment_id);
  const rows = answerRows();
  const missing = rows.filter(row => !row.selected_option).length;
  if (missing && !confirm(`${{missing}} items are unanswered. Export partial answers anyway?`)) return;
  const columns = ['annotator_id', 'assignment_id', 'item_uid', 'selected_option', 'timestamp', 'time_spent_ms', 'comment'];
  const csv = [columns.join(',')].concat(rows.map(row => columns.map(col => csvEscape(row[col])).join(','))).join('\\n') + '\\n';
  download(`${{ANNOTATOR_ID}}_human_ablation_answers.csv`, csv, 'text/csv');
}}

document.getElementById('next-btn').addEventListener('click', goNext);
document.getElementById('prev-btn').addEventListener('click', goPrev);
document.getElementById('export-jsonl').addEventListener('click', exportJsonl);
document.getElementById('export-csv').addEventListener('click', exportCsv);
window.addEventListener('beforeunload', () => {{
  if (ITEMS[current]) recordTime(ITEMS[current].assignment_id);
}});
render();
</script>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def render_model_labeled_html(path: Path, items: list[dict[str, Any]]) -> None:
    rows = []
    for idx, item in enumerate(items, 1):
        rows.append(
            f"""
  <section class="card">
    <p class="small muted">Item {idx} / {len(items)}</p>
    <h2>{html.escape(item['comparison'])}</h2>
    <p><strong>context_uid:</strong> {html.escape(str(item['context_uid']))}</p>
    <p><strong>example_id:</strong> {html.escape(str(item['example_id']))}</p>
    <h3>Situation</h3><div class="box">{html.escape(str(item.get('situation', '')))}</div>
    <h3>Dialogue</h3><div class="box">{html.escape(str(item.get('dialogue_history', '')))}</div>
    <div class="responses" style="margin-top:16px;">
      <div>
        <div class="response-title">final_emoh response</div>
        <div class="box">{html.escape(str(item.get('response_final_emoh', '')))}</div>
        <p class="small muted">words: {word_count(str(item.get('response_final_emoh', '')))}</p>
      </div>
      <div>
        <div class="response-title">{html.escape(str(item.get('ablation_name', '')))} response</div>
        <div class="box">{html.escape(str(item.get('response_ablation', '')))}</div>
        <p class="small muted">words: {word_count(str(item.get('response_ablation', '')))}</p>
      </div>
    </div>
  </section>"""
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human Ablation Internal Model-Labeled Inspection</title>
  <style>{STYLE}</style>
</head>
<body>
<main class="app">
  <section class="top">
    <h1>Human Ablation Internal Model-Labeled Inspection</h1>
    <p>This file is for author-side quality inspection only. It intentionally shows model labels and must not be sent to annotators.</p>
  </section>
  {''.join(rows)}
</main>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def sample_model_labeled(items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 41)
    targets = {"final_vs_no_rerank": 7, "final_vs_no_role_aware": 7, "final_vs_zero_control": 6}
    sampled: list[dict[str, Any]] = []
    for comparison, count in targets.items():
        candidates = [row for row in items if row["comparison"] == comparison]
        rng.shuffle(candidates)
        sampled.extend(candidates[:count])
    rng.shuffle(sampled)
    return sampled


def make_blind_key(assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "assignment_id": row["assignment_id"],
            "annotator_id": row["annotator_id"],
            "item_uid": row["item_uid"],
            "context_uid": row["context_uid"],
            "example_id": row["example_id"],
            "comparison": row["comparison"],
            "ablation_name": row["ablation_name"],
            "a_is": row["a_is"],
            "b_is": row["b_is"],
            "final_position": row["final_position"],
            "ablation_position": row["ablation_position"],
        }
        for row in assignments
    ]


def summary_tables(
    output_dir: Path,
    assignments: list[dict[str, Any]],
    annotator_ids: list[str],
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    load_rows: list[dict[str, Any]] = []
    for aid in annotator_ids:
        subset = [row for row in assignments if row["annotator_id"] == aid]
        counts = Counter(row["comparison"] for row in subset)
        load_rows.append(
            {
                "annotator_id": aid,
                "total_items": len(subset),
                "final_vs_no_rerank": counts["final_vs_no_rerank"],
                "final_vs_no_role_aware": counts["final_vs_no_role_aware"],
                "final_vs_zero_control": counts["final_vs_zero_control"],
            }
        )
    write_csv(
        output_dir / "annotator_load_summary.csv",
        load_rows,
        ["annotator_id", "total_items", "final_vs_no_rerank", "final_vs_no_role_aware", "final_vs_zero_control"],
    )

    balance_rows: list[dict[str, Any]] = []
    for comp in [c["comparison"] for c in COMPARISONS]:
        subset = [row for row in assignments if row["comparison"] == comp]
        balance_rows.append(
            {
                "comparison": comp,
                "unique_items": len({row["item_uid"] for row in subset}),
                "judgments": len(subset),
                "final_in_A": sum(1 for row in subset if row["final_position"] == "A"),
                "final_in_B": sum(1 for row in subset if row["final_position"] == "B"),
            }
        )
    write_csv(
        output_dir / "comparison_balance.csv",
        balance_rows,
        ["comparison", "unique_items", "judgments", "final_in_A", "final_in_B"],
    )

    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_context[row["context_uid"]].append(row)
    context_rows: list[dict[str, Any]] = []
    for record in selected:
        subset = by_context[record["context_uid"]]
        annotators = sorted({row["annotator_id"] for row in subset})
        context_rows.append(
            {
                "context_uid": record["context_uid"],
                "example_id": record["example_id"],
                "dialogue_id": record.get("dialogue_id", ""),
                "judgment_count": len(subset),
                "distinct_annotators": len(annotators),
                "held_out_annotator": next((aid for aid in annotator_ids if aid not in annotators), ""),
                "annotators": " ".join(annotators),
            }
        )
    write_csv(
        output_dir / "context_assignment_summary.csv",
        context_rows,
        [
            "context_uid",
            "example_id",
            "dialogue_id",
            "judgment_count",
            "distinct_annotators",
            "held_out_annotator",
            "annotators",
        ],
    )
    return load_rows, balance_rows, context_rows


def blind_html_forbidden_hits(output_dir: Path, annotator_ids: list[str]) -> list[str]:
    hits: list[str] = []
    for aid in annotator_ids:
        path = output_dir / f"{aid}.html"
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for pattern in FORBIDDEN_BLIND_PATTERNS:
            if pattern.lower() in text:
                hits.append(f"{path.name}: {pattern}")
    return hits


def validate_outputs(
    output_dir: Path,
    selected: list[dict[str, Any]],
    items: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    annotator_ids: list[str],
    annotators_per_item: int,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["annotator_html_count"] = sum(1 for aid in annotator_ids if (output_dir / f"{aid}.html").exists())
    checks["model_labeled_html_exists"] = (output_dir / "model_labeled_20.html").exists()
    checks["selected_contexts"] = len(selected)
    checks["unassigned_items"] = len(items)
    checks["assignments"] = len(assignments)
    checks["load_counts"] = dict(Counter(row["annotator_id"] for row in assignments))
    checks["item_annotator_counts_ok"] = all(
        count == annotators_per_item for count in Counter(row["item_uid"] for row in assignments).values()
    )
    context_seen: dict[tuple[str, str], int] = Counter((row["annotator_id"], row["context_uid"]) for row in assignments)
    checks["annotator_repeated_context_count"] = sum(1 for count in context_seen.values() if count > 1)
    checks["comparison_judgments"] = dict(Counter(row["comparison"] for row in assignments))
    checks["empty_response_assignments"] = sum(
        1 for row in assignments if not norm_text(row["response_a"]) or not norm_text(row["response_b"])
    )
    checks["identical_response_assignments"] = sum(
        1 for row in assignments if norm_text(row["response_a"]) == norm_text(row["response_b"])
    )
    checks["empty_context_assignments"] = sum(1 for row in assignments if not norm_text(row["dialogue_history"]))
    checks["final_position_counts"] = dict(Counter(row["final_position"] for row in assignments))
    checks["blind_forbidden_hits"] = blind_html_forbidden_hits(output_dir, annotator_ids)
    final_word_counts = [word_count(row["final_emoh"]) for row in selected]
    checks["selected_final_min_words"] = min(final_word_counts) if final_word_counts else 0
    checks["selected_final_complete_sentence_count"] = sum(
        1 for row in selected if looks_complete_sentence(row["final_emoh"])
    )
    checks["passed"] = (
        checks["annotator_html_count"] == len(annotator_ids)
        and checks["model_labeled_html_exists"]
        and len(selected) == 50
        and len(items) == 150
        and len(assignments) == 450
        and all(count == 45 for count in checks["load_counts"].values())
        and checks["item_annotator_counts_ok"]
        and checks["annotator_repeated_context_count"] == 0
        and all(count == 150 for count in checks["comparison_judgments"].values())
        and checks["empty_response_assignments"] == 0
        and checks["identical_response_assignments"] == 0
        and checks["empty_context_assignments"] == 0
        and not checks["blind_forbidden_hits"]
        and checks["selected_final_min_words"] >= 15
        and checks["selected_final_complete_sentence_count"] == len(selected)
    )
    return checks


def write_missing_report(output_dir: Path, missing: list[str]) -> None:
    lines = [
        "# Missing Artifacts",
        "",
        "The human-ablation materials were not generated because required artifacts were missing or could not be identified.",
        "",
        "| Missing item |",
        "|---|",
    ]
    lines.extend(f"| {m} |" for m in missing)
    lines.extend(
        [
            "",
            "Please provide the missing generation outputs or the original inference commands/checkpoints for the corresponding ablation conditions.",
        ]
    )
    (output_dir / "MISSING_ARTIFACTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(
    output_dir: Path,
    paths: dict[str, Path],
    seed: int,
    selection_summary: dict[str, Any],
    exclusion_note: str,
    load_rows: list[dict[str, Any]],
    balance_rows: list[dict[str, Any]],
    checks: dict[str, Any],
) -> None:
    load_table = "\n".join(
        f"| {r['annotator_id']} | {r['total_items']} | {r['final_vs_no_rerank']} | {r['final_vs_no_role_aware']} | {r['final_vs_zero_control']} |"
        for r in load_rows
    )
    balance_table = "\n".join(
        f"| {r['comparison']} | {r['unique_items']} | {r['judgments']} | {r['final_in_A']} | {r['final_in_B']} |"
        for r in balance_rows
    )
    blind_status = "passed" if not checks.get("blind_forbidden_hits") else "failed"
    missing_status = "No missing artifacts were detected."
    readme = f"""# Human Ablation Evaluation Materials

## Purpose

This folder contains the blind pairwise annotation materials for the EmoStance generation-control ablation study. These files only prepare the human-evaluation task; they are not human-evaluation results.

## Design Summary

- Contexts: 50
- Comparisons per context: 3
- Question per item: 1
- Annotators: 10
- Annotators per item: 3
- Total blind judgments: 450
- Average annotator load: 45 A/B items
- Sampling seed: {seed}
- Assignment seed: {seed}

Question:

> {QUESTION}

## Comparisons

| Comparison | Definition | Project artifact |
|---|---|---|
| final_vs_no_rerank | final deployable system vs same pool before reranking | `{paths['final']}` and `{paths['no_rerank']}` |
| final_vs_no_role_aware | final deployable system vs predicted-control baseline without the role-aware predictor variant | `{paths['no_role_aware']}` |
| final_vs_zero_control | final deployable system vs null/zero stance control | `{paths['zero_control']}` |

`final_emoh` is loaded from `selection_type=rerank_control`. `no_rerank` is loaded from `selection_type=raw_first` in the same selected-test artifact, so it uses the same candidate pool whenever the project artifact exposes it. `no_role_aware` is loaded from `control_type=predicted`, the project generation-control predicted-control baseline before the role-aware c7mix variant. `zero_control` is loaded from `control_type=zero`.

Gold control, shuffled control, and oracle gold selection are intentionally excluded from this human evaluation.

## Reused Artifacts

- Main HTML/CSS/JS style and offline export behavior were adapted from `runs/human_eval_other_methods_main/build_other_methods_eval.py`.
- Final/rerank and no-rerank outputs were reused from `{paths['final']}`.
- Predicted-control and zero-control outputs were reused from `{paths['no_role_aware']}`.
- {exclusion_note}

No generation or inference jobs were rerun.

## Sampling

The script selected 50 contexts where all four systems had non-empty outputs and where all three final-vs-ablation pairs were not identical. The script preferred contexts not used by the main baseline human evaluation when the exclusion file was available.

Additional final-system response filter: `final_emoh` must contain at least 15 words and must end like a complete sentence. The script applies this as a selection filter over existing generations; it does not rewrite or expand model outputs.

Selection summary:

```json
{json.dumps(selection_summary, ensure_ascii=False, indent=2)}
```

## Annotator Assignment

For each context, one annotator is held out and the other nine annotators are partitioned across the three comparisons. This ensures that the same annotator sees at most one comparison for the same context.

| Annotator | Total | final_vs_no_rerank | final_vs_no_role_aware | final_vs_zero_control |
|---|---:|---:|---:|---:|
{load_table}

## A/B Randomization

A/B order is randomized independently for each assigned item and balanced by comparison. The blind key is saved in `blind_key.jsonl` and is not embedded in annotator HTML files.

| Comparison | Unique items | Judgments | final in A | final in B |
|---|---:|---:|---:|---:|
{balance_table}

## Scoring Rule

Scoring is counted from the `final_emoh` perspective:

- If the selected anonymous response is `final_emoh`, count a final win.
- If the selected anonymous response is the ablation response, count a final loss.
- `Tie / both are equally good` and `Neither / both are bad` are neutral and should be excluded from decisive win-rate denominators.

## Files

- Blind annotator HTML: `annotator_01.html` through `annotator_10.html`
- Internal labeled inspection HTML: `model_labeled_20.html`
- Selected contexts: `selected_contexts.jsonl`
- Unassigned items: `items_unassigned.jsonl`
- Assignments with real mappings: `assignments.jsonl`
- Blind key: `blind_key.jsonl`
- Load summary: `annotator_load_summary.csv`
- Comparison balance: `comparison_balance.csv`
- Context assignment summary: `context_assignment_summary.csv`

## Quality Checks

| Check | Result |
|---|---|
| 10 annotator HTML files exist | {checks['annotator_html_count'] == 10} |
| model_labeled_20.html exists | {checks['model_labeled_html_exists']} |
| selected_contexts = 50 | {checks['selected_contexts'] == 50} |
| unassigned items = 150 | {checks['unassigned_items'] == 150} |
| assignments = 450 | {checks['assignments'] == 450} |
| each annotator has 45 items | {all(count == 45 for count in checks['load_counts'].values())} |
| each item has 3 annotators | {checks['item_annotator_counts_ok']} |
| no annotator repeats a context | {checks['annotator_repeated_context_count'] == 0} |
| each comparison has 150 judgments | {all(count == 150 for count in checks['comparison_judgments'].values())} |
| Response A/B non-empty | {checks['empty_response_assignments'] == 0} |
| Response A/B not identical | {checks['identical_response_assignments'] == 0} |
| context non-empty | {checks['empty_context_assignments'] == 0} |
| selected final_emoh min word count >= 15 | {checks['selected_final_min_words'] >= 15} |
| selected final_emoh responses are complete sentences | {checks['selected_final_complete_sentence_count'] == checks['selected_contexts']} |
| blind HTML forbidden-label check | {blind_status} |
| final_emoh A/B balance | {checks['final_position_counts']} |
| missing artifacts | {missing_status} |

Overall quality-check status: {"PASSED" if checks.get("passed") else "FAILED"}

## Notes and Limitations

The no-role-aware condition is mapped to the existing project `predicted control baseline` artifact. This follows the project ablation naming in which role-aware predicted control is a later deployable variant. If a separate file explicitly named `w/o role-aware target predictor` is added later, the script can be rerun with `--no-role-aware-output-path`.

The internal `model_labeled_20.html` file displays true system labels and must not be shared with annotators.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = default_paths(root, args)
    records, missing = load_project_outputs(paths)
    if missing:
        write_missing_report(output_dir, missing)
        print("Missing required artifacts. See human_ablation/MISSING_ARTIFACTS.md")
        return

    excluded_dialogues, exclusion_note = load_excluded_dialogues(paths["exclude"])
    selected, selection_summary = select_contexts(
        records,
        excluded_dialogues,
        args.num_contexts,
        args.seed,
        args.min_final_words,
        args.require_final_complete_sentence,
    )
    items = build_unassigned_items(selected)
    annotator_ids = [f"annotator_{i:02d}" for i in range(1, args.num_annotators + 1)]
    assignments = assign_items(items, args.num_annotators, args.annotators_per_item, args.seed)

    write_jsonl(output_dir / "selected_contexts.jsonl", selected)
    write_jsonl(output_dir / "items_unassigned.jsonl", items)
    write_jsonl(output_dir / "assignments.jsonl", assignments)
    write_jsonl(output_dir / "blind_key.jsonl", make_blind_key(assignments))

    for aid in annotator_ids:
        subset = [row for row in assignments if row["annotator_id"] == aid]
        subset = sorted(subset, key=lambda row: (row["context_uid"], row["comparison"]))
        random.Random(args.seed + int(aid.rsplit("_", 1)[1])).shuffle(subset)
        render_annotator_html(output_dir / f"{aid}.html", aid, subset)

    labeled = sample_model_labeled(items, args.seed)
    write_jsonl(output_dir / "model_labeled_20.jsonl", labeled)
    render_model_labeled_html(output_dir / "model_labeled_20.html", labeled)

    load_rows, balance_rows, _context_rows = summary_tables(output_dir, assignments, annotator_ids, selected)
    checks = validate_outputs(output_dir, selected, items, assignments, annotator_ids, args.annotators_per_item)
    write_readme(output_dir, paths, args.seed, selection_summary, exclusion_note, load_rows, balance_rows, checks)

    if not checks.get("passed"):
        print("Generated files, but quality checks failed. See human_ablation/README.md")
    else:
        print("Done.")
        print("Generated:")
        print("- 10 annotator HTML files")
        print("- 1 model-labeled inspection HTML file")
        print("- selected contexts, assignments, and blind key")
        print("- load, comparison, and context summaries")
        print("Open human_ablation/annotator_01.html in browser to start annotation.")


if __name__ == "__main__":
    main()
