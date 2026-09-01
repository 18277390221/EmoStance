from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from system_baseline.utils.io import load_yaml, read_json, read_jsonl, relpath, write_csv, write_json, write_jsonl
from system_baseline.utils.metrics import aggregate_system_metrics
from system_baseline.utils.text import clean_prediction, clean_text, context_key, history_from_context_list, normalize_for_match


DISPLAY_NAMES = {
    "emostance": "EmoStance / Ours",
    "ours": "EmoStance / Ours",
    "llm_only": "LLM-only",
    "llm_prompt": "LLM-prompt",
    "llm_sft": "LLM-SFT",
    "empo_dpo": "EmPO-DPO",
    "case": "CASE",
    "aptness": "APTNESS",
    "sibyl": "Sibyl",
}

PREDICTION_FIELDS = [
    "prediction",
    "generated_response",
    "response",
    "output",
    "candidate",
    "text",
    "case_response",
    "initial_response",
    "mistral_sibyl_response",
    "sibyl_response",
    "EmPO-SFT",
    "EmPO-DPO",
    "llm_sft",
    "empo_dpo",
]

METRIC_FIELDS = [
    "Method",
    "N",
    "BERTScore",
    "ROUGE-L",
    "BLEU-2",
    "METEOR",
    "Distinct-1",
    "Distinct-2",
    "Self-BLEU",
    "Generic",
    "Runnable",
    "Coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate system generation files with a single metric implementation.")
    parser.add_argument("--config", default="system_baseline/configs/system_eval.yaml")
    parser.add_argument("--input", default="system_baseline/data/ed_test_canonical.jsonl")
    parser.add_argument("--generation_dir", default="system_baseline/outputs/generations")
    parser.add_argument("--output_dir", default="system_baseline/outputs/metrics")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--skip-bertscore", action="store_true")
    return parser.parse_args()


def canonical_indices(examples: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["example_id"]): row for row in examples}
    by_content: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}
    by_dialogue_turn: dict[tuple[str, str], dict[str, Any]] = {}
    for row in examples:
        by_content[context_key(row.get("situation"), row.get("history", []), row.get("reference"))] = row
        dialogue_id = str(row.get("dialogue_id") or "")
        turn_id = str(row.get("turn_id") or "")
        if dialogue_id and turn_id:
            by_dialogue_turn[(dialogue_id, turn_id)] = row
    return {"by_id": by_id, "by_content": by_content, "by_dialogue_turn": by_dialogue_turn}


def pick_prediction(row: Mapping[str, Any], system_id: str | None = None) -> str:
    fields = list(PREDICTION_FIELDS)
    if system_id == "llm_sft":
        fields = ["LLM-SFT", "EmPO-SFT", "llm_sft"] + fields
    if system_id == "empo_dpo":
        fields = ["EmPO-DPO", "empo_dpo"] + fields
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return clean_prediction(value)
    return ""


def artifact_history(row: Mapping[str, Any]) -> list[dict[str, str]]:
    history = row.get("history") or row.get("context")
    if isinstance(history, list):
        return history_from_context_list(history)
    if isinstance(history, str):
        out: list[dict[str, str]] = []
        text = history.replace("<SEP>", "\n")
        for idx, line in enumerate(text.splitlines()):
            line = clean_text(line)
            if not line:
                continue
            if ":" in line:
                role, content = line.split(":", 1)
                role_norm = role.strip().lower()
                speaker = "B" if role_norm in {"b", "assistant", "listener"} else "A" if role_norm in {"a", "user", "speaker"} else ("A" if idx % 2 == 0 else "B")
                out.append({"speaker": speaker, "text": clean_text(content)})
            else:
                out.append({"speaker": "A" if idx % 2 == 0 else "B", "text": line})
        return out
    return []


def match_artifact_row(row: Mapping[str, Any], indices: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    by_id = indices["by_id"]
    for key in ("example_id", "context_id", "id"):
        value = str(row.get(key) or "")
        if value in by_id:
            return by_id[value], "example_id"
        if "_direct_" in value and value.split("_direct_", 1)[0] in by_id:
            return by_id[value.split("_direct_", 1)[0]], "example_id_prefix"
    dialogue_id = str(row.get("dialogue_id") or "")
    turn_id = str(row.get("turn_id") or "")
    if dialogue_id and turn_id:
        candidates = [
            (dialogue_id, turn_id),
            (dialogue_id, str(int(turn_id) + 1)) if turn_id.isdigit() else ("", ""),
            (dialogue_id, str(int(turn_id) - 1)) if turn_id.isdigit() else ("", ""),
        ]
        for cand in candidates:
            if cand in indices["by_dialogue_turn"]:
                return indices["by_dialogue_turn"][cand], "dialogue_turn"
    ref = row.get("reference") or row.get("gold_response") or row.get("target") or row.get("response_reference") or ""
    key = context_key(row.get("situation", ""), artifact_history(row), ref)
    if key in indices["by_content"]:
        return indices["by_content"][key], "content_reference"
    # Some old outputs do not preserve role tags. Fall back to normalized context text and reference.
    ref_norm = normalize_for_match(ref)
    situation_norm = normalize_for_match(row.get("situation", ""))
    history_norm = tuple(normalize_for_match(t.get("text", "")) for t in artifact_history(row))
    for content_key, example in indices["by_content"].items():
        if content_key[0] == situation_norm and content_key[1] == history_norm and content_key[2] == ref_norm:
            return example, "content_reference_scan"
    return None, "unmatched"


def load_standard_generation(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        prediction = pick_prediction(row)
        if not prediction:
            continue
        out.append({**row, "prediction": prediction})
    return out


def validate_full_coverage(rows: list[dict[str, Any]], expected_ids: set[str]) -> tuple[bool, dict[str, Any]]:
    ids = [str(row.get("example_id", "")) for row in rows]
    unique = set(ids)
    missing = expected_ids - unique
    extra = unique - expected_ids
    duplicate_count = len(ids) - len(unique)
    ok = len(ids) == len(expected_ids) and not missing and not extra and duplicate_count == 0
    return ok, {
        "generated_n": len(ids),
        "matched_n": len(unique & expected_ids),
        "missing_example_id_count": len(missing),
        "extra_example_id_count": len(extra),
        "duplicate_example_id_count": duplicate_count,
    }


def metric_row(
    system_id: str,
    display_name: str,
    rows: list[dict[str, Any]],
    canonical_by_id: Mapping[str, dict[str, Any]],
    eval_cfg: Mapping[str, Any],
    compute_bertscore: bool,
    runnable: bool,
    coverage: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: str(r["example_id"]))
    predictions = [str(r["prediction"]) for r in ordered]
    references = [str(canonical_by_id[str(r["example_id"])]["reference"]) for r in ordered]
    metrics, per_rows, bert_note = aggregate_system_metrics(
        predictions,
        references,
        compute_bertscore=compute_bertscore,
        bertscore_model_type=str(eval_cfg.get("bertscore_model_type", "distilbert-base-uncased")),
        bertscore_num_layers=int(eval_cfg.get("bertscore_num_layers", 6)),
        bertscore_batch_size=int(eval_cfg.get("bertscore_batch_size", 64)),
        self_bleu_max_items=int(eval_cfg.get("self_bleu_max_items", 200)),
        self_bleu_seed=int(eval_cfg.get("self_bleu_seed", 20260512)),
    )
    for per, gen in zip(per_rows, ordered):
        per["system_id"] = system_id
        per["Method"] = display_name
        per["example_id"] = str(gen["example_id"])
    table = {
        "Method": display_name,
        "N": len(rows),
        "BERTScore": metrics["bertscore_f1"],
        "ROUGE-L": metrics["rouge_l"],
        "BLEU-2": metrics["bleu_2"],
        "METEOR": metrics["meteor"],
        "Distinct-1": metrics["distinct_1"],
        "Distinct-2": metrics["distinct_2"],
        "Self-BLEU": metrics["self_bleu"],
        "Generic": metrics["generic_response_rate"],
        "Runnable": runnable,
        "Coverage": coverage,
    }
    return table, per_rows, {"system_id": system_id, "display_name": display_name, "metrics": metrics, "bert_score_note": bert_note}


def load_generation_status(root: Path) -> dict[str, Any]:
    path = root / "system_baseline/outputs/logs/generation_status.json"
    if path.exists():
        return read_json(path)
    return {"systems": []}


def configured_artifacts(cfg: Mapping[str, Any], root: Path) -> dict[str, list[Path]]:
    systems = cfg.get("systems", {}) if isinstance(cfg.get("systems"), dict) else {}
    out: dict[str, list[Path]] = {}
    for system_id, block in systems.items():
        paths = []
        for raw in block.get("artifact_candidates", []) or []:
            p = root / str(raw)
            if p.exists():
                paths.append(p)
        # Also diagnose the legacy system_baseline/generations directory if present.
        legacy = root / "system_baseline/generations" / f"{system_id}.jsonl"
        if legacy.exists():
            paths.append(legacy)
        out[str(system_id)] = paths
    return out


def evaluate_artifacts(
    cfg: Mapping[str, Any],
    root: Path,
    examples: list[dict[str, Any]],
    indices: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    compute_bertscore: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_ids = {str(row["example_id"]) for row in examples}
    canonical_by_id = indices["by_id"]
    diagnostic_rows: list[dict[str, Any]] = []
    artifact_status: list[dict[str, Any]] = []
    for system_id, paths in configured_artifacts(cfg, root).items():
        matched_by_id: dict[str, dict[str, Any]] = {}
        match_methods: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        raw_count = 0
        for path in paths:
            for row in read_jsonl(path):
                raw_count += 1
                example, method = match_artifact_row(row, indices)
                match_methods[method] = match_methods.get(method, 0) + 1
                if not example:
                    continue
                prediction = pick_prediction(row, system_id)
                if not prediction:
                    continue
                example_id = str(example["example_id"])
                if example_id in matched_by_id:
                    continue
                matched_by_id[example_id] = {
                    "example_id": example_id,
                    "prediction": prediction,
                    "source_path": relpath(path, root),
                }
                source_counts[relpath(path, root)] = source_counts.get(relpath(path, root), 0) + 1
        rows = list(matched_by_id.values())
        artifact_status.append(
            {
                "system_id": system_id,
                "display_name": DISPLAY_NAMES.get(system_id, system_id),
                "raw_artifact_rows": raw_count,
                "matched_n": len(rows),
                "expected_n": len(expected_ids),
                "coverage": len(rows) / len(expected_ids) if expected_ids else 0.0,
                "artifact_paths": [relpath(p, root) for p in paths],
                "match_methods": match_methods,
                "source_counts": source_counts,
            }
        )
        if rows:
            table, _, extra = metric_row(
                system_id,
                DISPLAY_NAMES.get(system_id, system_id),
                rows,
                canonical_by_id,
                eval_cfg,
                compute_bertscore=compute_bertscore,
                runnable=False,
                coverage=len(rows) / len(expected_ids) if expected_ids else 0.0,
            )
            diagnostic_rows.append(
                {
                    **table,
                    "SystemId": system_id,
                    "DiagnosticOnly": True,
                    "ArtifactPaths": "; ".join(relpath(p, root) for p in paths),
                    "BERTScoreNote": extra["bert_score_note"],
                }
            )
    return diagnostic_rows, artifact_status


def write_coverage_report(
    path: Path,
    canonical_path: Path,
    examples: list[dict[str, Any]],
    generation_status: Mapping[str, Any],
    full_status: Mapping[str, dict[str, Any]],
    artifact_status: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    root: Path,
) -> None:
    expected = len(examples)
    artifact_by_id = {row["system_id"]: row for row in artifact_status}
    configured = [str(x) for x in cfg.get("system_order", [])] or list(DISPLAY_NAMES)
    gen_by_id = {row.get("system_id"): row for row in generation_status.get("systems", []) if isinstance(row, dict)}
    lines = [
        "# Coverage Report",
        "",
        f"Canonical ED test set: `{relpath(canonical_path, root)}`",
        f"Canonical examples: `{expected}`",
        "",
        "Main table policy: only `full_run_completed` systems with exactly the canonical `example_id` set are included.",
        "Diagnostic artifact rows are coverage-limited and are not used as the fair leaderboard.",
        "",
        "| system | status | expected N | generated N | generated matched N | missing generated ids | artifact matched N | notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for system_id in configured:
        gen = gen_by_id.get(system_id, {})
        art = artifact_by_id.get(system_id, {})
        status = gen.get("status") or "not_runnable_from_current_repository"
        if full_status.get(system_id):
            status = "full_run_completed"
        elif status == "not_runnable_from_current_repository" and art.get("matched_n", 0):
            status = "artifact_only"
        generated = int(gen.get("generated_n") or full_status.get(system_id, {}).get("generated_n") or 0)
        matched = int(gen.get("matched_n") or full_status.get(system_id, {}).get("matched_n") or 0)
        artifact_matched = int(art.get("matched_n") or 0)
        missing = expected - matched if matched <= expected else 0
        reason = clean_text(gen.get("reason") or "")
        if status == "artifact_only":
            reason = f"available artifact matched N={artifact_matched}; not used in main table"
        elif status == "full_run_completed" and artifact_matched == expected and system_id == "case":
            reason = (
                reason
                + f" Repository-local CASE full ED output matched all {expected} canonical examples "
                "and was standardized into the main generation file."
            ).strip()
        elif status == "full_run_completed" and system_id == "aptness":
            reason = (
                reason
                + " Repository-local APT-RAG/Mistral reproduction was run on all canonical ED inputs; "
                "coverage-limited old APTNESS artifacts remain diagnostic only."
            ).strip()
        elif status == "full_run_completed" and system_id == "sibyl":
            reason = (
                reason
                + " Repository-local Mistral-Sibyl LoRA generator was run with the full Sibyl_test "
                "preconstructed knowledge file aligned to canonical example_id; old sample artifacts remain diagnostic only."
            ).strip()
        elif artifact_matched:
            reason = (reason + f" Diagnostic artifact matched N={artifact_matched}; not used in main table.").strip()
        lines.append(
            f"| {DISPLAY_NAMES.get(system_id, system_id)} | `{status}` | {expected} | {generated} | {matched} | {missing} | {artifact_matched} | {reason} |"
        )
    lines += [
        "",
        "## Commands",
        "",
        "- `python -m system_baseline.scripts.find_project_data`",
        "- `python -m system_baseline.scripts.prepare_ed_test --config system_baseline/configs/system_eval.yaml`",
        "- `python -m system_baseline.scripts.run_system_generation --config system_baseline/configs/system_eval.yaml --input system_baseline/data/ed_test_canonical.jsonl --output_dir system_baseline/outputs/generations`",
        "- `python -m system_baseline.scripts.evaluate_system_metrics --input system_baseline/data/ed_test_canonical.jsonl --generation_dir system_baseline/outputs/generations --output_dir system_baseline/outputs/metrics`",
        "",
        "## Configured Paths",
        "",
    ]
    for system_id, block in (cfg.get("systems", {}) or {}).items():
        lines.append(f"### {DISPLAY_NAMES.get(system_id, system_id)}")
        for key in ("model_path", "tokenizer_path", "adapter_path", "checkpoint", "generator_dir", "stance_dir", "prepared_dir"):
            if block.get(key):
                lines.append(f"- {key}: `{block[key]}`")
        for artifact in block.get("artifact_candidates", []) or []:
            lines.append(f"- artifact: `{artifact}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diagnostic_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & N & BERTScore & ROUGE-L & BLEU-2 & METEOR & Distinct-1 & Self-BLEU & Generic \\\\",
        "\\midrule",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value in (None, ""):
                return "--"
            try:
                return f"{float(value):.4f}"
            except Exception:
                return str(value)
        lines.append(
            f"{row['Method']} & {row['N']} & {fmt(row.get('BERTScore'))} & {fmt(row.get('ROUGE-L'))} & "
            f"{fmt(row.get('BLEU-2'))} & {fmt(row.get('METEOR'))} & {fmt(row.get('Distinct-1'))} & "
            f"{fmt(row.get('Self-BLEU'))} & {fmt(row.get('Generic'))} \\\\"
        )
    if not rows:
        lines.append("\\multicolumn{9}{c}{No available artifacts matched the canonical set.} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Coverage-limited diagnostic evaluation of available prior-system output artifacts. These rows are not used as the main fair comparison because the available outputs do not cover the identical ED test subset.}",
        "\\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    cfg = load_yaml(root / args.config)
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation"), dict) else {}
    examples = read_jsonl(root / args.input)
    if not examples:
        raise ValueError(f"No canonical examples found at {args.input}")
    indices = canonical_indices(examples)
    canonical_by_id = indices["by_id"]
    expected_ids = set(canonical_by_id)
    generation_dir = root / args.generation_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_bertscore = bool(eval_cfg.get("compute_bertscore", True)) and not args.skip_bertscore
    generation_status = load_generation_status(root)
    gen_status_by_id = {row.get("system_id"): row for row in generation_status.get("systems", []) if isinstance(row, dict)}
    full_rows: list[dict[str, Any]] = []
    full_json: list[dict[str, Any]] = []
    all_per_example: list[dict[str, Any]] = []
    full_status: dict[str, dict[str, Any]] = {}

    for path in sorted(generation_dir.glob("*.jsonl")):
        system_id = path.stem
        rows = load_standard_generation(path)
        ok, info = validate_full_coverage(rows, expected_ids)
        if not ok:
            continue
        display = DISPLAY_NAMES.get(system_id, gen_status_by_id.get(system_id, {}).get("display_name", system_id))
        table, per_rows, extra = metric_row(
            system_id,
            display,
            rows,
            canonical_by_id,
            eval_cfg,
            compute_bertscore=compute_bertscore,
            runnable=bool(gen_status_by_id.get(system_id, {}).get("runnable", True)),
            coverage=1.0,
        )
        full_rows.append(table)
        full_json.append({**extra, "coverage": info})
        all_per_example.extend(per_rows)
        full_status[system_id] = info

    full_rows.sort(key=lambda r: list(DISPLAY_NAMES.values()).index(r["Method"]) if r["Method"] in DISPLAY_NAMES.values() else 999)
    write_csv(output_dir / "system_metrics_full.csv", full_rows, fieldnames=METRIC_FIELDS)
    write_json(output_dir / "system_metrics_full.json", {"rows": full_json, "table_rows": full_rows})
    write_jsonl(output_dir / "per_example_metrics_full.jsonl", all_per_example)

    diagnostic_rows, artifact_status = evaluate_artifacts(
        cfg,
        root,
        examples,
        indices,
        eval_cfg,
        compute_bertscore=compute_bertscore,
    )
    diag_dir = root / "system_baseline/outputs/diagnostics"
    write_csv(diag_dir / "available_artifact_metrics.csv", diagnostic_rows)
    write_diagnostic_latex(diag_dir / "available_artifact_table.tex", diagnostic_rows)
    write_json(diag_dir / "available_artifact_status.json", artifact_status)
    write_coverage_report(
        output_dir / "coverage_report.md",
        root / args.input,
        examples,
        generation_status,
        full_status,
        artifact_status,
        cfg,
        root,
    )
    print(f"Full coverage systems: {len(full_rows)}")
    print(f"Wrote {relpath(output_dir / 'system_metrics_full.csv', root)}")
    print(f"Wrote {relpath(output_dir / 'coverage_report.md', root)}")


if __name__ == "__main__":
    main()
