from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from eval_utils import DISPLAY_NAMES, METHOD_ORDER, env_payload, load_yaml, write_json


MAIN_COLS = [
    ("bertscore_f1", "BERTScore"),
    ("rouge_1", "ROUGE-1"),
    ("rouge_2", "ROUGE-2"),
    ("rouge_l", "ROUGE-L"),
    ("bleu_1", "BLEU-1"),
    ("bleu_2", "BLEU-2"),
    ("bleu_3", "BLEU-3"),
    ("bleu_4", "BLEU-4"),
    ("meteor", "METEOR"),
]

DIVERSITY_COLS = [
    ("distinct_1", "Distinct-1"),
    ("distinct_2", "Distinct-2"),
    ("unique_response_ratio", "Unique Resp."),
    ("unique_start_ratio", "Unique Start"),
    ("self_bleu", "Self-BLEU"),
    ("generic_response_rate", "Generic"),
    ("repetition_rate", "Repetition"),
    ("empty_response_rate", "Empty"),
    ("role_marker_leakage_rate", "Role Leak"),
]

LATEX_DIVERSITY_COLS = [
    ("distinct_1", "Distinct-1"),
    ("distinct_2", "Distinct-2"),
    ("unique_response_ratio", "Unique Resp."),
    ("unique_start_ratio", "Unique Start"),
    ("self_bleu", "Self-BLEU"),
    ("generic_response_rate", "Generic"),
    ("repetition_rate", "Repetition"),
    ("role_marker_leakage_rate", "Role Leak"),
]

STANCE_COLS = [
    ("stance_ce", "CE"),
    ("stance_jsd", "JSD"),
    ("stance_accuracy", "Acc."),
    ("stance_macro_f1", "Macro-F1"),
    ("stance_entropy", "Entropy"),
]


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.load(path.open("r", encoding="utf-8"))
    return payload.get("metrics", payload) if isinstance(payload, dict) else {}


def fmt(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def visible_path(method_id: str, value: Any) -> str:
    text = str(value or "")
    if method_id == "llm_sft":
        text = text.replace("EmPO-SFT", "LLM-SFT")
    return text


def markdown_table(metrics: Mapping[str, Mapping[str, Any]], cols: list[tuple[str, str]]) -> list[str]:
    header = "| Method | " + " | ".join(markdown_col_label(label) for _, label in cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    lines = [header, sep]
    for method_id in METHOD_ORDER:
        row = metrics.get(method_id, {})
        lines.append("| " + DISPLAY_NAMES[method_id] + " | " + " | ".join(fmt(row.get(key)) for key, _ in cols) + " |")
    return lines


def markdown_col_label(label: str) -> str:
    up = {"BERTScore", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "Distinct-1", "Distinct-2", "Unique Resp.", "Unique Start", "Acc.", "Macro-F1"}
    down = {"Self-BLEU", "Generic", "Repetition", "Empty", "Role Leak", "CE", "JSD"}
    if label in up:
        return f"{label} ↑"
    if label in down:
        return f"{label} ↓"
    return label


def latex_table(metrics: Mapping[str, Mapping[str, Any]], cols: list[tuple[str, str]], caption: str, label: str, path: Path) -> None:
    align = "l" + "r" * len(cols)
    lines = [
        f"\\begin{{table}}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        "Method & " + " & ".join(col_label(label) for _, label in cols) + " \\\\",
        "\\midrule",
    ]
    for method_id in METHOD_ORDER:
        row = metrics.get(method_id, {})
        lines.append(DISPLAY_NAMES[method_id].replace("EmoStance / Ours", "Ours") + " & " + " & ".join(fmt(row.get(key)) for key, _ in cols) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def col_label(label: str) -> str:
    arrows = {
        "BERTScore": "$\\uparrow$",
        "ROUGE-1": "$\\uparrow$",
        "ROUGE-2": "$\\uparrow$",
        "ROUGE-L": "$\\uparrow$",
        "BLEU-1": "$\\uparrow$",
        "BLEU-2": "$\\uparrow$",
        "BLEU-3": "$\\uparrow$",
        "BLEU-4": "$\\uparrow$",
        "METEOR": "$\\uparrow$",
        "Distinct-1": "$\\uparrow$",
        "Distinct-2": "$\\uparrow$",
        "Unique Resp.": "$\\uparrow$",
        "Unique Start": "$\\uparrow$",
        "Self-BLEU": "$\\downarrow$",
        "Generic": "$\\downarrow$",
        "Repetition": "$\\downarrow$",
        "Empty": "$\\downarrow$",
        "Role Leak": "$\\downarrow$",
        "CE": "$\\downarrow$",
        "JSD": "$\\downarrow$",
        "Acc.": "$\\uparrow$",
        "Macro-F1": "$\\uparrow$",
    }
    return f"{label} {arrows.get(label, '')}".strip()


def significance_markdown(rows: list[dict[str, Any]], limit: int = 80) -> list[str]:
    lines = [
        "| Metric | Baseline | Common | Ours | Baseline | Diff | 95% CI | p |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows[:limit]:
        ci = f"[{fmt(row.get('ci_low'))}, {fmt(row.get('ci_high'))}]"
        lines.append(
            f"| {row.get('metric')} | {row.get('baseline')} | {row.get('common_examples')} | "
            f"{fmt(row.get('ours_mean'))} | {fmt(row.get('baseline_mean'))} | {fmt(row.get('diff'))} | {ci} | {fmt(row.get('p_value'))} |"
        )
    return lines


def significance_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Metric & Baseline & Ours & Baseline & Diff & 95\\% CI & $p$ \\\\",
        "\\midrule",
    ]
    for row in rows[:40]:
        ci = f"[{fmt(row.get('ci_low'))}, {fmt(row.get('ci_high'))}]"
        lines.append(f"{row.get('metric')} & {row.get('baseline')} & {fmt(row.get('ours_mean'))} & {fmt(row.get('baseline_mean'))} & {fmt(row.get('diff'))} & {ci} & {fmt(row.get('p_value'))} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Paired bootstrap significance comparing Ours against each baseline.}",
        "\\label{tab:system_significance}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def llm_judge_placeholder(path: Path) -> None:
    path.write_text(
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Method & Empathy $\\uparrow$ & Informativity $\\uparrow$ & Fluency $\\uparrow$ & Context Consistency $\\uparrow$ \\\\\n"
        "\\midrule\n"
        "\\multicolumn{5}{l}{Not run; supplementary diagnostic only.} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Optional LLM-as-judge diagnostics. Context Consistency is defined as concise, clear, relevant, and consistent with the dialogue history.}\n"
        "\\label{tab:llm_judge_optional}\n"
        "\\end{table}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build system-level evaluation report and LaTeX tables.")
    parser.add_argument("--metrics_dir", default="system_baseline/metrics")
    parser.add_argument("--logs_dir", default="system_baseline/logs")
    parser.add_argument("--output_dir", default="system_baseline/reports")
    parser.add_argument("--config", default="system_baseline/configs/system_eval.yaml")
    args = parser.parse_args()

    root = Path(".").resolve()
    metrics_dir = root / args.metrics_dir
    logs_dir = root / args.logs_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(metrics_dir / "per_method_metrics.json")
    significance = load_csv_rows(metrics_dir / "pairwise_significance.csv")
    significance_json_path = metrics_dir / "pairwise_significance.json"
    significance_payload = json.load(significance_json_path.open("r", encoding="utf-8")) if significance_json_path.exists() else {}
    generation_status = {}
    status_path = logs_dir / "generation_status.json"
    if status_path.exists():
        generation_status = json.load(status_path.open("r", encoding="utf-8"))
    discovered_path = logs_dir / "discovered_runners.json"
    discovered = json.load(discovered_path.open("r", encoding="utf-8")) if discovered_path.exists() else {}
    cfg = load_yaml(root / args.config)
    env = env_payload(root)
    bert_values = [metrics.get(m, {}).get("bertscore_f1") for m in METHOD_ORDER]
    bert_available = any(v not in (None, "", "NA") for v in bert_values)
    bert_note = (
        f"BERTScore-F1 was computed with `{cfg.get('evaluation', {}).get('bertscore_model_type', 'distilbert-base-uncased')}` "
        f"(num_layers={cfg.get('evaluation', {}).get('bertscore_num_layers', 6)})."
        if bert_available
        else "In this run, the optional BERTScore dependency is unavailable, so the BERTScore column is reported as NA rather than replaced with a proxy metric."
    )

    latex_table(metrics, MAIN_COLS, "Main automatic system metrics on the ED test set.", "tab:main_system_metrics", out_dir / "table_main_system_metrics.tex")
    latex_table(metrics, LATEX_DIVERSITY_COLS, "Diversity and degeneration diagnostics.", "tab:diversity_degen", out_dir / "table_diversity_and_degen.tex")
    latex_table(metrics, STANCE_COLS, "Stance consistency diagnostics when reusable generated-stance scores are available.", "tab:stance_diagnostics", out_dir / "table_stance_diagnostics.tex")
    significance_latex(significance, out_dir / "table_significance.tex")
    llm_judge_placeholder(out_dir / "table_llm_judge_optional.tex")

    methods_table = [
        "| method_id | display name | runner | checkpoint | status |",
        "|---|---|---|---|---|",
    ]
    status_by_method = {m.get("method_id"): m for m in generation_status.get("methods", []) if isinstance(m, dict)}
    discovered_methods = discovered.get("methods", {}) if isinstance(discovered, dict) else {}
    for method_id in METHOD_ORDER:
        st = status_by_method.get(method_id, {})
        disc = discovered_methods.get(method_id, {})
        methods_table.append(
            f"| {method_id} | {DISPLAY_NAMES[method_id]} | `{visible_path(method_id, st.get('runner') or disc.get('runner') or '')}` | "
            f"`{visible_path(method_id, st.get('checkpoint') or disc.get('checkpoint') or '')}` | {st.get('status') or disc.get('status') or 'unknown'} |"
        )

    lines = [
        "# System-Level Baseline Evaluation on ED Test Set",
        "",
        "## 1. Evaluation Setup",
        "",
        "This system-level evaluation uses the EmpatheticDialogues test set / prepared ED test set and standardizes existing project generations. No model is retrained, no baseline algorithm is reimplemented, and all methods use existing project runners or previously generated outputs from those runners.",
        "",
        "Model inputs are restricted to situation, dialogue history up to the current turn, and speaker-role markers. Gold target responses, original ED emotion labels, emoji annotations, stance clusters, stance vectors, and oracle target-side labels are not used as model inputs.",
        "",
        "Response length is controlled only by the shared maximum decoding length where generation is run. Explicit length statistics are not used as main evaluation metrics. Following CASE and TOOL-ED, we report diversity and degeneration diagnostics instead.",
        "",
        f"Decoding config: `{json.dumps(cfg.get('generation', {}), ensure_ascii=False)}`",
        "",
        "## 2. Methods",
        "",
        *methods_table,
        "",
        "## 3. Main Automatic Metrics",
        "",
        "BERTScore is treated as the main semantic similarity sanity-check when the optional package is available. ROUGE, BLEU, and METEOR are reference-based comparability metrics and cannot by themselves prove empathy quality.",
        bert_note,
        "",
        *markdown_table(metrics, MAIN_COLS),
        "",
        "## 4. Diversity and Degeneration Diagnostics",
        "",
        "Distinct-1/2 are diversity diagnostics, not length metrics. Empty, too-short, repetition, generic response, role-marker leakage, multi-turn leakage, and context-copy rates are system quality diagnostics.",
        "",
        *markdown_table(metrics, DIVERSITY_COLS),
        "",
        "## 5. Stance Consistency Diagnostics",
        "",
        "These metrics evaluate stance consistency rather than human-perceived response quality. Stance diagnostics are reported only when reusable generated-stance scores are already present; no new evaluator is trained here.",
        "",
        *markdown_table(metrics, STANCE_COLS),
        "",
        "## 6. Optional LLM-as-Judge Diagnostics",
        "",
        "LLM-as-judge results are supplementary diagnostics and are not used as a replacement for human evaluation. This optional diagnostic was not run in this offline pass. Context Consistency is defined as concise, clear, relevant, and consistent with the dialogue history.",
        "",
        "## 7. Paired Bootstrap Significance",
        "",
        str(significance_payload.get("note") or "Paired bootstrap uses example_id as the resampling unit."),
        "",
        *significance_markdown(significance),
        "",
        "## 8. Interpretation",
        "",
        "Do not draw conclusions from BLEU/ROUGE alone. BERTScore, Distinct, degeneration diagnostics, and stance diagnostics should be interpreted separately. Automatic generation metrics are system-level diagnostics; human evaluation should remain the final quality judgment.",
        "",
        "## 9. Missing or Skipped Methods",
        "",
    ]
    missing_path = out_dir / "missing_or_skipped_methods.md"
    if missing_path.exists():
        lines.extend(missing_path.read_text(encoding="utf-8").splitlines())
    else:
        lines.append("No missing/skipped-method report was found.")
    lines += [
        "",
        "## 10. Reproducibility",
        "",
        f"- git commit: `{env.get('git_commit')}`",
        f"- python: `{env.get('python')}`",
        f"- python executable: `{env.get('python_executable')}`",
        f"- optional dependencies: `{json.dumps(env.get('optional_dependencies', {}), ensure_ascii=False)}`",
        f"- commands log: `{args.logs_dir}/commands.log`",
        f"- random seed: `{cfg.get('generation', {}).get('seed')}`",
        "",
    ]
    (out_dir / "system_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "system_eval_report.tex").write_text(
        "\\input{table_main_system_metrics.tex}\n"
        "\\input{table_diversity_and_degen.tex}\n"
        "\\input{table_stance_diagnostics.tex}\n"
        "\\input{table_llm_judge_optional.tex}\n"
        "\\input{table_significance.tex}\n",
        encoding="utf-8",
    )
    write_json(out_dir / "system_eval_report_payload.json", {"metrics": metrics, "significance": significance})
    print(f"Wrote {out_dir / 'system_eval_report.md'}")
    print(f"Wrote {out_dir / 'table_main_system_metrics.tex'}")


if __name__ == "__main__":
    main()
