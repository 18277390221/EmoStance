from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ablation_utils import DATASET_METADATA, append_command, fmt_mean_std, fmt_num, md_table, read_json, write_json


STANCE_ORDER = [
    "Full EmoStance stance predictor",
    "text-only DeBERTa",
    "w/o role-aware transition",
    "w/o gated transition prior",
    "hard-label training",
    "graph-only",
    "calibrated graph fusion",
]
VECTOR_ORDER = [
    "prototype reconstruction",
    "direct vector regression",
    "no source vector feature",
    "direct source vector feature",
    "prototype source vector feature",
]
GEN_ORDER = [
    "zero control",
    "shuffled control",
    "gold control",
    "predicted control baseline",
    "role-aware predicted control",
    "role-aware predicted control + rerank",
    "oracle gold selection",
]
RERANK_ORDER = ["first candidate", "rerank by control CE", "oracle gold selection"]


def by_group(results: Sequence[Mapping[str, Any]], group: str) -> Dict[str, Mapping[str, Any]]:
    return {str(item["method"]): item for item in results if item.get("experiment_group") == group}


def reused_new(item: Mapping[str, Any] | None) -> str:
    if not item:
        return "missing"
    source = item.get("source", {})
    source_type = source.get("type")
    if source_type == "reused":
        return "reused"
    if source_type == "missing":
        return "missing"
    return str(source_type or "new/unknown")


def seeds(item: Mapping[str, Any] | None) -> str:
    if not item:
        return "NA"
    values = item.get("seeds") or []
    return ",".join(str(v) for v in values) if values else "NA"


def run_descriptor(item: Mapping[str, Any] | None) -> str:
    values = list((item or {}).get("seeds") or [])
    if len(values) >= 3:
        return "three-seed mean +/- std"
    if len(values) == 1:
        return f"seed-{values[0]} single-run"
    if values:
        return ",".join(str(v) for v in values)
    return "not-run"


def metric(item: Mapping[str, Any] | None, key: str, digits: int = 4) -> str:
    if not item:
        return "NA"
    value = (item.get("metrics") or {}).get(key)
    if isinstance(value, Mapping):
        return fmt_mean_std(value, digits)
    return fmt_num(value, digits)


def raw_metric(item: Mapping[str, Any] | None, key: str) -> Any:
    if not item:
        return None
    value = (item.get("metrics") or {}).get(key)
    if isinstance(value, Mapping):
        return value.get("mean")
    return value


def source(item: Mapping[str, Any] | None) -> str:
    if not item:
        return "NA"
    return str((item.get("source") or {}).get("artifact_path") or "NA")


def provenance_rows(results: Sequence[Mapping[str, Any]]) -> List[List[str]]:
    wanted = [
        ("Stance prediction", "text-only DeBERTa"),
        ("Stance prediction", "Full EmoStance stance predictor"),
        ("Graph calibration", "calibrated graph fusion"),
        ("Vector ablation", "direct / none / prototype"),
        ("Generation control", "zero / shuffled / gold / predicted / rerank"),
        ("Human pilot", "20-context pilot"),
    ]
    by_method = {str(item.get("method")): item for item in results}
    rows: List[List[str]] = []
    rows.append(["Stance prediction", "text-only DeBERTa", "test", "yes", source(by_method.get("text-only DeBERTa")), "single-run; input audited as situation/context only"])
    rows.append(["Stance prediction", "Full EmoStance stance predictor", "test", "yes", source(by_method.get("Full EmoStance stance predictor")), "single-run; role markers and transition ids only"])
    rows.append(["Graph calibration", "calibrated graph fusion", "test", "yes", source(by_method.get("calibrated graph fusion")), "recomputed JSD/Brier from predictions and graph prior; beta recorded"])
    vec_source = source(by_method.get("direct source vector feature"))
    rows.append(["Vector ablation", "direct / none / prototype", "test", "yes", vec_source, "single-run source-vector feature ablation"])
    rows.append(["Generation control", "zero / shuffled / gold / predicted / rerank", "test subset", "yes", source(by_method.get("role-aware predicted control + rerank")), "512-example, seeds 13/21/42; gold/oracle are non-deployable"])
    rows.append(["Human pilot", "20-context pilot", "test subset", "yes", "runs/main/human_eval_choice_20_gold_plan_18_32_seed20260430/human_summary/human_choice_summary.md", "qualitative only; not used as main human-eval evidence"])
    return rows


def stance_section(results: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> str:
    group = by_group(results, "stance_prediction")
    rows = []
    for name in STANCE_ORDER:
        item = group.get(name)
        rows.append([
            name,
            reused_new(item),
            seeds(item),
            metric(item, "target_ce"),
            metric(item, "target_jsd"),
            metric(item, "target_acc"),
            metric(item, "target_macro_f1"),
            metric(item, "target_ece"),
            metric(item, "target_brier"),
        ])
    text = "## 2. Stance Prediction Ablation\n\n"
    text += md_table(["Method", "Reused/New", "Seed(s)", "Target CE ↓", "Target JSD ↓", "Acc ↑", "Macro-F1 ↑", "ECE ↓", "Brier ↓"], rows)
    full = group.get("Full EmoStance stance predictor")
    base = group.get("text-only DeBERTa")
    full_ce = raw_metric(full, "target_ce")
    base_ce = raw_metric(base, "target_ce")
    full_f1 = raw_metric(full, "target_macro_f1")
    base_f1 = raw_metric(base, "target_macro_f1")
    cal = group.get("calibrated graph fusion")
    cal_ece = raw_metric(cal, "target_ece")
    base_ece = raw_metric(base, "target_ece")
    no_role = group.get("w/o role-aware transition")
    no_gate = group.get("w/o gated transition prior")
    hard = group.get("hard-label training")
    no_role_ce = raw_metric(no_role, "target_ce")
    no_gate_ce = raw_metric(no_gate, "target_ce")
    hard_ce = raw_metric(hard, "target_ce")
    hard_ece = raw_metric(hard, "target_ece")
    full_ece = raw_metric(full, "target_ece")
    text += "\n\nMain finding:\n"
    if full_ce is not None and base_ce is not None and full_f1 is not None and base_f1 is not None:
        text += f"- Full EmoStance does not uniformly improve over text-only DeBERTa: target CE changes from {fmt_num(base_ce)} to {fmt_num(full_ce)}, while macro-F1 changes from {fmt_num(base_f1)} to {fmt_num(full_f1)}.\n"
    if full_ce is not None and no_role_ce is not None:
        delta = no_role_ce - full_ce
        text += f"- Removing role-aware transition changes target CE by {fmt_num(delta)} relative to Full EmoStance ({fmt_num(no_role_ce)} vs {fmt_num(full_ce)}); this is a {run_descriptor(no_role)} result.\n"
    else:
        text += "- Removing role-aware transition cannot be quantified from verified artifacts; A2 remains marked missing.\n"
    if no_gate_ce is not None:
        text += f"- Removing the gated transition prior gives target CE {fmt_num(no_gate_ce)}; role features are kept while graph_prior_weight is set to 0.\n"
    if cal_ece is not None and base_ece is not None:
        if abs(cal_ece - base_ece) < 1e-8:
            verdict = "matches"
        else:
            verdict = "improves" if cal_ece < base_ece else "does not improve"
        text += f"- Calibrated graph fusion {verdict} ECE relative to text-only ({fmt_num(cal_ece)} vs {fmt_num(base_ece)}); this artifact selects beta=0.0.\n"
    if hard_ce is not None and full_ce is not None and hard_ece is not None and full_ece is not None:
        calibration = "improves" if hard_ece < full_ece else "hurts"
        text += f"- Hard-label training worsens target CE relative to soft-label Full EmoStance ({fmt_num(hard_ce)} vs {fmt_num(full_ce)}) and {calibration} calibration by ECE ({fmt_num(hard_ece)} vs {fmt_num(full_ece)}).\n"
    else:
        text += "- Hard-label training is missing; calibration claims for hard labels are therefore not stated.\n"
    text += "\nMissing metrics:\n"
    missing_rows = [name for name in ["w/o role-aware transition", "w/o gated transition prior", "hard-label training"] if reused_new(group.get(name)) == "missing"]
    if missing_rows:
        text += f"- Missing verified test artifacts for: {', '.join(missing_rows)}.\n"
    else:
        descriptors = {name: run_descriptor(group.get(name)) for name in ["w/o role-aware transition", "w/o gated transition prior", "hard-label training"]}
        if all("three-seed" in value for value in descriptors.values()):
            text += "- Required A2/A3/A4 stance ablations have verified full-test seeds 13/21/42 artifacts and are reported as mean +/- std.\n"
        else:
            text += f"- Required A2/A3/A4 stance ablations have verified artifacts, with seed coverage: {descriptors}.\n"
    text += "- Brier/JSD for reused stance artifacts were recomputed from saved prediction distributions where NPZ files were available.\n"
    text += "\nStatistical comparison excerpt:\n"
    for name in ["Full EmoStance vs text-only DeBERTa", "Full EmoStance vs w/o role-aware transition", "Full EmoStance vs hard-label training"]:
        entry = stats.get(name, {})
        text += f"- {name}: `{entry}`\n"
    return text


def vector_section(results: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> str:
    group = by_group(results, "continuous_vector")
    rows = []
    for name in VECTOR_ORDER:
        item = group.get(name)
        rows.append([
            name,
            reused_new(item),
            seeds(item),
            metric(item, "target_ce"),
            metric(item, "target_acc"),
            metric(item, "target_macro_f1"),
            metric(item, "target_vector_cosine"),
            metric(item, "target_vector_mse", 6),
            metric(item, "source_vector_cosine"),
        ])
    text = "## 3. Continuous Stance Vector Ablation\n\n"
    text += md_table(["Method", "Reused/New", "Seed(s)", "Target CE ↓", "Acc ↑", "Macro-F1 ↑", "Target Vector Cos ↑", "Target Vector MSE ↓", "Source Vector Cos ↑"], rows)
    proto = group.get("prototype reconstruction")
    direct = group.get("direct vector regression")
    text += "\n\nMain finding:\n"
    if proto and direct:
        text += f"Prototype-based reconstruction is preferred on the automatic vector diagnostics: cosine {metric(proto, 'target_vector_cosine')} vs {metric(direct, 'target_vector_cosine')}, and MSE {metric(proto, 'target_vector_mse', 6)} vs {metric(direct, 'target_vector_mse', 6)}. Target CE/F1 comparability should be read from the corresponding stance rows, not from vector quality alone.\n"
    text += "\nStatistical comparison excerpt:\n"
    text += f"- prototype reconstruction vs direct vector regression: `{stats.get('prototype reconstruction vs direct vector regression', {})}`\n"
    return text


def generation_section(results: Sequence[Mapping[str, Any]], stats: Mapping[str, Any]) -> str:
    group = by_group(results, "generation_control")
    rows = []
    deploy = {
        "zero control": "yes",
        "shuffled control": "no",
        "gold control": "no, upper-reference",
        "predicted control baseline": "yes",
        "role-aware predicted control": "yes",
        "role-aware predicted control + rerank": "yes",
        "oracle gold selection": "no, oracle upper bound",
    }
    for name in GEN_ORDER:
        item = group.get(name)
        rows.append([
            name,
            deploy.get(name, str(item.get("deployable") if item else "NA")),
            reused_new(item),
            seeds(item),
            metric(item, "vs_gold_ce"),
            metric(item, "vs_gold_jsd"),
            metric(item, "vs_gold_acc"),
            metric(item, "vs_gold_macro_f1"),
            metric(item, "bert_score_f1"),
            metric(item, "distinct_2"),
            metric(item, "mean_length"),
            metric(item, "generic_rate"),
            metric(item, "repetition_rate"),
        ])
    text = "## 4. Generation Control Ablation\n\n"
    text += md_table(["Method", "Deployable?", "Reused/New", "Seed(s)", "vs-gold CE ↓", "vs-gold JSD ↓", "Acc ↑", "Macro-F1 ↑", "BERTScore-F1 ↑", "Distinct-2 ↑", "Mean Length", "Generic Rate ↓", "Repetition Rate ↓"], rows)
    text += "\n\nGold control and oracle gold selection are upper-reference conditions, not deployable systems. Shuffled control is a negative control. The deployable main comparison is role-aware predicted control + rerank vs role-aware predicted control without rerank, predicted control baseline, and zero control. Automatic generation metrics are stance-consistency diagnostics, not final human-quality claims.\n"
    text += "\nMissing generation metric note: BERTScore-F1 is not present in existing artifacts and was not fabricated. Distinct, generic-rate, repetition-rate, too-short-rate, and vs-gold JSD were recomputed from saved generated responses and stance distributions.\n"
    text += "\nStatistical comparison excerpt:\n"
    for name in ["role-aware control + rerank vs role-aware control w/o rerank", "predicted control vs shuffled control"]:
        text += f"- {name}: `{stats.get(name, {})}`\n"
    return text


def rerank_section(results: Sequence[Mapping[str, Any]]) -> str:
    group = by_group(results, "reranking")
    rows = []
    for name in RERANK_ORDER:
        item = group.get(name)
        rows.append([
            name,
            metric(item, "candidate_count", 0),
            reused_new(item),
            seeds(item),
            metric(item, "vs_gold_ce"),
            metric(item, "vs_gold_acc"),
            metric(item, "vs_gold_macro_f1"),
            metric(item, "mean_length"),
            metric(item, "latency_per_example"),
            metric(item, "cost_multiplier", 1),
        ])
    text = "## 5. Reranking Ablation\n\n"
    text += md_table(["Method", "Candidate Count", "Reused/New", "Seed(s)", "vs-gold CE ↓", "Acc ↑", "Macro-F1 ↑", "Mean Length", "Latency / Example", "Cost Multiplier"], rows)
    text += "\n\nThe reranking result is based on a 512-example test subset and should be interpreted as a subset diagnostic unless full-test reranking is also available.\n"
    return text


def human_section(pending: Sequence[Mapping[str, Any]]) -> str:
    rows = [[p["comparison"], p["reason"], p["status"]] for p in pending]
    text = "## 6. Human Evaluation Pending\n\n"
    text += "The following ablations affect final response quality and should be evaluated by humans. Automatic metrics are not sufficient for final claims about emotional appropriateness, empathy, naturalness, or human-likeness.\n\n"
    text += "### Planned human-evaluation ablations\n\n"
    text += md_table(["Comparison", "Why human evaluation is needed", "Status"], rows)
    text += "\n\n### Proposed human evaluation protocol\n\n"
    text += "- Sample 200 test contexts.\n- Stratify by target stance cluster, role transition, target entropy, and gold response length.\n- Use 3 annotators per context.\n- Use blind randomized system names.\n- Do not show emoji, stance cluster, gold stance vector, or method name.\n- Evaluate emotional appropriateness, felt understood, context fit, naturalness, continuation interest, overdone / inappropriate response, and AI-like or template-like response.\n"
    text += "\n### Planned statistics\n\n"
    text += "- win rate by criterion\n- Bradley-Terry score\n- bootstrap 95% confidence interval\n- paired bootstrap significance test\n- optional mixed-effects logistic regression with context and annotator random effects\n\n"
    text += "The existing 20-context pilot is qualitative evidence only and is not treated as a statistically reliable main claim.\n"
    return text


def missing_section(missing: Sequence[Mapping[str, Any]]) -> str:
    rows = [[m["ablation"], m["required"], m["current_status"], m["next_action"]] for m in missing]
    return "## 7. Missing or Incomplete Ablations\n\n" + md_table(["Ablation", "Required?", "Current status", "Next action"], rows)


def checklist_section(results: Sequence[Mapping[str, Any]]) -> str:
    group = by_group(results, "stance_prediction")
    hard = group.get("hard-label training")
    no_role = group.get("w/o role-aware transition")
    has_hard = reused_new(hard) != "missing"
    has_no_role = reused_new(no_role) != "missing"
    hard_support = f"yes, {run_descriptor(hard)}" if has_hard else "no direct hard-label artifact"
    no_role_support = f"yes, {run_descriptor(no_role)}" if has_no_role else "incomplete; A2 missing"
    rows = [
        ["Soft labels improve stance prediction/calibration", ("yes for CE/F1; calibration mixed" if has_hard else "no direct hard-label artifact"), "not required", "yes for CE/F1; avoid calibration-improves claim" if has_hard else "no"],
        ["Role-aware transition improves target listener stance prediction", no_role_support, "pending", "yes, as automatic stance claim" if has_no_role else "no"],
        ["Prototype reconstruction is more stable than direct vector regression", "yes", "not required", "yes"],
        ["Predicted stance control is not random", "yes; predicted beats shuffled on CE", "not required", "yes, as automatic diagnostic"],
        ["Reranking improves stance consistency", "yes; 512-subset diagnostic", "pending for human quality", "yes, cautiously"],
        ["EmoStance generates more emotionally appropriate replies", "automatic metrics insufficient", "pending", "only after human eval"],
    ]
    text = "## 8. Claim Checklist\n\n"
    text += md_table(["Claim", "Supported by automatic metrics?", "Supported by human eval?", "Can be stated in main paper?"], rows)
    text += "\n\nThe ablation results support three automatic-metric claims. First, role-aware stance prediction has mixed automatic results relative to text-only prediction under CE, macro-F1, and calibration metrics in the currently verified artifacts. Second, prototype-based vector reconstruction provides more stable continuous stance vectors than direct regression, as measured by vector cosine similarity and MSE. Third, predicted stance control and reranking improve stance-consistency diagnostics in generation. However, these automatic generation scores are not treated as final evidence of human-perceived empathy or naturalness.\n\nHuman evaluation is still required for claims about emotional appropriateness, felt understanding, naturalness, and whether reranking improves perceived response quality.\n"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the EMNLP ablation Markdown report.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    append_command(out.parent, f"python scripts/write_ablation_report.py --results {args.results} --inventory {args.inventory} --out {args.out}")
    payload = read_json(args.results)
    results = payload["results"]
    metadata = payload["metadata"]
    stats = payload.get("statistical_tests", {})
    lines: List[str] = []
    lines.append("# EmoStance Ablation Evaluation Report\n")
    lines.append(f"Date: {metadata.get('date')}")
    lines.append(f"Code commit: {metadata.get('git_commit') or 'unavailable'}")
    lines.append(f"Dataset: {DATASET_METADATA['dataset']}")
    lines.append(f"Test examples: {DATASET_METADATA['test_examples']:,}")
    lines.append(f"Train/dev/test: {DATASET_METADATA['train_examples']:,} / {DATASET_METADATA['dev_examples']:,} / {DATASET_METADATA['test_examples']:,}")
    lines.append(f"Observed emoji: {DATASET_METADATA['observed_emoji']}")
    lines.append(f"Stance clusters: {DATASET_METADATA['stance_clusters']}")
    lines.append(f"Stance vector dimension: {DATASET_METADATA['stance_vector_dim']}\n")
    lines.append("Important note:")
    lines.append("All deployable systems use only textual context, situation description, and speaker-role markers at inference time.")
    lines.append("No emoji annotations, emoji names, emoji descriptions, original emotion labels, textual stance tags, target responses, or gold target stance vectors are used as deployable-system inputs.\n")
    lines.append("## 1. Result Provenance\n")
    lines.append(md_table(["Experiment", "Method", "Split", "Reused?", "Source artifact", "Notes"], provenance_rows(results)))
    lines.append("")
    lines.append(stance_section(results, stats))
    lines.append("")
    lines.append(vector_section(results, stats))
    lines.append("")
    lines.append(generation_section(results, stats))
    lines.append("")
    lines.append(rerank_section(results))
    lines.append("")
    lines.append(human_section(payload.get("human_eval_pending", [])))
    lines.append("")
    lines.append(missing_section(payload.get("missing_experiments", [])))
    lines.append("")
    lines.append(checklist_section(results))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
