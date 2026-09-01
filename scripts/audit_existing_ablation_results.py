from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ablation_utils import (
    MAIN_RUN,
    append_command,
    extract_seed,
    md_table,
    read_json,
    read_jsonl,
    read_text,
    rel,
    sanitize,
    summarize_metric_object,
    write_json,
)


PRIORITY_PATHS = [
    "runs/main/report.md",
    "runs/main/main_results_summary/report.md",
    "runs/main/source_vector_feature_ablation/report.md",
    "runs/main/generator_control_eval_role_weight025_c7mix050_compare/report.md",
    "runs/main/human_eval_choice_20_gold_plan_18_32_seed20260430/human_summary/human_choice_summary.md",
]

RESULT_GLOBS = [
    "**/report.md",
    "**/*.json",
    "**/*.jsonl",
    "**/*.csv",
    "**/*.tsv",
    "**/*.tex",
    "**/human_summary/*.md",
    "**/generation*.jsonl",
    "**/metrics*.json",
    "**/eval*.json",
]

LEAKAGE_TERMS = [
    "target_response",
    "target response",
    "emoji annotation",
    "emoji_annotation",
    "emoji name",
    "emoji description",
    "original emotion",
    "stance cluster label",
    "stance vector",
    "gold target stance",
]


def infer_experiment(path: Path) -> str:
    parts = path.parts
    name = path.parent.name
    text = str(path).lower()
    if "source_vector_feature_ablation" in text or "srcvec" in text:
        return "continuous_vector"
    if "generator_control" in text or "rerank" in text or "generator_gold" in text:
        return "generation_control"
    if "human_eval" in text or "human_summary" in text:
        return "human_eval"
    if "stance_role_aware" in text or name in {"stance_fp32", "stance", "ablations", "ablations_role_aware"}:
        return "stance_prediction"
    if "c7_gate" in text:
        return "c7_gate"
    return parts[-2] if len(parts) > 1 else "unknown"


def infer_method(path: Path, obj: Any | None = None) -> str:
    text = str(path).lower()
    parent = path.parent.name
    if "stance_fp32" in text:
        return "text-only DeBERTa"
    if "stance_role_aware_weight025" in text:
        return "Full EmoStance stance predictor / role-aware weight025"
    if "stance_role_aware_no_focal" in text:
        return "role-aware no focal/class weighting variant"
    if "stance_role_aware" in text and "srcvec" not in text:
        return "role-aware stance predictor"
    if "source_vector_feature_ablation" in text:
        return "direct / none / prototype source-vector feature"
    if "generator_control_eval_512" in text:
        return "zero / shuffled / gold / predicted control"
    if "role_weight025_c7mix050" in text:
        return "role-aware predicted control c7mix050"
    if "role_weight025" in text:
        return "role-aware predicted control"
    if "rerank_c7mix050" in text:
        return "role-aware control + rerank / oracle selection"
    if "human_eval_choice_20" in text:
        return "20-context human pilot"
    if isinstance(obj, Mapping) and "method" in obj:
        return str(obj["method"])
    return parent


def infer_split(path: Path, obj: Any | None = None) -> str | None:
    text = str(path).lower()
    if "test" in text and "dev" not in text:
        return "test"
    if "dev" in text or "valid" in text:
        return "dev"
    if isinstance(obj, Mapping):
        splits = [key for key in ("test", "dev", "train") if key in obj]
        if splits:
            return "/".join(splits)
        if "splits" in obj:
            return str(obj["splits"])
    return None


def dataset_size(path: Path, obj: Any | None = None) -> int | None:
    text = str(path).lower()
    if "512" in text:
        return 512
    if "20" in text and "human_eval" in text:
        return 20
    if isinstance(obj, Mapping):
        for key in ("examples", "num_examples", "selected", "test_examples"):
            value = obj.get(key)
            if isinstance(value, int):
                return value
    return None


def input_config_status(path: Path, obj: Any | None = None, sample: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    text = str(path).lower()
    legal = "uncertain"
    notes: List[str] = []
    if "stance_fp32" in text or "stance_role_aware" in text:
        legal = "legal_context_only"
        notes.append("training script uses build_model_text(row): situation + context; role-aware model additionally consumes role/next_role/transition ids.")
    if "generator_control" in text or "rerank" in text:
        legal = "legal_context_plus_control"
        notes.append("generation prompt contains situation/context/role marker; control vectors are internal control conditions.")
        if "gold" in text or (sample and sample.get("control_type") == "gold"):
            notes.append("gold control is non-deployable upper-reference.")
    if "prepared" in text:
        legal = "contains_labels_for_training_or_evaluation"
        notes.append("prepared data stores weak labels/vectors; not itself a deployable inference input.")
    if sample:
        forbidden_in_prompt = False
        prompt = str(sample.get("prompt") or sample.get("input") or "")
        for term in LEAKAGE_TERMS:
            if term in prompt.lower():
                forbidden_in_prompt = True
                break
        if forbidden_in_prompt:
            legal = "leakage_risk"
            notes.append("prompt/input text appears to include a forbidden target/label term.")
        elif "prompt" in sample:
            notes.append("sample prompt inspected; no forbidden target/emoji/stance label term found.")
    return {"input_config": legal, "notes": "; ".join(notes) if notes else "configuration not explicit in artifact"}


def reuse_decision(path: Path, experiment: str, method: str, input_status: str, obj: Any | None = None) -> Dict[str, str]:
    text = str(path).lower()
    if input_status == "leakage_risk":
        return {"reuse_status": "not_reusable", "reason": "input contains possible target/label leakage"}
    if "smoke" in text:
        return {"reuse_status": "not_reusable", "reason": "smoke run; useful only for debugging"}
    if "train.jsonl" in text:
        return {"reuse_status": "not_reusable", "reason": "training split artifact, not evaluation"}
    if experiment == "human_eval":
        if "choice_20_gold_plan" in text and "human_summary" in text:
            return {"reuse_status": "reusable", "reason": "20-context pilot; qualitative evidence only, not main human-eval claim"}
        return {"reuse_status": "uncertain", "reason": "human-eval artifact; use only if protocol/config is verified"}
    if experiment in {"stance_prediction", "continuous_vector", "generation_control"}:
        if path.suffix in {".json", ".md", ".jsonl"}:
            return {"reuse_status": "reusable", "reason": "evaluation artifact with recognizable method/split; deployability handled per method"}
    if path.suffix in {".csv", ".tsv", ".tex"}:
        return {"reuse_status": "uncertain", "reason": "derived table; prefer upstream JSON/JSONL artifact"}
    return {"reuse_status": "uncertain", "reason": "method/config could not be mapped to required ablation table"}


def load_obj(path: Path) -> Any | None:
    try:
        if path.suffix == ".json":
            return read_json(path)
        if path.suffix == ".jsonl":
            return {"sample": (read_jsonl(path, limit=1) or [None])[0], "num_rows": sum(1 for _ in path.open("r", encoding="utf-8"))}
        if path.suffix in {".md", ".tex"}:
            return {"text_excerpt": read_text(path, limit_chars=20000)}
    except Exception as exc:
        return {"parse_error": str(exc)}
    return None


def audit_path(path: Path) -> Dict[str, Any]:
    obj = load_obj(path)
    sample = obj.get("sample") if isinstance(obj, Mapping) and isinstance(obj.get("sample"), Mapping) else None
    experiment = infer_experiment(path)
    method = infer_method(path, obj)
    status = input_config_status(path, obj, sample)
    decision = reuse_decision(path, experiment, method, status["input_config"], obj)
    metrics: Dict[str, Any] = {}
    if path.name.startswith("metrics") and isinstance(obj, Mapping):
        metrics = summarize_metric_object(obj)
    elif path.name in {"metrics.json", "report.md"} and isinstance(obj, Mapping):
        metrics = summarize_metric_object(obj)
    elif isinstance(obj, Mapping) and "num_rows" in obj:
        metrics = {"num_rows": obj.get("num_rows")}
    return sanitize(
        {
            "artifact_path": rel(path),
            "experiment_name": experiment,
            "method_name": method,
            "split": infer_split(path, obj),
            "dataset_size": dataset_size(path, obj),
            "seed": extract_seed(path, obj if isinstance(obj, Mapping) else None),
            "metrics": metrics,
            "decoding_config": extract_decoding_config(path),
            "input_config": status["input_config"],
            "reuse_status": decision["reuse_status"],
            "reason": decision["reason"] + (f"; {status['notes']}" if status["notes"] else ""),
        }
    )


def extract_decoding_config(path: Path) -> Dict[str, Any] | None:
    config = path.parent / "config.json"
    if not config.exists():
        return None
    try:
        obj = read_json(config)
    except Exception:
        return None
    keys = [
        "max_examples",
        "num_candidates",
        "do_sample",
        "temperature",
        "top_p",
        "min_words",
        "length_penalty",
        "base_model",
    ]
    found = {key: obj.get(key) for key in keys if key in obj}
    return found or None


def discover(runs_roots: List[Path]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for priority in PRIORITY_PATHS:
        path = Path(priority)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists() and path not in seen:
            paths.append(path)
            seen.add(path)
    for root in runs_roots:
        if not root.exists():
            continue
        for glob in RESULT_GLOBS:
            for path in root.glob(glob):
                if path.is_file() and path not in seen:
                    paths.append(path)
                    seen.add(path)
    return sorted(paths, key=lambda p: (0 if str(p) in PRIORITY_PATHS else 1, str(p)))


def write_inventory_md(path: Path, items: List[Dict[str, Any]]) -> None:
    rows = []
    for item in items:
        rows.append(
            [
                item["experiment_name"],
                item["method_name"],
                item.get("split") or "NA",
                item["reuse_status"],
                item["artifact_path"],
                item["reason"],
            ]
        )
    content = "# Result Inventory\n\n"
    content += md_table(["Experiment", "Method", "Split", "Reuse status", "Artifact", "Reason"], rows)
    content += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit existing ablation/evaluation artifacts.")
    parser.add_argument("--runs_root", action="append", default=[], help="Runs root to scan. Can be passed multiple times.")
    parser.add_argument("--out_dir", default="runs/emnlp_ablation_report")
    args = parser.parse_args()

    roots = [Path(p) for p in args.runs_root] or [MAIN_RUN.parent, Path("runs")]
    out_dir = Path(args.out_dir)
    append_command(out_dir, "python scripts/audit_existing_ablation_results.py --runs_root " + " --runs_root ".join(str(p) for p in roots) + f" --out_dir {out_dir}")
    paths = discover(roots)
    items = [audit_path(path) for path in paths]
    write_json(out_dir / "result_inventory.json", {"artifacts": items})
    write_inventory_md(out_dir / "result_inventory.md", items)


if __name__ == "__main__":
    main()
