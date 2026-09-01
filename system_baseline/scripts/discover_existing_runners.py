from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from eval_utils import DISPLAY_NAMES, METHOD_ORDER, ensure_system_dirs, rel, write_json


KEYWORDS = re.compile(r"(generate|infer|inference|predict|run_eval|run_generation|test|decode|human_eval)", re.I)

SEARCH_DIRS = [
    "baseline",
    "baseline/APTNESS",
    "baseline/CASE",
    "baseline/Sibyl",
    "baseline/EmPO",
    "Mistral-7B-Instruct-v0.3",
    "latent_stance_control",
    "runs",
    "scripts",
]

KNOWN_HINTS: dict[str, dict[str, Any]] = {
    "ours": {
        "runner_candidates": [
            "latent_stance_control/generate_and_rerank.py",
            "baseline_human/scripts/08_generate_ours_length_controlled.py",
        ],
        "checkpoint_candidates": [
            "runs/main/generator_gold",
            "runs/main/stance_fp32",
        ],
        "notes": "Main deployable EmoStance setting: predicted control with control-consistency reranking when available.",
    },
    "llm_only": {
        "runner_candidates": ["baseline_human/scripts/01_prepare_candidates.py"],
        "checkpoint_candidates": ["Mistral-7B-Instruct-v0.3"],
        "notes": "Prompt-only local Mistral baseline; no fine-tuned checkpoint.",
    },
    "llm_prompt": {
        "runner_candidates": ["baseline_human/scripts/01_prepare_candidates.py"],
        "checkpoint_candidates": ["Mistral-7B-Instruct-v0.3"],
        "notes": "Empathy-prompt local Mistral baseline; no fine-tuned checkpoint.",
    },
    "llm_sft": {
        "runner_candidates": ["baseline/EmPO/scripts/06_generate.py"],
        "checkpoint_candidates": ["baseline/EmPO/outputs/EmPO-SFT"],
        "notes": "Original EmPO-SFT checkpoint is reported as LLM-SFT.",
    },
    "empo_dpo": {
        "runner_candidates": ["baseline/EmPO/scripts/06_generate.py"],
        "checkpoint_candidates": ["baseline/EmPO/outputs/EmPO-DPO"],
        "notes": "EmPO DPO baseline.",
    },
    "case": {
        "runner_candidates": [
            "baseline/CASE/scripts/run_case_baseline.sh",
            "baseline/CASE/scripts/export_case_generations.py",
            "baseline/CASE/scripts/run_case_sample_eval.py",
        ],
        "checkpoint_candidates": ["baseline/CASE/outputs/case_baseline"],
        "notes": "CASE baseline runner and exported generations were found.",
    },
    "aptness": {
        "runner_candidates": [
            "baseline/APTNESS/evaluate.py",
            "baseline/APTNESS/aptrag_mistral_baseline/run_apt_rag.py",
        ],
        "checkpoint_candidates": ["baseline_human/data/raw_generations/aptness/apt_response_index"],
        "notes": "APTNESS/APT-RAG artifacts and partial generated outputs were found.",
    },
    "sibyl": {
        "runner_candidates": [
            "baseline/Sibyl/batch_generate_4cci_ED.py",
            "baseline/Sibyl/batch_generate_llama3_ED.py",
            "baseline/Sibyl/local_mistral_baseline/scripts/run_sample.sh",
        ],
        "checkpoint_candidates": ["baseline/Sibyl/local_mistral_baseline/outputs/response_generator_lora"],
        "notes": "Sibyl local Mistral baseline artifacts were found when present.",
    },
}


def first_existing(root: Path, candidates: list[str]) -> str | None:
    for item in candidates:
        path = root / item
        if path.exists():
            return rel(path, root)
    return None


def scan_keyword_files(root: Path) -> list[str]:
    hits: list[str] = []
    for dirname in SEARCH_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt"}:
                continue
            if KEYWORDS.search(path.name):
                hits.append(rel(path, root))
    return sorted(set(hits))


def discover(root: Path) -> dict[str, Any]:
    keyword_hits = scan_keyword_files(root)
    methods: dict[str, Any] = {}
    for method_id in METHOD_ORDER:
        hint = KNOWN_HINTS[method_id]
        runner = first_existing(root, hint["runner_candidates"])
        checkpoint = first_existing(root, hint["checkpoint_candidates"])
        status = "found" if runner or checkpoint else "missing"
        reason = "" if status == "found" else "No executable generation script or checkpoint found."
        methods[method_id] = {
            "method_id": method_id,
            "display_name": DISPLAY_NAMES[method_id],
            "status": status,
            "runner": runner,
            "checkpoint": checkpoint,
            "notes": hint["notes"],
            "reason": reason,
        }
    return {
        "project_root": str(root),
        "searched_dirs": SEARCH_DIRS,
        "keyword_hits": keyword_hits,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover existing generation runners and checkpoints.")
    parser.add_argument("--project_root", default=".", help="Project root.")
    parser.add_argument("--output", default="system_baseline/logs/discovered_runners.json")
    args = parser.parse_args()

    ensure_system_dirs()
    root = Path(args.project_root).resolve()
    payload = discover(root)
    write_json(root / args.output, payload)
    print(f"Wrote {args.output}")
    for method_id in METHOD_ORDER:
        item = payload["methods"][method_id]
        print(f"{method_id}: {item['status']} runner={item['runner']} checkpoint={item['checkpoint']}")


if __name__ == "__main__":
    main()
