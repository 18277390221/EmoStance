from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_utils import clean_text, ensure_system_dirs, history_from_any, serialize_history, write_jsonl


def load_baseline_human_contexts(root: Path) -> list[dict[str, Any]]:
    path = root / "baseline_human/data/ed_test_contexts.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row.get("context_id") or "")
            if not example_id or example_id in seen:
                continue
            seen.add(example_id)
            history = history_from_any(row.get("history") or row.get("serialized_context"))
            source_role = history[-1]["role"] if history else ""
            source_text = history[-1]["text"] if history else ""
            rows.append(
                {
                    "example_id": example_id,
                    "dialogue_id": str(row.get("dialogue_id") or ""),
                    "turn_id": str(row.get("turn_id") or ""),
                    "situation": clean_text(row.get("situation", "")),
                    "history": history,
                    "source_role": source_role,
                    "target_role": str(row.get("target_role") or ""),
                    "gold_response": clean_text(row.get("gold_response", "")),
                    "source_text": source_text,
                    "serialized_context": serialize_history(history),
                    "metadata": {
                        "source_path": "baseline_human/data/ed_test_contexts.jsonl",
                        "input_restriction": "Model inputs must exclude gold_response and metadata labels.",
                    },
                }
            )
    return rows


def load_csv_turn_contexts(path: Path) -> list[dict[str, Any]]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_dialogue[str(row.get("conv_id") or row.get("dialogue_id") or "")].append(row)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dialogue_id, turns in sorted(by_dialogue.items()):
        turns.sort(key=lambda r: int(str(r.get("utterance_idx") or 0)))
        history: list[dict[str, str]] = []
        for row in turns:
            utterance_idx = int(str(row.get("utterance_idx") or 0))
            speaker_idx = str(row.get("speaker_idx") or "")
            role = "A" if speaker_idx in {"0", "A", "speaker"} else "B"
            utterance = clean_text(row.get("utterance", ""))
            if history:
                example_id = f"{dialogue_id}_turn_{utterance_idx}"
                if example_id not in seen:
                    seen.add(example_id)
                    source_role = history[-1]["role"] if history else ""
                    rows.append(
                        {
                            "example_id": example_id,
                            "dialogue_id": dialogue_id,
                            "turn_id": str(utterance_idx),
                            "situation": clean_text(row.get("prompt", "")),
                            "history": list(history),
                            "source_role": source_role,
                            "target_role": role,
                            "gold_response": utterance,
                            "source_text": history[-1]["text"] if history else "",
                            "serialized_context": serialize_history(history),
                            "metadata": {
                                "source_path": str(path),
                                "input_restriction": "Model inputs must exclude gold_response and metadata labels.",
                            },
                        }
                    )
            if utterance:
                history.append({"role": role, "text": utterance})
    return rows


def build_inputs(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = load_baseline_human_contexts(root)
    if rows:
        return rows, "baseline_human/data/ed_test_contexts.jsonl"
    for candidate in [
        root / "empatheticdialogues/test.csv",
        root / "data/empatheticdialogues/test.csv",
    ]:
        if candidate.exists():
            return load_csv_turn_contexts(candidate), str(candidate)
    raise FileNotFoundError("Could not find an ED test split.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sanitized ED test inputs for system-level generation.")
    parser.add_argument("--project_root", default=".", help="Project root.")
    parser.add_argument("--output", default="system_baseline/data/ed_test_inputs.jsonl")
    args = parser.parse_args()

    ensure_system_dirs()
    root = Path(args.project_root).resolve()
    rows, source = build_inputs(root)
    write_jsonl(root / args.output, rows)
    duplicate_count = len(rows) - len({row["example_id"] for row in rows})
    print(f"Wrote {args.output}")
    print(f"Source: {source}")
    print(f"Examples: {len(rows)}")
    print(f"Duplicate example_id count: {duplicate_count}")


if __name__ == "__main__":
    main()
