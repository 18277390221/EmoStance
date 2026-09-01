from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from system_baseline.utils.io import PROJECT_ROOT, git_commit, load_yaml, read_json, relpath, write_json, write_jsonl
from system_baseline.utils.text import build_input_text, clean_text, context_key, history_from_context_list, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical ED test set for fair system evaluation.")
    parser.add_argument("--config", default="system_baseline/configs/system_eval.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source", default=None, help="Override ED test source path.")
    parser.add_argument("--output", default="system_baseline/data/ed_test_canonical.jsonl")
    parser.add_argument("--meta-output", default="system_baseline/data/ed_test_canonical.meta.json")
    return parser.parse_args()


def source_candidates(root: Path, cfg: dict[str, Any], override: str | None) -> list[Path]:
    if override:
        return [root / override]
    configured = cfg.get("data", {}).get("ed_test_source_candidates") or []
    if configured:
        return [root / str(path) for path in configured]
    return [
        root / "baseline/Sibyl/ED_data/test.json",
        root / "empatheticdialogues/test.csv",
        root / "pre_data/test.json",
        root / "runs/main/prepared/test.jsonl",
    ]


def build_from_sibyl_json(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        return []
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        context = item.get("context")
        target = clean_text(item.get("target") or item.get("gold_response") or item.get("response") or "")
        situation = clean_text(item.get("situation", ""))
        if not isinstance(context, list) or not target:
            continue
        history = history_from_context_list(context)
        if not history:
            continue
        target_speaker = "A" if len(history) % 2 == 0 else "B"
        turn_id = len(history) + 1
        dialogue_id = clean_text(item.get("dialogue_id") or "")
        raw_id = clean_text(item.get("example_id") or item.get("id") or "")
        example_id = raw_id or stable_hash([situation, history, target], prefix="ed")
        row = {
            "example_id": example_id,
            "dialogue_id": dialogue_id or stable_hash([situation, history[:1]], prefix="dialogue"),
            "turn_id": turn_id,
            "situation": situation,
            "history": history,
            "reference": target,
            "target_speaker": target_speaker,
            "source_path": str(path),
        }
        row["input_text"] = build_input_text(row)
        rows.append(row)
    return dedupe_examples(rows)


def build_from_dialogue_json(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list):
        return []
    rows: list[dict[str, Any]] = []
    for dialogue in data:
        if not isinstance(dialogue, dict) or not isinstance(dialogue.get("turns"), list):
            continue
        dialogue_id = clean_text(dialogue.get("dialogue_id") or "")
        situation = clean_text(dialogue.get("situation") or "")
        history: list[dict[str, str]] = []
        for idx, turn in enumerate(dialogue["turns"]):
            if not isinstance(turn, dict):
                continue
            role = clean_text(turn.get("role") or turn.get("speaker") or ("A" if idx % 2 == 0 else "B"))
            if role.lower() in {"user", "speaker", "0"}:
                speaker = "A"
            elif role.lower() in {"assistant", "listener", "1"}:
                speaker = "B"
            else:
                speaker = role if role in {"A", "B"} else ("A" if idx % 2 == 0 else "B")
            text = clean_text(turn.get("utterance") or turn.get("text") or turn.get("content") or "")
            if history and text:
                target_speaker = speaker
                # The canonical main split follows the ED listener-response setup
                # used by CASE/Sibyl/APTNESS: only B/listener targets.
                if target_speaker == "B":
                    turn_id = int(turn.get("turn_id", idx)) + 1
                    example_id = f"{dialogue_id}_turn_{turn_id}" if dialogue_id else stable_hash([situation, history, text, idx], prefix="ed")
                    row = {
                        "example_id": example_id,
                        "dialogue_id": dialogue_id or stable_hash([situation, history[:1]], prefix="dialogue"),
                        "turn_id": turn_id,
                        "situation": situation,
                        "history": list(history),
                        "reference": text,
                        "target_speaker": target_speaker,
                        "source_path": str(path),
                    }
                    row["input_text"] = build_input_text(row)
                    rows.append(row)
            if text:
                history.append({"speaker": speaker, "text": text})
    return dedupe_examples(rows)


def build_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    for item in data:
        if not isinstance(item, dict):
            continue
        context = item.get("context") or item.get("history")
        response = clean_text(item.get("response") or item.get("reference") or item.get("gold_response") or "")
        situation = clean_text(item.get("situation", ""))
        if isinstance(context, str):
            history = []
            for line in context.splitlines():
                if ":" in line:
                    role, text = line.split(":", 1)
                    role = role.strip()
                    history.append({"speaker": role if role in {"A", "B"} else ("A" if len(history) % 2 == 0 else "B"), "text": clean_text(text)})
        elif isinstance(context, list):
            history = history_from_context_list(context)
        else:
            history = []
        if not history or not response:
            continue
        target_speaker = clean_text(item.get("next_role") or item.get("target_speaker") or ("A" if len(history) % 2 == 0 else "B"))
        if target_speaker != "B":
            continue
        turn_id = int(item.get("turn_id", len(history))) + 1
        dialogue_id = clean_text(item.get("dialogue_id") or "")
        example_id = f"{dialogue_id}_turn_{turn_id}" if dialogue_id else stable_hash([situation, history, response], prefix="ed")
        row = {
            "example_id": example_id,
            "dialogue_id": dialogue_id or stable_hash([situation, history[:1]], prefix="dialogue"),
            "turn_id": turn_id,
            "situation": situation,
            "history": history,
            "reference": response,
            "target_speaker": target_speaker,
            "source_path": str(path),
        }
        row["input_text"] = build_input_text(row)
        rows.append(row)
    return dedupe_examples(rows)


def build_from_csv(path: Path) -> list[dict[str, Any]]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dialogue_id = clean_text(row.get("conv_id") or row.get("dialogue_id") or "")
            if dialogue_id:
                by_dialogue[dialogue_id].append(row)
    rows: list[dict[str, Any]] = []
    for dialogue_id, turns in sorted(by_dialogue.items()):
        def idx_value(row: dict[str, Any]) -> int:
            try:
                return int(str(row.get("utterance_idx") or row.get("turn_id") or 0))
            except Exception:
                return 0
        turns.sort(key=idx_value)
        history: list[dict[str, str]] = []
        for raw in turns:
            idx = idx_value(raw)
            role = "A" if idx % 2 == 1 else "B"
            text = clean_text(raw.get("utterance") or raw.get("text") or "")
            if history and text and role == "B":
                situation = clean_text(raw.get("prompt") or raw.get("situation") or "")
                row = {
                    "example_id": f"{dialogue_id}_turn_{idx}",
                    "dialogue_id": dialogue_id,
                    "turn_id": idx,
                    "situation": situation,
                    "history": list(history),
                    "reference": text,
                    "target_speaker": role,
                    "source_path": str(path),
                }
                row["input_text"] = build_input_text(row)
                rows.append(row)
            if text:
                history.append({"speaker": role, "text": text})
    return dedupe_examples(rows)


def dedupe_examples(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_content: set[tuple[str, tuple[str, ...], str]] = set()
    for row in rows:
        key = context_key(row.get("situation"), row.get("history", []), row.get("reference"))
        example_id = str(row.get("example_id") or stable_hash([key], prefix="ed"))
        if example_id in seen_ids or key in seen_content:
            continue
        row["example_id"] = example_id
        seen_ids.add(example_id)
        seen_content.add(key)
        out.append(row)
    return out


def load_source(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix == ".json" and "sibyl" in str(path).lower():
        rows = build_from_sibyl_json(path)
        if rows:
            return rows
    if suffix == ".json":
        rows = build_from_dialogue_json(path)
        if rows:
            return rows
        return build_from_sibyl_json(path)
    if suffix == ".jsonl":
        return build_from_jsonl(path)
    if suffix == ".csv":
        return build_from_csv(path)
    return []


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    cfg = load_yaml(root / args.config)
    chosen_path: Path | None = None
    rows: list[dict[str, Any]] = []
    for path in source_candidates(root, cfg, args.source):
        rows = load_source(path)
        if rows:
            chosen_path = path
            break
    if not chosen_path or not rows:
        raise FileNotFoundError("Could not construct canonical ED test set from configured/project sources.")
    output = root / args.output
    meta_output = root / args.meta_output
    for row in rows:
        if row.get("source_path"):
            row["source_path"] = relpath(row["source_path"], root)
    write_jsonl(output, rows)
    meta = {
        "source_data_path": relpath(chosen_path, root),
        "number_of_examples": len(rows),
        "number_of_unique_example_id": len({row["example_id"] for row in rows}),
        "split_name": "test",
        "construction_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(root),
        "target_policy": "ED listener-response examples only; target_speaker is B for the canonical main comparison.",
        "input_policy": "input_text contains only situation, dialogue history, and speaker-role markers; reference is not included.",
    }
    write_json(meta_output, meta)
    print(f"Wrote {relpath(output, root)}")
    print(f"Source: {relpath(chosen_path, root)}")
    print(f"Examples: {len(rows)}")
    print(f"Unique example_id: {meta['number_of_unique_example_id']}")


if __name__ == "__main__":
    main()
