#!/usr/bin/env python3
"""Join released EmojiDialogue metadata with a local EmpatheticDialogues copy."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TOKEN_REPLACEMENTS = {"_comma_": ",", "_COLON_": ":"}


def decode_ed_text(value: str | None) -> str:
    text = value or ""
    for source, target in TOKEN_REPLACEMENTS.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def load_ed_dialogues(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[str(row["conv_id"])].append(row)

    dialogues: dict[str, dict[str, Any]] = {}
    for dialogue_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["utterance_idx"]))
        role_by_speaker: dict[str, str] = {}
        turns = []
        for turn_id, row in enumerate(rows):
            speaker = str(row.get("speaker_idx", ""))
            if speaker not in role_by_speaker:
                role_by_speaker[speaker] = chr(ord("A") + len(role_by_speaker))
            turns.append(
                {
                    "turn_id": turn_id,
                    "speaker": role_by_speaker[speaker],
                    "utterance": decode_ed_text(row.get("utterance")),
                }
            )
        dialogues[dialogue_id] = {
            "dialogue_id": dialogue_id,
            "situation": decode_ed_text(rows[0].get("prompt")),
            "num_turns": len(turns),
            "speakers": sorted(set(role_by_speaker.values())),
            "turns": turns,
        }
    return dialogues


def load_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        handle = path.open(mode="r", encoding="utf-8")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def reconstruct_split(
    metadata: list[dict[str, Any]],
    ed_dialogues: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for item in metadata:
        dialogue_id = str(item["dialogue_id"])
        if dialogue_id not in ed_dialogues:
            raise KeyError(f"Dialogue {dialogue_id!r} is absent from the official ED split")
        dialogue = json.loads(json.dumps(ed_dialogues[dialogue_id], ensure_ascii=False))
        dialogue["split"] = str(item["split"])
        turns = dialogue["turns"]
        for annotation in item.get("turn_annotations", []):
            turn_id = int(annotation["turn_id"])
            if turn_id < 0 or turn_id >= len(turns):
                raise IndexError(f"Invalid turn {turn_id} for {dialogue_id}")
            expected_speaker = str(annotation.get("speaker", ""))
            if expected_speaker and turns[turn_id]["speaker"] != expected_speaker:
                raise ValueError(
                    f"Speaker mismatch for {dialogue_id} turn {turn_id}: "
                    f"metadata={expected_speaker}, ED={turns[turn_id]['speaker']}"
                )
            turns[turn_id]["emoji_annotation"] = {
                "selected_emoji": annotation["selected_emoji"],
                "unicode": annotation.get("unicode"),
                "confidence": annotation.get("confidence"),
            }
        if any("emoji_annotation" not in turn for turn in turns):
            raise ValueError(f"Incomplete turn annotations for {dialogue_id}")
        output.append(dialogue)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, default=Path("data/annotation_metadata"))
    parser.add_argument("--ed-root", type=Path, required=True, help="Directory containing official train.csv, valid.csv, and test.csv files.")
    parser.add_argument("--output-root", type=Path, default=Path("private_data/reconstructed"))
    args = parser.parse_args()

    metadata_files = sorted(args.metadata_root.glob("*/*.jsonl.gz"))
    if not metadata_files:
        raise FileNotFoundError(f"No released metadata found under {args.metadata_root}")

    ed_cache: dict[str, dict[str, dict[str, Any]]] = {}
    for metadata_path in metadata_files:
        model = metadata_path.parent.name
        split = metadata_path.name.removesuffix(".jsonl.gz")
        csv_name = "valid.csv" if split in {"dev", "valid", "validation"} else f"{split}.csv"
        if csv_name not in ed_cache:
            ed_path = args.ed_root / csv_name
            if not ed_path.exists():
                raise FileNotFoundError(ed_path)
            ed_cache[csv_name] = load_ed_dialogues(ed_path)

        reconstructed = reconstruct_split(load_metadata(metadata_path), ed_cache[csv_name])
        output_path = args.output_root / model / f"{split}_emoji_annotations.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(reconstructed, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(reconstructed)} dialogues to {output_path}")


if __name__ == "__main__":
    main()
