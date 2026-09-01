#!/usr/bin/env python3
"""Export EmojiDialogue annotations without redistributing dialogue text.

The source files are the private, text-bearing annotation files used during
development. The output contains only stable EmpatheticDialogues identifiers,
speaker/turn indices, emoji votes, and confidence scores.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable


FORBIDDEN_KEYS = {
    "context",
    "prompt",
    "raw_model_output",
    "situation",
    "text",
    "utterance",
}


def iter_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from iter_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_keys(child)


def metadata_record(dialogue: dict[str, Any], split: str) -> dict[str, Any]:
    turns = []
    for index, turn in enumerate(dialogue.get("turns", [])):
        annotation = turn.get("emoji_annotation") or {}
        selected = annotation.get("selected_emoji")
        if not selected:
            continue
        turns.append(
            {
                "turn_id": int(turn.get("turn_id", index)),
                "speaker": str(turn.get("speaker", turn.get("role", ""))),
                "selected_emoji": str(selected),
                "unicode": annotation.get("unicode"),
                "confidence": annotation.get("confidence"),
            }
        )
    record = {
        "dialogue_id": str(dialogue["dialogue_id"]),
        "split": split,
        "turn_annotations": turns,
    }
    leaked = FORBIDDEN_KEYS.intersection(iter_keys(record))
    if leaked:
        raise ValueError(f"Forbidden text-bearing keys in metadata: {sorted(leaked)}")
    return record


def write_deterministic_gzip_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    digest.update(line.encode("utf-8"))
                    text.write(line)
                    count += 1
    return count, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest: dict[str, Any] = {"schema_version": "1.0", "files": []}
    source_files = sorted(args.source_root.glob("*/*_emoji_annotations.json"))
    if not source_files:
        raise FileNotFoundError(f"No annotation files found under {args.source_root}")

    for source in source_files:
        model = source.parent.name
        split = source.name.removesuffix("_emoji_annotations.json")
        with source.open(encoding="utf-8") as handle:
            dialogues = json.load(handle)
        if not isinstance(dialogues, list):
            raise ValueError(f"Expected a list in {source}")

        output = args.output_root / model / f"{split}.jsonl.gz"
        records = (metadata_record(dialogue, split) for dialogue in dialogues)
        count, content_sha256 = write_deterministic_gzip_jsonl(output, records)
        manifest["files"].append(
            {
                "path": output.relative_to(args.output_root).as_posix(),
                "model": model,
                "split": split,
                "dialogues": count,
                "jsonl_content_sha256": content_sha256,
            }
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest['files'])} metadata files to {args.output_root}")


if __name__ == "__main__":
    main()
