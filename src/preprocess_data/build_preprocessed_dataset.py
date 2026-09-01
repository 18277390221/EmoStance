import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "valid", "test")
QUESTION_START_RE = re.compile(
    r"""^[\s"'“”‘’(\[]*
    (who|what|when|where|why|how|
    do|does|did|
    is|are|am|was|were|
    can|could|would|should|will|
    have|has|had|may|might)\b""",
    re.IGNORECASE | re.VERBOSE,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_question_utterance(text: str) -> bool:
    normalized = (text or "").strip()
    if not normalized:
        return False
    if "?" in normalized or "？" in normalized:
        return True
    return bool(QUESTION_START_RE.match(normalized))


def build_role_pair(current_role: str | None, future_role: str | None) -> str | None:
    if not current_role or not future_role:
        return None
    return f"{current_role}2{future_role}"


def load_emoji_pool(pool_path: Path) -> set[str]:
    emoji_pool = load_json(pool_path)
    if not isinstance(emoji_pool, list):
        raise ValueError(f"Emoji pool must be a list: {pool_path}")
    return {item.get("emoji") for item in emoji_pool if isinstance(item, dict) and item.get("emoji")}


def build_dialogue_signature(dialogue: dict[str, Any]) -> tuple[Any, ...]:
    turns = dialogue.get("turns") or []
    return (
        dialogue.get("situation"),
        dialogue.get("num_turns") or len(turns),
        tuple(
            (
                turn.get("turn_id", idx),
                turn.get("speaker"),
                turn.get("utterance"),
            )
            for idx, turn in enumerate(turns)
        ),
    )


def collect_model_dialogues(data_dir: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    dialogues_by_split: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        split: defaultdict(dict) for split in SPLITS
    }

    for model_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for split in SPLITS:
            input_path = model_dir / f"{split}_emoji_annotations.json"
            if not input_path.exists():
                continue

            dialogues = load_json(input_path)
            if not isinstance(dialogues, list):
                raise ValueError(f"Expected a list of dialogues in {input_path}")

            for dialogue in dialogues:
                dialogue_id = dialogue.get("dialogue_id")
                if not dialogue_id:
                    raise ValueError(f"Missing dialogue_id in {input_path}")
                dialogues_by_split[split][dialogue_id][model_dir.name] = dialogue

    return dialogues_by_split


def merge_dialogue_annotations(
    dialogue_group: dict[str, dict[str, Any]],
    *,
    all_models: list[str],
    split: str,
    emoji_pool: set[str],
) -> tuple[dict[str, Any], set[str]]:
    available_models = sorted(dialogue_group)
    base_model = available_models[0]
    base_dialogue = dialogue_group[base_model]
    base_signature = build_dialogue_signature(base_dialogue)
    unknown_emojis: set[str] = set()

    for model_name in available_models[1:]:
        other_signature = build_dialogue_signature(dialogue_group[model_name])
        if other_signature != base_signature:
            dialogue_id = base_dialogue.get("dialogue_id")
            raise ValueError(
                f"Dialogue content mismatch for {dialogue_id} between {base_model} and {model_name}"
            )

    base_turns = base_dialogue.get("turns") or []
    merged_turns: list[dict[str, Any]] = []

    for idx, base_turn in enumerate(base_turns):
        next_turn = base_turns[idx + 1] if idx + 1 < len(base_turns) else None
        lag2_turn = base_turns[idx + 2] if idx + 2 < len(base_turns) else None
        current_role = base_turn.get("speaker")
        next_role = (next_turn or {}).get("speaker")
        lag2_role = (lag2_turn or {}).get("speaker")

        emoji_by_model: dict[str, str | None] = {}
        emoji_unicode_by_model: dict[str, str | None] = {}
        confidence_by_model: dict[str, int | None] = {}
        confidence_desc_by_model: dict[str, str | None] = {}
        annotation_error_by_model: dict[str, Any] = {}
        next_emoji_by_model: dict[str, str | None] = {}
        next_confidence_by_model: dict[str, int | None] = {}

        for model_name, dialogue in dialogue_group.items():
            model_turns = dialogue.get("turns") or []
            if idx >= len(model_turns):
                raise ValueError(
                    f"Turn count mismatch for dialogue {dialogue.get('dialogue_id')} in {model_name}"
                )

            annotation = (model_turns[idx].get("emoji_annotation") or {})
            emoji_value = annotation.get("selected_emoji")
            if emoji_value and emoji_value not in emoji_pool:
                unknown_emojis.add(emoji_value)

            emoji_by_model[model_name] = emoji_value
            emoji_unicode_by_model[model_name] = annotation.get("unicode")
            confidence_by_model[model_name] = annotation.get("confidence")
            confidence_desc_by_model[model_name] = annotation.get("confidence_desc")
            annotation_error_by_model[model_name] = annotation.get("error")

            model_next_turn = model_turns[idx + 1] if idx + 1 < len(model_turns) else None
            next_annotation = (model_next_turn or {}).get("emoji_annotation") or {}
            next_emoji_value = next_annotation.get("selected_emoji")
            if next_emoji_value and next_emoji_value not in emoji_pool:
                unknown_emojis.add(next_emoji_value)

            next_emoji_by_model[model_name] = next_emoji_value
            next_confidence_by_model[model_name] = next_annotation.get("confidence")

        for model_name in all_models:
            emoji_by_model.setdefault(model_name, None)
            emoji_unicode_by_model.setdefault(model_name, None)
            confidence_by_model.setdefault(model_name, None)
            confidence_desc_by_model.setdefault(model_name, None)
            annotation_error_by_model.setdefault(model_name, None)
            next_emoji_by_model.setdefault(model_name, None)
            next_confidence_by_model.setdefault(model_name, None)

        merged_turns.append(
            {
                "turn_id": base_turn.get("turn_id", idx),
                "turn_index": idx,
                "role": current_role,
                "utterance": base_turn.get("utterance"),
                "is_question": is_question_utterance(base_turn.get("utterance", "")),
                "is_last_turn": idx == len(base_turns) - 1,
                "next_role": next_role,
                "role_pair": build_role_pair(current_role, next_role),
                "lag2_role_pair": build_role_pair(current_role, lag2_role),
                "emoji": emoji_by_model,
                "emoji_unicode": emoji_unicode_by_model,
                "confidence": confidence_by_model,
                "confidence_desc": confidence_desc_by_model,
                "annotation_error": annotation_error_by_model,
                "next_emoji": next_emoji_by_model,
                "next_confidence": next_confidence_by_model,
            }
        )

    merged_dialogue = {
        "dialogue_id": base_dialogue.get("dialogue_id"),
        "split": split,
        "situation": base_dialogue.get("situation"),
        "num_turns": base_dialogue.get("num_turns") or len(base_turns),
        "speakers": base_dialogue.get("speakers") or [],
        "annotation_mode": base_dialogue.get("annotation_mode"),
        "available_models": available_models,
        "turns": merged_turns,
    }
    return merged_dialogue, unknown_emojis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged dialogue-level emoji datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing model subfolders with train/valid/test annotation files.",
    )
    parser.add_argument(
        "--emoji-pool",
        type=Path,
        default=Path("data/face_emojis_0.json"),
        help="Path to the emoji pool JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pre_data"),
        help="Directory for preprocessed output JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emoji_pool = load_emoji_pool(args.emoji_pool)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_models = sorted(path.name for path in args.data_dir.iterdir() if path.is_dir())
    dialogues_by_split = collect_model_dialogues(args.data_dir)
    merged_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_emojis_by_split: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        for dialogue_id in sorted(dialogues_by_split[split]):
            dialogue_group = dialogues_by_split[split][dialogue_id]
            if len(dialogue_group) != len(all_models):
                continue

            merged_dialogue, unknown_emojis = merge_dialogue_annotations(
                dialogue_group,
                all_models=all_models,
                split=split,
                emoji_pool=emoji_pool,
            )
            merged_records[split].append(merged_dialogue)
            unknown_emojis_by_split[split].update(unknown_emojis)

    for split in SPLITS:
        output_path = args.output_dir / f"{split}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(merged_records[split], f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(merged_records[split])} records to {output_path}")

        unknown = sorted(unknown_emojis_by_split[split])
        if unknown:
            preview = ", ".join(unknown[:10])
            print(f"Warning: {split} has {len(unknown)} emojis missing from the pool: {preview}")


if __name__ == "__main__":
    main()
