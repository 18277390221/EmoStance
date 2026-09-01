from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class UtteranceKey:
    split: str
    dialogue_id: str
    turn_id: int


def read_records(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for key in ("records", "data", "utterances", "annotations", "nodes", "emojis", "items", "clusters"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError("Reading parquet requires pandas and pyarrow.") from exc
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported file type: {path}")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_role(role: Any, turn_id: int) -> str:
    value = str(role).strip().upper() if role is not None else ""
    if value in {"A", "SPEAKER", "USER", "0"}:
        return "A"
    if value in {"B", "LISTENER", "ASSISTANT", "1"}:
        return "B"
    return "A" if int(turn_id) % 2 == 0 else "B"


def get_first(row: Dict[str, Any], names: Sequence[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            value = row[name]
            if not (isinstance(value, float) and np.isnan(value)):
                return value
    return default


def entropy(prob: Sequence[float], eps: float = 1e-12) -> float:
    arr = np.asarray(prob, dtype=np.float64)
    arr = arr / max(float(arr.sum()), eps)
    return float(-(arr * np.log(arr + eps)).sum())


def split_from_path(path: Path) -> str:
    name = path.name.lower()
    if name.startswith("train"):
        return "train"
    if name.startswith("valid") or name.startswith("validation") or name.startswith("dev"):
        return "dev"
    if name.startswith("test"):
        return "test"
    return "train"


def flatten_dialogue_annotation_records(path: str | Path, model_name: str | None = None) -> List[Dict[str, Any]]:
    """Read either utterance-row annotations or this repo's dialogue-with-turns JSON.

    Existing files such as `data/gpt-5.4/train_emoji_annotations.json` contain one
    record per dialogue and a `turns` array. This function flattens those files to
    one row per `(model, dialogue_id, turn_id)` so the soft-label merger can treat
    each model vote uniformly.
    """

    path = Path(path)
    records = read_records(path)
    inferred_model = model_name or path.parent.name
    inferred_split = split_from_path(path)
    flattened: List[Dict[str, Any]] = []
    for record in records:
        turns = record.get("turns")
        if isinstance(turns, list):
            dialogue_id = str(get_first(record, ["dialogue_id", "dialog_id", "conv_id", "conversation_id", "id"]))
            situation = str(get_first(record, ["situation", "prompt", "scenario"], ""))
            split = str(get_first(record, ["split", "dataset_split"], inferred_split))
            for idx, turn in enumerate(turns):
                ann = turn.get("emoji_annotation") or turn.get("annotation") or {}
                emoji = get_first(ann, ["selected_emoji", "emoji", "label", "prediction", "pred_emoji"], "")
                confidence = get_first(ann, ["confidence", "conf", "score", "prob"], 1.0)
                flattened.append(
                    {
                        "dialogue_id": dialogue_id,
                        "turn_id": int(get_first(turn, ["turn_id", "turn", "utterance_idx", "utt_id", "index"], idx)),
                        "split": "dev" if str(split).lower() == "valid" else split,
                        "role": normalize_role(get_first(turn, ["speaker", "role", "speaker_id"], ""), idx),
                        "text": str(get_first(turn, ["utterance", "text", "sentence", "content"], "")),
                        "situation": situation,
                        "model": inferred_model,
                        "emoji": str(emoji).strip(),
                        "confidence": float(confidence) if confidence not in (None, "") else 1.0,
                    }
                )
        else:
            row = dict(record)
            row.setdefault("model", inferred_model)
            row.setdefault("split", inferred_split)
            flattened.append(row)
    return flattened


def discover_annotation_files(annotation_root: str | Path, glob_pattern: str = "*/*_emoji_annotations.json") -> List[Path]:
    root = Path(annotation_root)
    files = sorted(root.glob(glob_pattern))
    if not files:
        files = sorted(root.glob("**/*_emoji_annotations.json"))
    return files


def load_annotation_records(annotation_paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in annotation_paths:
        rows.extend(flatten_dialogue_annotation_records(path))
    return rows


def group_annotation_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[UtteranceKey, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        dialogue_id = str(get_first(row, ["dialogue_id", "dialog_id", "conv_id", "conversation_id", "id"]))
        turn_id = int(get_first(row, ["turn_id", "turn", "utterance_idx", "utt_id", "index"], 0))
        split = str(get_first(row, ["split", "dataset_split"], "train"))
        split = "dev" if split.lower() in {"valid", "validation"} else split
        grouped[UtteranceKey(split, dialogue_id, turn_id)].append(row)

    utterances: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        base = rows[0]
        split = str(get_first(base, ["split", "dataset_split"], "train"))
        split = "dev" if split.lower() in {"valid", "validation"} else split
        text = str(get_first(base, ["text", "utterance", "sentence", "content"]))
        situation = str(get_first(base, ["situation", "prompt", "scenario"], ""))
        role = normalize_role(get_first(base, ["role", "speaker", "speaker_id"], ""), key.turn_id)
        votes: Dict[str, float] = defaultdict(float)
        conf_sum = 0.0
        model_votes: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            emoji = str(get_first(row, ["emoji", "selected_emoji", "label", "prediction", "pred_emoji"], "")).strip()
            if not emoji:
                continue
            confidence = max(float(get_first(row, ["confidence", "conf", "score", "prob"], 1.0)), 0.0)
            votes[emoji] += confidence
            conf_sum += confidence
            model = str(get_first(row, ["model", "model_name"], "unknown"))
            model_votes[model] = {"emoji": emoji, "confidence": confidence}
        utterances.append(
            {
                "dialogue_id": key.dialogue_id,
                "turn_id": key.turn_id,
                "split": key.split,
                "role": role,
                "text": text,
                "situation": situation,
                "emoji_votes": dict(votes),
                "model_votes": model_votes,
                "confidence_sum": conf_sum,
            }
        )
    utterances.sort(key=lambda x: (x["dialogue_id"], int(x["turn_id"])))
    return utterances


def iter_adjacent_pairs(utterances: List[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    by_dialogue: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for utt in utterances:
        by_dialogue[(str(utt.get("split", "train")), str(utt["dialogue_id"]))].append(utt)
    for key in sorted(by_dialogue):
        turns = sorted(by_dialogue[key], key=lambda x: int(x["turn_id"]))
        context_parts: List[str] = []
        for idx in range(len(turns) - 1):
            cur = turns[idx]
            nxt = turns[idx + 1]
            context_parts.append(f'{cur["role"]}: {cur["text"]}')
            yield cur, nxt, "\n".join(context_parts)


def load_prepared_split(prepared_dir: str | Path, split: str) -> List[Dict[str, Any]]:
    path = Path(prepared_dir) / f"{split}.jsonl"
    if not path.exists():
        return []
    return read_records(path)


def build_model_text(row: Dict[str, Any]) -> str:
    situation = row.get("situation", "")
    context = row.get("context", "")
    if situation:
        return f"Situation: {situation}\nDialogue:\n{context}"
    return f"Dialogue:\n{context}"


class StanceDataset:
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(build_model_text(row), truncation=True, max_length=self.max_length, padding=False)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "source_cluster": np.asarray(row["source_cluster"], dtype=np.float32),
            "target_cluster": np.asarray(row["target_cluster"], dtype=np.float32),
            "source_vector": np.asarray(row["source_vector"], dtype=np.float32),
            "target_vector": np.asarray(row["target_vector"], dtype=np.float32),
            "transition_id": row.get("transition", "A->B"),
        }
