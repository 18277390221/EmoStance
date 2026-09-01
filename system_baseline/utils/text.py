from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("_comma_", ",")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",")
    return text


def normalize_for_match(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: Any) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text or ""))]


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def serialize_history(history: Sequence[Mapping[str, Any]], role_key: str = "speaker") -> str:
    lines: list[str] = []
    for idx, turn in enumerate(history):
        role = str(turn.get(role_key) or turn.get("role") or ("A" if idx % 2 == 0 else "B")).strip() or "A"
        text = clean_text(turn.get("text") or turn.get("content") or "")
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_input_text(example: Mapping[str, Any]) -> str:
    situation = clean_text(example.get("situation", ""))
    history = serialize_history(example.get("history", []) if isinstance(example.get("history"), list) else [])
    target = clean_text(example.get("target_speaker") or example.get("target_role") or "B")
    parts: list[str] = []
    if situation:
        parts.append(f"Situation:\n{situation}")
    parts.append("Dialogue history:")
    parts.append(history if history else "N/A")
    parts.append(f"Next speaker: {target}")
    return "\n\n".join(parts)


def stable_hash(parts: Sequence[Any], prefix: str = "ed") -> str:
    payload = "\n".join(clean_text(p) if not isinstance(p, (dict, list, tuple)) else repr(p) for p in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def history_from_context_list(context: Sequence[Any]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for idx, item in enumerate(context):
        if isinstance(item, dict):
            text = clean_text(item.get("text") or item.get("content") or item.get("utterance") or "")
            role_raw = str(item.get("speaker") or item.get("role") or "").strip().lower()
            if role_raw in {"assistant", "listener", "b", "1"}:
                speaker = "B"
            elif role_raw in {"user", "speaker", "a", "0"}:
                speaker = "A"
            else:
                speaker = "A" if idx % 2 == 0 else "B"
        else:
            text = clean_text(item)
            speaker = "A" if idx % 2 == 0 else "B"
        if text:
            history.append({"speaker": speaker, "text": text})
    return history


def clean_prediction(value: Any) -> str:
    text = clean_text(value)
    prefixes = [
        "Response:",
        "Reply:",
        "Assistant:",
        "Listener:",
        "Speaker:",
        "Generated response:",
        "Mistral-Sibyl Response:",
        "B:",
        "A:",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
                changed = True
    for marker in ("\nA:", "\nB:", "\nUser:", "\nAssistant:", "\nSpeaker:", "\nListener:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text.strip().strip('"').strip()


def context_key(situation: Any, history: Sequence[Mapping[str, Any]], reference: Any = "") -> tuple[str, tuple[str, ...], str]:
    return (
        normalize_for_match(situation),
        tuple(normalize_for_match(turn.get("text", "")) for turn in history),
        normalize_for_match(reference),
    )

