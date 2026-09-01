from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SYSTEM_DIR = Path(__file__).resolve().parents[1]
ROOT = SYSTEM_DIR.parent
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

METHOD_ORDER = [
    "ours",
    "llm_only",
    "llm_prompt",
    "llm_sft",
    "empo_dpo",
    "case",
    "aptness",
    "sibyl",
]

DISPLAY_NAMES = {
    "ours": "EmoStance / Ours",
    "llm_only": "LLM-only",
    "llm_prompt": "LLM-prompt",
    "llm_sft": "LLM-SFT",
    "empo_dpo": "EmPO-DPO",
    "case": "CASE",
    "aptness": "APTNESS",
    "sibyl": "Sibyl",
}

RESPONSE_FIELDS = {
    "ours": ["generated_response", "response", "output", "prediction"],
    "llm_only": ["generated_response", "response", "output", "prediction"],
    "llm_prompt": ["generated_response", "response", "output", "prediction"],
    "llm_sft": ["LLM-SFT", "EmPO-SFT", "llm_sft", "response", "generated_response", "output"],
    "empo_dpo": ["EmPO-DPO", "empo_dpo", "response", "generated_response", "output"],
    "case": ["case_response", "response", "generated_response", "output", "prediction"],
    "aptness": ["response", "initial_response", "generated_response", "output", "prediction"],
    "sibyl": ["mistral_sibyl_response", "sibyl_response", "response", "generated_response", "output"],
}

GENERIC_PATTERNS = [
    "i'm sorry",
    "i am sorry",
    "sorry to hear",
    "that sounds",
    "i understand",
    "i know how you feel",
    "that's great",
    "that's awesome",
    "oh no",
]


def ensure_system_dirs() -> None:
    for name in ["configs", "scripts", "data", "generations", "metrics", "reports", "logs"]:
        (SYSTEM_DIR / name).mkdir(parents=True, exist_ok=True)


def rel(path: str | Path, root: Path | None = None) -> str:
    path = Path(path)
    base = root or ROOT
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {path}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize(dict(row)), ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    if fieldnames is None:
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(str(key))
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("_comma_", ",")
    text = text.replace(" ,", ",")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_match(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: Any) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text or ""))]


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def word_count(text: Any) -> int:
    return len(tokenize(text))


def clean_response(value: Any) -> str:
    text = clean_text(value)
    prefixes = [
        "Response:",
        "Reply:",
        "Assistant:",
        "Listener:",
        "Speaker:",
        "Generated response:",
        "B:",
        "A:",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return text.strip().strip('"').strip()


def history_from_any(value: Any) -> list[dict[str, str]]:
    if isinstance(value, list):
        out: list[dict[str, str]] = []
        for i, item in enumerate(value):
            if isinstance(item, dict):
                role_raw = str(item.get("role") or item.get("speaker") or item.get("speaker_id") or "")
                text = clean_text(item.get("text") or item.get("content") or item.get("utterance") or item.get("sentence") or "")
            else:
                role_raw = ""
                text = clean_text(item)
            role = normalize_role(role_raw, i)
            if text:
                out.append({"role": role, "text": text})
        return out
    text = str(value or "")
    out = []
    for i, part in enumerate(re.split(r"\s*<SEP>\s*|\n+", text)):
        part = clean_text(part)
        if not part:
            continue
        match = re.match(r"^(A|B|Speaker|Listener|user|assistant)\s*:\s*(.+)$", part, flags=re.I)
        if match:
            role = normalize_role(match.group(1), i)
            utt = clean_text(match.group(2))
        else:
            role = "A" if i % 2 == 0 else "B"
            utt = part
        out.append({"role": role, "text": utt})
    return out


def normalize_role(role: Any, index: int = 0) -> str:
    value = str(role or "").strip().lower()
    if value in {"a", "speaker", "user", "0"}:
        return "A"
    if value in {"b", "listener", "assistant", "1"}:
        return "B"
    return "A" if index % 2 == 0 else "B"


def serialize_history(history: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(f"{row.get('role', 'A')}: {clean_text(row.get('text', ''))}" for row in history)


def pick_response(row: Mapping[str, Any], method_id: str) -> str:
    for field in RESPONSE_FIELDS.get(method_id, []):
        value = row.get(field)
        if value not in (None, ""):
            return clean_response(value)
    for field in ["generated_response", "response", "output", "prediction", "text", "reply", "candidate"]:
        value = row.get(field)
        if value not in (None, ""):
            return clean_response(value)
    return ""


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def distinct_n(responses: Sequence[str], n: int) -> float:
    all_ngrams: list[tuple[str, ...]] = []
    for text in responses:
        all_ngrams.extend(ngrams(tokenize(text), n))
    return safe_div(len(set(all_ngrams)), len(all_ngrams))


def modified_precision(candidate: Sequence[str], references: Sequence[Sequence[str]], n: int) -> float:
    cand_counts = Counter(ngrams(candidate, n))
    if not cand_counts:
        return 0.0
    max_ref: Counter[tuple[str, ...]] = Counter()
    for ref in references:
        ref_counts = Counter(ngrams(ref, n))
        for gram, count in ref_counts.items():
            if count > max_ref[gram]:
                max_ref[gram] = count
    clipped = sum(min(count, max_ref[gram]) for gram, count in cand_counts.items())
    return safe_div(clipped, sum(cand_counts.values()))


def sentence_bleu(candidate_text: str, reference_texts: Sequence[str], max_n: int = 4) -> float:
    cand = tokenize(candidate_text)
    refs = [tokenize(r) for r in reference_texts if r is not None]
    if not cand or not refs:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        p = modified_precision(cand, refs, n)
        precisions.append(max(p, 1e-9))
    cand_len = len(cand)
    ref_len = min((len(r) for r in refs), key=lambda x: (abs(x - cand_len), x))
    bp = 1.0 if cand_len > ref_len else math.exp(1.0 - safe_div(ref_len, cand_len)) if cand_len else 0.0
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def rouge_n_f1(candidate_text: str, reference_text: str, n: int) -> float:
    cand = Counter(ngrams(tokenize(candidate_text), n))
    ref = Counter(ngrams(tokenize(reference_text), n))
    if not cand or not ref:
        return 0.0
    overlap = sum(min(cand[k], ref[k]) for k in cand)
    precision = safe_div(overlap, sum(cand.values()))
    recall = safe_div(overlap, sum(ref.values()))
    return safe_div(2 * precision * recall, precision + recall)


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(candidate_text: str, reference_text: str) -> float:
    cand = tokenize(candidate_text)
    ref = tokenize(reference_text)
    if not cand or not ref:
        return 0.0
    lcs = lcs_len(cand, ref)
    precision = safe_div(lcs, len(cand))
    recall = safe_div(lcs, len(ref))
    return safe_div(2 * precision * recall, precision + recall)


def meteor_fallback(candidate_text: str, reference_text: str) -> float:
    cand = tokenize(candidate_text)
    ref = tokenize(reference_text)
    if not cand or not ref:
        return 0.0
    overlap = sum((Counter(cand) & Counter(ref)).values())
    precision = safe_div(overlap, len(cand))
    recall = safe_div(overlap, len(ref))
    return safe_div(10 * precision * recall, recall + 9 * precision)


def repetition_rate(text: str) -> float:
    toks = tokenize(text)
    if len(toks) < 2:
        return 0.0
    bigrams = ngrams(toks, 2)
    trigrams = ngrams(toks, 3)
    repeated = (len(bigrams) - len(set(bigrams))) + (len(trigrams) - len(set(trigrams)))
    total = len(bigrams) + len(trigrams)
    return safe_div(repeated, total)


def generic_response(text: str) -> bool:
    lower = clean_text(text).lower()
    return any(pattern in lower for pattern in GENERIC_PATTERNS)


def role_marker_leak(text: str) -> bool:
    return bool(re.search(r"(^|\n)\s*(A|B|Speaker|Listener|User|Assistant)\s*:", str(text or "")))


def multi_turn_leak(text: str) -> bool:
    return len(re.findall(r"(^|\n)\s*(A|B|Speaker|Listener|User|Assistant)\s*:", str(text or ""), flags=re.I)) >= 2


def context_copy_rate(response: str, history: Sequence[Mapping[str, Any]]) -> float:
    resp = set(tokenize(response))
    ctx: set[str] = set()
    for turn in history:
        ctx.update(tokenize(turn.get("text", "")))
    if not resp:
        return 0.0
    return safe_div(len(resp & ctx), len(resp))


def get_git_commit(root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def env_payload(root: Path = ROOT) -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "git_commit": get_git_commit(root),
        "cwd": str(root),
        "optional_dependencies": {
            name: module_available(name)
            for name in ["bert_score", "rouge_score", "nltk", "evaluate", "transformers", "torch"]
        },
    }


def module_available(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False
