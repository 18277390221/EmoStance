#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_LANGUAGE = "zh-Hans"
DEFAULT_SEED = 20260519
EXPECTED_MODEL_HINTS = ("claude", "deepseek", "gemini", "gpt")
SPLIT_ORDER = ("train", "valid", "test")

TRANSLATION_SYSTEM_PROMPT = (
    "You are a careful translation engine for dialogue annotation. Translate the input faithfully into "
    "Simplified Chinese. Do not add explanations, notes, summaries, alternatives, or extra text. "
    "Preserve emojis, speaker names, punctuation, and line breaks when appropriate. Return only the "
    "Chinese translation."
)
TRANSLATION_USER_PREFIX = (
    "Translate the following English text into Simplified Chinese for human annotation. Return only the translation:\n\n"
)


@dataclass(frozen=True)
class ModelSource:
    canonical_name: str
    source_name: str
    source_dir: Path
    split_paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline emoji weak-label human-audit HTML files.")
    parser.add_argument("--n", type=int, default=300, help="Number of model-dialogue packages to sample.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out_dir", default="data_eval")
    parser.add_argument("--model_path", default=None, help="Optional local Mistral-7B-Instruct-v0.3 path.")
    parser.add_argument("--force_resample", action="store_true", help="Overwrite an existing sample manifest.")
    parser.add_argument("--reuse_sample", action="store_true", help="Reuse an existing sample manifest when present.")
    parser.add_argument(
        "--skip_existing_translations",
        action="store_true",
        help="Reuse cached translations; missing cache entries are still translated.",
    )
    parser.add_argument("--smoke_test_n", type=int, default=0, help="Generate a small smoke-test build under data_eval/smoke_test.")
    parser.add_argument("--translation_batch_size", type=int, default=2)
    return parser.parse_args()


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_ed_text(text: Any) -> str:
    return str(text or "").replace("_comma_", ",").strip()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_model_name(name: str) -> str:
    low = name.lower()
    if "claude" in low or "sonnet" in low:
        return "Claude Sonnet"
    if "deepseek" in low:
        return "DeepSeek"
    if "gemini" in low:
        return "Gemini"
    if "gpt" in low or "openai" in low:
        return "GPT"
    return name


def discover_model_sources(project_root: Path) -> list[ModelSource]:
    data_dir = project_root / "data"
    candidates: list[ModelSource] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset annotation directory not found: {data_dir}")
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        split_paths = {
            split: child / f"{split}_emoji_annotations.json"
            for split in SPLIT_ORDER
            if (child / f"{split}_emoji_annotations.json").exists()
        }
        if len(split_paths) < 3:
            continue
        source_name = child.name
        canonical = canonical_model_name(source_name)
        if not any(hint in source_name.lower() for hint in EXPECTED_MODEL_HINTS):
            continue
        candidates.append(ModelSource(canonical, source_name, child, split_paths))
    by_canonical: dict[str, ModelSource] = {}
    for source in candidates:
        by_canonical[source.canonical_name] = source
    ordered_names = ["Claude Sonnet", "DeepSeek", "Gemini", "GPT"]
    ordered = [by_canonical[name] for name in ordered_names if name in by_canonical]
    if len(ordered) < 4:
        found = ", ".join(f"{s.canonical_name}={s.source_dir}" for s in ordered)
        raise RuntimeError(f"Expected four LLM emoji annotator model directories; found {len(ordered)}: {found}")
    print("Detected LLM annotator model mapping:")
    for source in ordered:
        print(f"  - {source.canonical_name}: {source.source_dir}")
    return ordered


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_records(source: ModelSource) -> dict[str, dict[str, dict[str, Any]]]:
    data: dict[str, dict[str, dict[str, Any]]] = {}
    for split, path in source.split_paths.items():
        rows = read_json(path)
        split_records: dict[str, dict[str, Any]] = {}
        if not isinstance(rows, list):
            raise ValueError(f"Expected list in {path}, got {type(rows)}")
        for row in rows:
            dialogue_id = str(row.get("dialogue_id") or row.get("conv_id") or row.get("conversation_id") or "")
            if not dialogue_id:
                continue
            split_records[dialogue_id] = row
        data[split] = split_records
    return data


def load_all_model_records(sources: list[ModelSource]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    return {source.canonical_name: load_model_records(source) for source in sources}


def load_emotion_metadata(project_root: Path) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    ed_dir = project_root / "empatheticdialogues"
    for split in SPLIT_ORDER:
        path = ed_dir / f"{split}.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue_id = str(row.get("conv_id") or row.get("dialogue_id") or "")
                if not dialogue_id:
                    continue
                meta.setdefault(
                    dialogue_id,
                    {
                        "original_emotion": clean_ed_text(row.get("context", "")),
                        "ed_prompt": clean_ed_text(row.get("prompt", "")),
                        "ed_split": split,
                        "ed_source_path": str(path),
                    },
                )
    return meta


def largest_remainder_quotas(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Cannot compute quotas from zero split counts.")
    raw = {split: counts[split] * n / total for split in counts}
    quotas = {split: int(raw[split]) for split in counts}
    remaining = n - sum(quotas.values())
    order = sorted(counts, key=lambda split: (raw[split] - quotas[split], counts[split]), reverse=True)
    for split in order[:remaining]:
        quotas[split] += 1
    return quotas


def model_slots(model_names: list[str], n: int, rng: random.Random) -> list[str]:
    base = n // len(model_names)
    remainder = n % len(model_names)
    slots: list[str] = []
    for idx, model in enumerate(model_names):
        slots.extend([model] * (base + (1 if idx < remainder else 0)))
    rng.shuffle(slots)
    return slots


def sample_dialogues(
    all_records: dict[str, dict[str, dict[str, dict[str, Any]]]],
    n: int,
    seed: int,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    rng = random.Random(seed)
    model_names = list(all_records.keys())
    split_counts: dict[str, int] = {}
    common_ids: dict[str, list[str]] = {}
    for split in SPLIT_ORDER:
        id_sets = [set(all_records[model].get(split, {})) for model in model_names]
        common = sorted(set.intersection(*id_sets))
        common_ids[split] = common
        split_counts[split] = len(common)
    quotas = largest_remainder_quotas(split_counts, n)
    sampled: list[tuple[str, str]] = []
    for split in SPLIT_ORDER:
        ids = common_ids[split][:]
        rng.shuffle(ids)
        if len(ids) < quotas[split]:
            raise RuntimeError(f"Not enough common dialogues for split={split}: need {quotas[split]}, have {len(ids)}")
        sampled.extend((split, dialogue_id) for dialogue_id in ids[: quotas[split]])
    rng.shuffle(sampled)
    summary = {
        "dialogue_level_common_split_counts": split_counts,
        "sample_split_quotas": quotas,
        "sampled_dialogues": len(sampled),
    }
    return sampled, summary


def confidence_bin(confidence: Any) -> str:
    if confidence in (None, "", "NA"):
        return "missing"
    try:
        value = int(float(confidence))
    except (TypeError, ValueError):
        return "missing"
    if value <= 2:
        return "1-2"
    if value == 3:
        return "3"
    if value >= 4:
        return "4-5"
    return "missing"


def normalize_turns(row: dict[str, Any], sampled_model: str) -> list[dict[str, Any]]:
    turns = row.get("turns") or row.get("utterances") or []
    normalized: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        annotation = turn.get("emoji_annotation") if isinstance(turn, dict) else None
        annotation = annotation if isinstance(annotation, dict) else {}
        confidence = annotation.get("confidence")
        normalized.append(
            {
                "turn_index": int(turn.get("turn_id", turn.get("utterance_idx", turn.get("index", idx)))),
                "role": str(turn.get("speaker") or turn.get("role") or turn.get("speaker_role") or "unknown"),
                "english_original_text": clean_ed_text(
                    turn.get("utterance") or turn.get("text") or turn.get("utterance_text") or turn.get("response") or ""
                ),
                "displayed_emoji": annotation.get("selected_emoji") or annotation.get("emoji") or "MISSING",
                "confidence_hidden_if_available": confidence if confidence is not None else "",
                "confidence_bin": confidence_bin(confidence),
                "unicode": annotation.get("unicode", ""),
                "annotation_error": annotation.get("error", ""),
                "sampled_model": sampled_model,
            }
        )
    return sorted(normalized, key=lambda x: x["turn_index"])


def make_packages(
    sampled_dialogues: list[tuple[str, str]],
    model_names: list[str],
    all_records: dict[str, dict[str, dict[str, dict[str, Any]]]],
    sources: list[ModelSource],
    emotion_meta: dict[str, dict[str, str]],
    seed: int,
    n: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 17)
    slots = model_slots(model_names, n, rng)
    source_by_model = {source.canonical_name: source for source in sources}
    timestamp = datetime.now(timezone.utc).isoformat()
    packages: list[dict[str, Any]] = []
    for idx, ((split, dialogue_id), sampled_model) in enumerate(zip(sampled_dialogues, slots), 1):
        row = all_records[sampled_model][split][dialogue_id]
        meta = emotion_meta.get(dialogue_id, {})
        turns = normalize_turns(row, sampled_model)
        if len(turns) < 2:
            continue
        source_path = source_by_model[sampled_model].split_paths[split]
        situation = clean_ed_text(row.get("situation") or row.get("prompt") or meta.get("ed_prompt", ""))
        package = {
            "package_id": f"pkg_{idx:04d}",
            "dialogue_id": dialogue_id,
            "split": str(row.get("split") or split),
            "sampled_model": sampled_model,
            "original_emotion": meta.get("original_emotion", ""),
            "situation": situation,
            "situation_zh": "",
            "utterance_count": len(turns),
            "source_file_path": str(source_path),
            "data_provenance": {
                "annotation_source_dir": str(source_by_model[sampled_model].source_dir),
                "ed_source_path": meta.get("ed_source_path", ""),
                "annotation_mode": row.get("annotation_mode", ""),
            },
            "random_seed": seed,
            "sampling_timestamp": timestamp,
            "turns": turns,
        }
        packages.append(package)
    if len(packages) != n:
        raise RuntimeError(f"Expected {n} packages after normalization, got {len(packages)}.")
    return packages


def text_hash(text: str, model_path: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    payload = json.dumps(
        {"text": normalized, "target_language": TARGET_LANGUAGE, "model_path": str(model_path)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def should_translate(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(re.search(r"[A-Za-z]", stripped))


def strip_translation_prefix(text: str) -> str:
    stripped = text.strip()
    patterns = [
        r"^翻译[:：]\s*",
        r"^中文[:：]\s*",
        r"^Translation:\s*",
        r"^Here is the translation:\s*",
        r"^The translation is:\s*",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE).strip()
    return stripped.strip("\"' \n\r\t")


def clean_translation_output(text: str, source_text: str) -> str:
    """Remove common instruction-following artifacts from local model translations.

    The annotation UI shows the English source next to the Chinese text, so for
    single-utterance inputs we keep the first translated paragraph and remove
    grammar notes, pinyin-only parentheticals, and echoed English snippets.
    """
    cleaned = strip_translation_prefix(text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"^```(?:[a-zA-Z]+)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    if "\n\n" in cleaned:
        cleaned = cleaned.split("\n\n", 1)[0].strip()
    if "\n" not in (source_text or "") and "\n" in cleaned:
        cleaned = next((line.strip() for line in cleaned.splitlines() if line.strip()), cleaned.strip())
    cleaned = re.sub(r"\(([A-Za-z0-9 ,.;:'?!\\-]+)\)", "", cleaned).strip()
    cleaned = re.sub(r"^[.。]\s*", "", cleaned).strip()
    return cleaned or text.strip()


def resolve_model_path(project_root: Path, provided: str | None) -> Path:
    candidates: list[Path] = []
    if provided:
        candidates.append(Path(provided))
    if os.environ.get("MISTRAL_MODEL_PATH"):
        candidates.append(Path(os.environ["MISTRAL_MODEL_PATH"]))
    candidates.extend(
        [
            project_root / "models/Mistral-7B-Instruct-v0.3",
            project_root / "Mistral-7B-Instruct-v0.3",
            project_root / "checkpoints/Mistral-7B-Instruct-v0.3",
        ]
    )
    hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if hf_home:
        candidates.extend(Path(hf_home).glob("**/Mistral-7B-Instruct-v0.3"))
    for candidate in candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Local Mistral-7B-Instruct-v0.3 model not found. Set MISTRAL_MODEL_PATH=/path/to/Mistral-7B-Instruct-v0.3 "
        "or pass --model_path."
    )


class MistralTranslator:
    def __init__(self, project_root: Path, out_dir: Path, model_path_arg: str | None, batch_size: int) -> None:
        self.project_root = project_root
        self.model_path = resolve_model_path(project_root, model_path_arg)
        self.batch_size = max(1, batch_size)
        self.cache_path = out_dir / "translations" / "mistral_zh_translation_cache.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict[str, Any]] = {}
        changed_cache = False
        if self.cache_path.exists():
            for line in self.cache_path.open("r", encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    cleaned = clean_translation_output(row.get("zh_translation", ""), row.get("source_text", ""))
                    if cleaned != row.get("zh_translation", ""):
                        row["zh_translation"] = cleaned
                        changed_cache = True
                    self.cache[row["source_text_hash"]] = row
        if changed_cache:
            with self.cache_path.open("w", encoding="utf-8") as f:
                for row in self.cache.values():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.model = None
        self.tokenizer = None

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Translation requires torch and transformers in the current Python environment. "
                "Use the project virtual environment or install them locally; no remote API is used."
            ) from exc
        print(f"Loading local Mistral translation model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def prompt(self, text: str) -> str:
        assert self.tokenizer is not None
        messages = [
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": TRANSLATION_USER_PREFIX + text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return (
            f"[INST] <<SYS>>\n{TRANSLATION_SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{TRANSLATION_USER_PREFIX}{text} [/INST]"
        )

    def translate_texts(self, texts: list[str]) -> tuple[dict[str, str], int]:
        unique: list[str] = []
        seen: set[str] = set()
        model_path_text = str(self.model_path)
        translations: dict[str, str] = {}
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            key = text_hash(text, model_path_text)
            if not should_translate(text):
                translations[text] = text
            elif key in self.cache:
                translations[text] = self.cache[key]["zh_translation"]
            else:
                unique.append(text)
        if not unique:
            return translations, 0
        self.ensure_model()
        assert self.model is not None and self.tokenizer is not None
        import torch

        generation_config = {"do_sample": False, "max_new_tokens": "dynamic_cap_192"}
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            prompts = [self.prompt(text) for text in batch]
            print(f"Translating batch {start + 1}-{min(start + self.batch_size, len(unique))}/{len(unique)}", flush=True)
            source_lengths = [
                len(self.tokenizer(text, add_special_tokens=False).get("input_ids", []))
                for text in batch
            ]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            prompt_width = inputs["input_ids"].shape[1]
            max_source_len = max(source_lengths) if source_lengths else 32
            max_new_tokens = min(192, max(32, int(max_source_len * 1.6) + 24))
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            for text, output_ids in zip(batch, outputs):
                generated = output_ids[int(prompt_width) :]
                zh = self.tokenizer.decode(generated, skip_special_tokens=True)
                zh = clean_translation_output(zh, text)
                translations[text] = zh
                row = {
                    "source_text_hash": text_hash(text, model_path_text),
                    "source_text": text,
                    "zh_translation": zh,
                    "target_language": TARGET_LANGUAGE,
                    "model_path": model_path_text,
                    "generation_config": {
                        "do_sample": False,
                        "max_new_tokens": max_new_tokens,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.cache[row["source_text_hash"]] = row
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Translated {min(start + self.batch_size, len(unique))}/{len(unique)} new texts", flush=True)
        return translations, len(unique)


def collect_texts(packages: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for package in packages:
        if package.get("situation"):
            texts.append(package["situation"])
        for turn in package["turns"]:
            texts.append(turn["english_original_text"])
    return texts


def attach_translations(packages: list[dict[str, Any]], translations: dict[str, str]) -> None:
    for package in packages:
        package["situation_zh"] = translations.get(package.get("situation", ""), package.get("situation", ""))
        for turn in package["turns"]:
            src = turn["english_original_text"]
            turn["zh_translation"] = translations.get(src, src)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest_csv(path: Path, packages: list[dict[str, Any]]) -> None:
    fields = [
        "package_id",
        "dialogue_id",
        "split",
        "sampled_model",
        "original_emotion",
        "utterance_count",
        "source_file_path",
        "random_seed",
        "sampling_timestamp",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for package in packages:
            writer.writerow({field: package.get(field, "") for field in fields})


HTML_STYLE = """
:root {
  --bg: #f6f7fb;
  --panel: #ffffff;
  --ink: #1f2937;
  --muted: #667085;
  --line: #d7dee8;
  --soft: #eef3f8;
  --accent: #1f5f7a;
  --warn: #9a3412;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 Arial, "Noto Sans SC", sans-serif; }
.app { max-width: 1120px; margin: 0 auto; padding: 22px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px 20px; margin-bottom: 14px; }
h1 { font-size: 24px; margin: 0 0 10px; }
h2 { font-size: 18px; margin: 16px 0 8px; }
h3 { color: #475467; font-size: 14px; margin: 12px 0 6px; }
.muted { color: var(--muted); }
.small { font-size: 13px; }
.box { background: var(--soft); border: 1px solid #e1e8f0; border-radius: 8px; padding: 10px 12px; white-space: pre-wrap; }
.turn { border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 12px 0; background: #fff; }
.turn.active { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(31, 95, 122, .12); }
.turn-head { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: space-between; }
.emoji { font-size: 28px; padding: 4px 10px; border-radius: 8px; background: #fff7ed; border: 1px solid #fed7aa; }
.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; }
.labels { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 8px; }
label.option { border: 1px solid #bac7d8; border-radius: 8px; padding: 8px 10px; cursor: pointer; background: #fff; }
label.option:hover { background: #edf4fb; }
select, textarea { width: 100%; border: 1px solid #bac7d8; border-radius: 8px; padding: 8px 10px; font: inherit; }
textarea { min-height: 56px; resize: vertical; }
button { border: 1px solid #bac7d8; background: #fff; color: var(--ink); border-radius: 8px; padding: 9px 12px; font: inherit; cursor: pointer; }
button:hover { background: #edf4fb; }
button.primary { background: var(--accent); border-color: var(--accent); color: white; }
button.danger { color: var(--warn); border-color: #fdba74; }
.nav { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; align-items: center; }
.nav-group { display: flex; gap: 8px; flex-wrap: wrap; }
.progressbar { height: 10px; background: #e5ebf2; border-radius: 999px; overflow: hidden; }
.progressbar > div { height: 100%; width: 0; background: var(--accent); }
.hidden { display: none; }
@media (max-width: 760px) { .app { padding: 12px; } .columns { grid-template-columns: 1fr; } }
"""


def html_escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def render_html(path: Path, annotator_id: str, packages: list[dict[str, Any]], seed: int) -> None:
    rng = random.Random(seed)
    ordered = packages[:]
    rng.shuffle(ordered)
    data_json = json.dumps(ordered, ensure_ascii=False)
    html_text = f"""<!doctype html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Emoji 弱标注人工审核</title>
  <style>{HTML_STYLE}</style>
</head>
<body>
<main class="app">
  <section class="card">
    <h1>Emoji 弱标注人工审核</h1>
    <p>请逐轮判断显示的 emoji 是否能够合理表达该轮话语在当前对话中的情感姿态、人际态度或 conversational stance。不要推断说话人的真实心理状态，只判断这句话在对话中表达出来的姿态。</p>
    <p>如果 emoji 可以接受，请保持“合理”。</p>
    <p>如果 emoji 不是最佳选择，但仍然说得通，请标为“可疑但可接受”。</p>
    <p>如果 emoji 与该轮话语的语义、语气或上下文明显不符，请标为“明显不合理”。</p>
    <p class="small muted">For each turn, judge whether the displayed emoji plausibly expresses the affective stance, interpersonal stance, or conversational attitude of the utterance in this dialogue context. Do not infer the speaker’s true mental state. Judge only the expressed conversational stance.</p>
    <p class="small muted">标注者：{html_escape(annotator_id)}。快捷键：1 = 合理，2 = 可疑但可接受，3 = 明显不合理。</p>
    <div class="progressbar"><div id="progress-bar"></div></div>
    <p id="progress-text" class="small muted"></p>
  </section>
  <section id="package-card" class="card"></section>
  <section class="card">
    <div class="nav">
      <div class="nav-group">
        <button id="prev-btn">上一个对话</button>
        <button id="next-btn" class="primary">下一个对话</button>
        <button id="review-btn">标记本对话已检查</button>
      </div>
      <div class="nav-group">
        <button id="export-jsonl">导出 JSONL</button>
        <button id="export-csv">导出 CSV</button>
        <button id="export-summary">导出进度摘要</button>
        <button id="clear-local" class="danger">清除本地保存记录</button>
      </div>
    </div>
    <p id="status-text" class="small muted"></p>
  </section>
</main>
<script type="application/json" id="audit-data">{data_json}</script>
<script>
const ANNOTATOR_ID = {json.dumps(annotator_id)};
const STORAGE_KEY = "emoji_audit_" + ANNOTATOR_ID + "_v1";
const PACKAGES = JSON.parse(document.getElementById("audit-data").textContent);
const LABELS = ["reasonable", "questionable_but_acceptable", "clearly_unreasonable"];
const REASONS = [
  ["", "请选择原因（可选）"],
  ["wrong_emotion_or_tone", "情绪或语气不符"],
  ["too_strong", "情绪过强"],
  ["too_weak", "情绪过弱"],
  ["context_mismatch", "与上下文不符"],
  ["non_affective_or_irrelevant", "非情感表达或无关"],
  ["culturally_unclear", "文化含义不明确"],
  ["other", "其他"]
];
let current = 0;
let activeTurn = 0;
let state = loadState();

function escapeHtml(value) {{
  return String(value ?? "").replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{answers: {{}}, reviewed: {{}}}}; }}
  catch (err) {{ return {{answers: {{}}, reviewed: {{}}}}; }}
}}
function saveState() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); }}
function keyFor(pkg, turn) {{ return pkg.package_id + "::" + turn.turn_index; }}
function defaultAnswer(pkg, turn) {{
  const key = keyFor(pkg, turn);
  if (!state.answers[key]) {{
    state.answers[key] = {{
      label: "reasonable",
      reason_code: "",
      note: "",
      timestamp: new Date().toISOString()
    }};
  }}
  return state.answers[key];
}}
function render() {{
  const pkg = PACKAGES[current];
  activeTurn = Math.min(activeTurn, pkg.turns.length - 1);
  const situationBlock = pkg.situation ? `
    <h3>Situation / 情境</h3>
    <div class="columns"><div class="box">${{escapeHtml(pkg.situation)}}</div><div class="box">${{escapeHtml(pkg.situation_zh)}}</div></div>` : "";
  document.getElementById("package-card").innerHTML = `
    <p class="small muted">对话 ${{current + 1}} / ${{PACKAGES.length}} · Package ID: ${{escapeHtml(pkg.package_id)}} · Split: ${{escapeHtml(pkg.split)}}</p>
    ${{situationBlock}}
    <h2>完整对话</h2>
    <div id="turns"></div>
  `;
  const turns = document.getElementById("turns");
  pkg.turns.forEach((turn, idx) => {{
    const ans = defaultAnswer(pkg, turn);
    const reasonVisible = ans.label === "reasonable" ? "hidden" : "";
    const options = LABELS.map(label => {{
      const zh = label === "reasonable" ? "合理" : label === "questionable_but_acceptable" ? "可疑但可接受" : "明显不合理";
      return `<label class="option"><input type="radio" name="label_${{pkg.package_id}}_${{turn.turn_index}}" value="${{label}}" ${{ans.label === label ? "checked" : ""}}> ${{zh}}</label>`;
    }}).join("");
    const reasonOptions = REASONS.map(([code, zh]) => `<option value="${{code}}" ${{ans.reason_code === code ? "selected" : ""}}>${{zh}}</option>`).join("");
    const div = document.createElement("div");
    div.className = "turn" + (idx === activeTurn ? " active" : "");
    div.dataset.turnIndex = String(idx);
    div.innerHTML = `
      <div class="turn-head">
        <strong>Turn ${{turn.turn_index}} · Speaker ${{escapeHtml(turn.role)}}</strong>
        <span class="emoji" title="displayed emoji">${{escapeHtml(turn.displayed_emoji || "MISSING")}}</span>
      </div>
      <div class="columns">
        <div><h3>English original</h3><div class="box">${{escapeHtml(turn.english_original_text)}}</div></div>
        <div><h3>中文译文</h3><div class="box">${{escapeHtml(turn.zh_translation)}}</div></div>
      </div>
      <div class="labels">${{options}}</div>
      <div class="reason ${{reasonVisible}}">
        <select class="reason-select">${{reasonOptions}}</select>
      </div>
      <h3>备注（可选）</h3>
      <textarea class="note-box" placeholder="可选备注">${{escapeHtml(ans.note || "")}}</textarea>
    `;
    turns.appendChild(div);
    div.addEventListener("click", () => {{ activeTurn = idx; markActive(); }});
    div.querySelectorAll("input[type=radio]").forEach(input => {{
      input.addEventListener("change", () => {{
        const answer = defaultAnswer(pkg, turn);
        answer.label = input.value;
        if (answer.label === "reasonable") answer.reason_code = "";
        answer.timestamp = new Date().toISOString();
        saveState();
        render();
      }});
    }});
    div.querySelector(".reason-select").addEventListener("change", event => {{
      const answer = defaultAnswer(pkg, turn);
      answer.reason_code = event.target.value;
      answer.timestamp = new Date().toISOString();
      saveState();
    }});
    div.querySelector(".note-box").addEventListener("input", event => {{
      const answer = defaultAnswer(pkg, turn);
      answer.note = event.target.value;
      answer.timestamp = new Date().toISOString();
      saveState();
    }});
  }});
  document.getElementById("prev-btn").disabled = current === 0;
  updateProgress();
}}
function markActive() {{
  document.querySelectorAll(".turn").forEach((el, idx) => el.classList.toggle("active", idx === activeTurn));
}}
function setActiveLabel(label) {{
  const pkg = PACKAGES[current];
  const turn = pkg.turns[activeTurn];
  const ans = defaultAnswer(pkg, turn);
  ans.label = label;
  if (label === "reasonable") ans.reason_code = "";
  ans.timestamp = new Date().toISOString();
  saveState();
  render();
}}
function packageReviewed(pkg) {{ return !!state.reviewed[pkg.package_id]; }}
function updateProgress() {{
  const reviewed = PACKAGES.filter(pkg => packageReviewed(pkg)).length;
  const pct = PACKAGES.length ? reviewed / PACKAGES.length * 100 : 0;
  document.getElementById("progress-bar").style.width = pct + "%";
  document.getElementById("progress-text").textContent = `已检查对话：${{reviewed}} / ${{PACKAGES.length}}`;
  document.getElementById("status-text").textContent = "所有标注会自动保存在本浏览器 localStorage。";
}}
function rowsForExport() {{
  const rows = [];
  PACKAGES.forEach((pkg, orderIdx) => {{
    pkg.turns.forEach(turn => {{
      const ans = defaultAnswer(pkg, turn);
      rows.push({{
        annotator_id: ANNOTATOR_ID,
        package_id: pkg.package_id,
        package_order_index: orderIdx + 1,
        dialogue_id: pkg.dialogue_id,
        split: pkg.split,
        sampled_model: pkg.sampled_model,
        original_emotion: pkg.original_emotion || "",
        turn_index: turn.turn_index,
        role: turn.role,
        english_original_text: turn.english_original_text,
        zh_translation: turn.zh_translation,
        displayed_emoji: turn.displayed_emoji,
        confidence_hidden_if_available: turn.confidence_hidden_if_available ?? "",
        confidence_bin: turn.confidence_bin || "missing",
        label: ans.label || "reasonable",
        reason_code: ans.reason_code || "",
        note: ans.note || "",
        reviewed_package_boolean: !!state.reviewed[pkg.package_id],
        timestamp: ans.timestamp || new Date().toISOString()
      }});
    }});
  }});
  return rows;
}}
function download(filename, text, type) {{
  const blob = new Blob([text], {{type}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}}
function csvEscape(value) {{
  const text = String(value ?? "");
  return /[",\\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
}}
function exportJsonl() {{
  const lines = rowsForExport().map(row => JSON.stringify(row)).join("\\n") + "\\n";
  download(`emoji_audit_${{ANNOTATOR_ID}}.jsonl`, lines, "application/jsonl");
}}
function exportCsv() {{
  const rows = rowsForExport();
  const cols = Object.keys(rows[0] || {{}});
  const text = [cols.join(",")].concat(rows.map(row => cols.map(col => csvEscape(row[col])).join(","))).join("\\n") + "\\n";
  download(`emoji_audit_${{ANNOTATOR_ID}}.csv`, text, "text/csv");
}}
function exportSummary() {{
  const rows = rowsForExport();
  const counts = rows.reduce((acc, row) => {{ acc[row.label] = (acc[row.label] || 0) + 1; return acc; }}, {{}});
  const reviewed = PACKAGES.filter(pkg => packageReviewed(pkg)).length;
  const summary = {{annotator_id: ANNOTATOR_ID, packages: PACKAGES.length, reviewed_packages: reviewed, turn_labels: counts, export_time: new Date().toISOString()}};
  download(`emoji_audit_${{ANNOTATOR_ID}}_progress_summary.json`, JSON.stringify(summary, null, 2) + "\\n", "application/json");
}}
document.getElementById("prev-btn").addEventListener("click", () => {{ if (current > 0) {{ current--; activeTurn = 0; render(); }} }});
document.getElementById("next-btn").addEventListener("click", () => {{ if (current < PACKAGES.length - 1) {{ current++; activeTurn = 0; render(); }} }});
document.getElementById("review-btn").addEventListener("click", () => {{ state.reviewed[PACKAGES[current].package_id] = true; saveState(); updateProgress(); }});
document.getElementById("export-jsonl").addEventListener("click", exportJsonl);
document.getElementById("export-csv").addEventListener("click", exportCsv);
document.getElementById("export-summary").addEventListener("click", exportSummary);
document.getElementById("clear-local").addEventListener("click", () => {{ if (confirm("确认清除本地保存记录？")) {{ localStorage.removeItem(STORAGE_KEY); state = loadState(); render(); }} }});
document.addEventListener("keydown", event => {{
  if (event.target && ["TEXTAREA", "INPUT", "SELECT"].includes(event.target.tagName)) return;
  if (event.key === "1") setActiveLabel("reasonable");
  if (event.key === "2") setActiveLabel("questionable_but_acceptable");
  if (event.key === "3") setActiveLabel("clearly_unreasonable");
}});
render();
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def write_readme(out_dir: Path) -> None:
    text = """# Emoji 弱标注人工审核系统

本目录用于构建离线人工审核材料，目标是抽样检查四个 LLM annotator 给出的 emoji 弱标签是否在完整对话语境中合理。该任务不是 gold emotion labeling，也不要求标注者判断说话人的真实心理状态；它只审核显示的 emoji 是否能合理表达该轮话语在对话中呈现出来的情感姿态、人际态度或 conversational stance。

## 抽样方式

`build_human_audit.py` 会自动发现 `data/*/*_emoji_annotations.json` 中的四个模型标注文件，并按 finalized dialogue-level train/valid/test split 的实际 dialogue 数量比例抽取 300 个完整对话。每个 package 是“一个完整 dialogue + 一个 sampled LLM annotator model”。四个模型在 300 个 package 中均衡，每个模型 75 个 package。模型名只保存在隐藏元数据和导出文件中，不显示给标注者。

## 设置本地 Mistral 模型

脚本只使用本地模型，不调用远程 API。模型查找顺序包括命令行 `--model_path`、环境变量 `MISTRAL_MODEL_PATH`、`./models/Mistral-7B-Instruct-v0.3`、`./Mistral-7B-Instruct-v0.3` 和 `./checkpoints/Mistral-7B-Instruct-v0.3`。

如果脚本提示找不到模型，可以运行：

```bash
export MISTRAL_MODEL_PATH=/path/to/Mistral-7B-Instruct-v0.3
```

## 构建 HTML

推荐先运行 smoke test：

```bash
python data_eval/build_human_audit.py --smoke_test_n 6 --seed 20260519
python data_eval/validate_audit_outputs.py --out_dir data_eval --allow_smoke_test
```

正式构建：

```bash
python data_eval/build_human_audit.py --n 300 --seed 20260519
python data_eval/validate_audit_outputs.py --out_dir data_eval
```

如果当前 shell 的 `python` 没有 torch/transformers，请使用项目虚拟环境，例如 `.venv/bin/python`。

## 标注者如何使用

三位标注者分别打开：

- `data_eval/html/annotator_01.html`
- `data_eval/html/annotator_02.html`
- `data_eval/html/annotator_03.html`

HTML 是完全离线的，可以双击打开，不需要服务器和网络。每个页面包含相同 300 个 package，但顺序不同。页面展示英文原文和本地 Mistral 生成的简体中文译文；中文只是阅读辅助，英文原文始终可见。

## 导出

标注完成后点击：

- `导出 JSONL`
- 或 `导出 CSV`

导出的每一行是一个 utterance-level 判断，包含 hidden sampled_model、confidence、split、dialogue_id、turn_index、label、reason_code 和 note。

## 聚合

收齐三份 JSONL 后运行：

```bash
python data_eval/aggregate_human_audit.py \\
  --inputs path/to/emoji_audit_annotator_01.jsonl path/to/emoji_audit_annotator_02.jsonl path/to/emoji_audit_annotator_03.jsonl \\
  --out_dir data_eval/results
```

聚合输出中的指标含义：

- `p_valid`：多数判断为合理的比例
- `p_ambiguous`：多数为可疑但可接受，或三类标签分歧的比例
- `p_invalid`：至少两位标注者认为明显不合理的比例
- `p_plausible`：`p_valid + p_ambiguous`

这些指标估计弱标签在代表性样本上的可接受性，不是完整语料每个稀有 emoji 或边界 case 的穷尽验证。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def write_paper_methods_snippet(out_dir: Path) -> None:
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    snippet = (
        "We conduct a representative dialogue-level human audit on 300 sampled model-dialogue packages. "
        "Packages are sampled from the finalized dialogue-level split in proportion to train, development, "
        "and test sizes, with the four LLM annotator models balanced across packages. Annotators view one "
        "complete dialogue at a time and inspect the emoji assigned by the sampled model to each utterance. "
        "For annotation convenience, the interface displays both the original English text and a Simplified "
        "Chinese translation produced by a local Mistral-7B-Instruct-v0.3 model. Annotators flag turns as "
        "questionable but plausible or clearly unreasonable with respect to the utterance meaning and dialogue "
        "context. Each package is reviewed by three independent annotators. We report turn-level valid, "
        "ambiguous, invalid, and plausible rates with package-clustered bootstrap confidence intervals. This "
        "audit estimates representative corpus-level weak-label validity and is not intended as exhaustive "
        "validation of every rare emoji or boundary-cluster case.\n"
    )
    (results_dir / "paper_methods_snippet.md").write_text(snippet, encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    project_root = project_root_from_script()
    base_out_dir = Path(args.out_dir)
    out_dir = base_out_dir / "smoke_test" if args.smoke_test_n else base_out_dir
    n = args.smoke_test_n if args.smoke_test_n else args.n
    out_dir.mkdir(parents=True, exist_ok=True)

    final_manifest = out_dir / "audit_sample_300_model_dialogue_packages.jsonl"
    if final_manifest.exists() and not args.force_resample and not args.reuse_sample:
        raise FileExistsError(
            f"Sample already exists at {final_manifest}. Use --reuse_sample to reuse or --force_resample to overwrite."
        )

    sources = discover_model_sources(project_root)
    all_records = load_all_model_records(sources)
    emotion_meta = load_emotion_metadata(project_root)
    model_names = [source.canonical_name for source in sources]

    if args.reuse_sample and final_manifest.exists():
        print(f"Reusing existing sample: {final_manifest}")
        packages = [json.loads(line) for line in final_manifest.open("r", encoding="utf-8") if line.strip()]
        sampling_summary = {"reused_sample": True, "sampled_dialogues": len(packages)}
    else:
        sampled_dialogues, sampling_summary = sample_dialogues(all_records, n, args.seed)
        packages = make_packages(sampled_dialogues, model_names, all_records, sources, emotion_meta, args.seed, n)

    translator = MistralTranslator(project_root, base_out_dir, args.model_path, args.translation_batch_size)
    translations, translated_new = translator.translate_texts(collect_texts(packages))
    attach_translations(packages, translations)

    write_jsonl(out_dir / "audit_sample_300_model_dialogue_packages.jsonl", packages)
    write_manifest_csv(out_dir / "audit_sample_manifest.csv", packages)
    (out_dir / "sampling_summary.json").write_text(json.dumps(sampling_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    html_dir = out_dir / "html"
    for idx in range(1, 4):
        render_html(html_dir / f"annotator_{idx:02d}.html", f"annotator_{idx:02d}", packages, args.seed + idx * 1009)
    write_readme(base_out_dir)
    write_paper_methods_snippet(base_out_dir)

    split_counts = {}
    model_counts = {}
    for package in packages:
        split_counts[package["split"]] = split_counts.get(package["split"], 0) + 1
        model_counts[package["sampled_model"]] = model_counts.get(package["sampled_model"], 0) + 1
    unique_texts = len(set(collect_texts(packages)))
    print("Done.")
    print(f"Discovered dataset path: {project_root / 'data'}")
    print(f"Number of dialogues in common split pools: {sampling_summary.get('dialogue_level_common_split_counts')}")
    print(f"Number of sampled packages: {len(packages)}")
    print(f"Split distribution: {split_counts}")
    print(f"Model distribution: {model_counts}")
    print(f"Unique English texts in sample: {unique_texts}")
    print(f"New unique texts translated into Chinese: {translated_new}")
    print("Output HTML paths:")
    for idx in range(1, 4):
        print(f"  - {html_dir / f'annotator_{idx:02d}.html'}")


def main() -> None:
    try:
        build(parse_args())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
