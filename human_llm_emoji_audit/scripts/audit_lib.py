from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SCHEMA_VERSION = "emostance_human_emoji_distribution_v1"
EXPERIMENT_ID = "human_llm_emoji_audit_test_120_seed42"
DEFAULT_MODELS = ("DeepSeek-V3.2", "claude-sonnet-4-6", "gemini-2.5-pro", "gpt-5.4")
DEFAULT_ANNOTATOR_SEEDS = (4201, 4202, 4203)
STRATA = ("low", "medium", "high")
QUESTION_TEXT = (
    "Given the situation, dialogue context, speaker role, and highlighted utterance, "
    "select the single emoji that best reflects the utterance's affective stance, "
    "interpersonal stance, or conversational attitude in context."
)
HELP_TEXT = (
    "More than one emoji may be plausible. Select the one you consider most appropriate. "
    "Then provide a confidence score from 1 to 5, where 1 means highly uncertain and 5 means "
    "highly confident. The goal is not to infer the speaker's true mental state or identify "
    "a gold emotion label."
)

PRIVATE_FIELD_TOKENS = (
    "llm_annotations",
    "llm_qE",
    "qE",
    "raw_entropy",
    "normalized_entropy",
    "disagreement_stratum",
    "qZ",
    "latent_region",
    "membership_matrix",
    "original_emotion",
    "gold_stance",
    "model_name",
)

PUBLIC_ITEM_KEYS = {
    "sample_id",
    "situation",
    "context_turns",
    "target_turn",
    "target_turn_index",
    "target_role",
}


class ExperimentError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def codepoint_string(text: str) -> str:
    return " ".join(f"U+{ord(char):04X}" for char in text)


def strip_variation_selectors(text: str) -> str:
    return text.replace("\ufe0f", "")


def add_text_presentation(text: str) -> str:
    return "".join(char + "\ufe0f" if char in {"\u2639", "\u263a", "\u2642", "\u2640", "\U0001F32B"} else char for char in text)


SKIN_TONES = {chr(code) for code in range(0x1F3FB, 0x1F400)}


def remove_skin_gender(text: str) -> str:
    stripped = "".join(char for char in text if char not in SKIN_TONES)
    stripped = stripped.replace("\u200d\u2642", "").replace("\u200d\u2640", "")
    stripped = stripped.replace("\u2642", "").replace("\u2640", "")
    return stripped


def load_inventory(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ExperimentError(f"Inventory must be a JSON list: {path}")
    seen: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict) or not row.get("emoji"):
            raise ExperimentError(f"Invalid inventory row {idx} in {path}")
        emoji = str(row["emoji"])
        if emoji in seen:
            raise ExperimentError(f"Duplicate emoji in inventory: {emoji!r}")
        seen.add(emoji)
        inventory.append(
            {
                "emoji": emoji,
                "unicode": row.get("unicode") or codepoint_string(emoji),
            }
        )
    return inventory


def load_membership_rows(path: Path, column: str = "membership_raw") -> tuple[dict[str, dict[str, float]], list[str]]:
    membership: dict[str, dict[str, float]] = defaultdict(dict)
    clusters: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"emoji", "cluster_id", column}
        if not required.issubset(reader.fieldnames or []):
            raise ExperimentError(f"Membership CSV {path} lacks required columns {sorted(required)}")
        for row in reader:
            emoji = row["emoji"]
            cluster = row["cluster_id"]
            value = float(row[column])
            if value < -1e-12:
                raise ExperimentError(f"Negative membership value for {emoji} / {cluster}")
            membership[emoji][cluster] = max(value, 0.0)
            clusters.add(cluster)
    ordered_clusters = sorted(clusters)
    for emoji, values in membership.items():
        total = sum(values.values())
        if total <= 0:
            raise ExperimentError(f"Membership row for {emoji!r} sums to zero")
        for cluster in ordered_clusters:
            values.setdefault(cluster, 0.0)
        for cluster in ordered_clusters:
            values[cluster] /= total
    return dict(membership), ordered_clusters


def resolve_membership(
    emoji: str,
    membership: dict[str, dict[str, float]],
    clusters: Sequence[str],
) -> tuple[dict[str, float], str, str | None]:
    if emoji in membership:
        return dict(membership[emoji]), "exact", None

    candidates = [
        strip_variation_selectors(emoji),
        add_text_presentation(emoji),
        add_text_presentation(strip_variation_selectors(emoji)),
        remove_skin_gender(emoji),
        add_text_presentation(remove_skin_gender(emoji)),
    ]

    manual_aliases = {
        "🌜": "🌛",
        "😗": "😚",
        "😙": "😚",
        "😶\u200d🌫": "😶\u200d🌫️",
    }
    if emoji in manual_aliases:
        candidates.insert(0, manual_aliases[emoji])

    for candidate in candidates:
        if candidate and candidate in membership:
            return dict(membership[candidate]), "unicode_alias", candidate

    uniform = 1.0 / len(clusters)
    return {cluster: uniform for cluster in clusters}, "unobserved_uniform", None


def make_inventory_with_membership(
    inventory: Sequence[dict[str, Any]],
    membership: dict[str, dict[str, float]],
    clusters: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in inventory:
        emoji = row["emoji"]
        values, source, alias = resolve_membership(emoji, membership, clusters)
        result.append(
            {
                "emoji": emoji,
                "unicode": row.get("unicode") or codepoint_string(emoji),
                "membership": {cluster: float(values.get(cluster, 0.0)) for cluster in clusters},
                "membership_source": source,
                "membership_alias": alias,
            }
        )
    return result


def membership_matrix(inventory_rows: Sequence[dict[str, Any]], clusters: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((len(inventory_rows), len(clusters)), dtype=float)
    for row_idx, row in enumerate(inventory_rows):
        values = row.get("membership") or {}
        for cluster_idx, cluster in enumerate(clusters):
            matrix[row_idx, cluster_idx] = float(values.get(cluster, 0.0))
        total = float(matrix[row_idx].sum())
        if total <= 0:
            raise ExperimentError(f"Membership row for {row.get('emoji')} sums to zero")
        matrix[row_idx] /= total
    return matrix


def q_from_weighted_votes(votes: Sequence[tuple[str, float]], emojis: Sequence[str]) -> dict[str, float]:
    if not votes:
        raise ExperimentError("Cannot aggregate qE from no votes")
    emoji_set = set(emojis)
    totals: dict[str, float] = defaultdict(float)
    denom = 0.0
    for emoji, confidence in votes:
        if emoji not in emoji_set:
            raise ExperimentError(f"Vote emoji {emoji!r} is outside the fixed candidate inventory")
        if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
            raise ExperimentError(f"Invalid confidence for {emoji!r}: {confidence!r}")
        confidence = float(confidence)
        if confidence < 1 or confidence > 5:
            raise ExperimentError(f"Confidence must be 1..5, got {confidence!r} for {emoji!r}")
        totals[emoji] += confidence
        denom += confidence
    if denom <= 0:
        raise ExperimentError("Confidence denominator is zero")
    return {emoji: weight / denom for emoji, weight in sorted(totals.items()) if weight > 0}


def distribution_vector(dist: dict[str, float], emojis: Sequence[str]) -> np.ndarray:
    index = {emoji: idx for idx, emoji in enumerate(emojis)}
    vector = np.zeros(len(emojis), dtype=float)
    for emoji, value in dist.items():
        if emoji not in index:
            raise ExperimentError(f"Distribution emoji {emoji!r} not in inventory")
        vector[index[emoji]] += float(value)
    total = float(vector.sum())
    if total <= 0:
        raise ExperimentError("Distribution sums to zero")
    return vector / total


def dict_from_vector(vector: np.ndarray, labels: Sequence[str], *, drop_zeros: bool = True) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, value in zip(labels, vector.tolist()):
        value = float(value)
        if not drop_zeros or value > 1e-12:
            result[str(label)] = value
    return result


def entropy_natural(probabilities: Iterable[float]) -> float:
    total = 0.0
    for probability in probabilities:
        p = float(probability)
        if p > 0:
            total -= p * math.log(p)
    return total


def normalized_vote_entropy(qe: dict[str, float]) -> tuple[float, float]:
    raw = entropy_natural(qe.values())
    return raw, raw / math.log(4)


def top_set(vector: Sequence[float], eps: float = 1e-12) -> set[int]:
    arr = np.asarray(vector, dtype=float)
    if arr.size == 0:
        return set()
    max_value = float(arr.max())
    return {idx for idx, value in enumerate(arr.tolist()) if abs(value - max_value) <= eps}


def top_label(dist: dict[str, float]) -> str:
    if not dist:
        raise ExperimentError("Empty distribution has no top label")
    max_value = max(dist.values())
    labels = sorted(label for label, value in dist.items() if abs(value - max_value) <= 1e-12)
    return labels[0]


def project_qz(qe: dict[str, float], emojis: Sequence[str], matrix: np.ndarray, clusters: Sequence[str]) -> dict[str, float]:
    vector = distribution_vector(qe, emojis)
    projected = vector @ matrix
    total = float(projected.sum())
    if total <= 0:
        raise ExperimentError("Projected region distribution sums to zero")
    projected = projected / total
    return dict_from_vector(projected, clusters)


def load_llm_annotations(project_root: Path, split: str, models: Sequence[str]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        path = project_root / "data" / model / f"{split}_emoji_annotations.json"
        records = read_json(path)
        if not isinstance(records, list):
            raise ExperimentError(f"LLM annotation file must be a list: {path}")
        by_dialogue: dict[str, Any] = {}
        for record in records:
            dialogue_id = str(record.get("dialogue_id", ""))
            if not dialogue_id:
                raise ExperimentError(f"Missing dialogue_id in {path}")
            if dialogue_id in by_dialogue:
                raise ExperimentError(f"Duplicate dialogue_id {dialogue_id} in {path}")
            by_dialogue[dialogue_id] = record
        by_model[model] = by_dialogue
    return by_model


def annotation_for_turn(model_record: dict[str, Any], turn_index: int, model: str, dialogue_id: str) -> dict[str, Any]:
    turns = model_record.get("turns")
    if not isinstance(turns, list):
        raise ExperimentError(f"Missing turns for {model} {dialogue_id}")
    for turn in turns:
        tid = int(turn.get("turn_id", turn.get("turn_index", -1)))
        if tid == turn_index:
            ann = turn.get("emoji_annotation")
            if not isinstance(ann, dict):
                raise ExperimentError(f"Missing emoji_annotation for {model} {dialogue_id} turn {turn_index}")
            if ann.get("error") not in (None, "", False):
                raise ExperimentError(f"Annotation error for {model} {dialogue_id} turn {turn_index}: {ann.get('error')}")
            emoji = ann.get("selected_emoji")
            confidence = ann.get("confidence")
            if emoji in (None, "") or confidence in (None, ""):
                raise ExperimentError(f"Missing selected_emoji/confidence for {model} {dialogue_id} turn {turn_index}")
            return {
                "model": model,
                "selected_emoji": str(emoji),
                "unicode": ann.get("unicode") or codepoint_string(str(emoji)),
                "confidence": int(confidence),
            }
    raise ExperimentError(f"Turn {turn_index} not found for {model} {dialogue_id}")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def role_transition(previous_role: str | None, target_role: str) -> str:
    if not previous_role:
        return f"START->{target_role}"
    return f"{previous_role}->{target_role}"


def build_candidate_pool(
    project_root: Path,
    split: str,
    inventory_rows: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    clusters: Sequence[str],
    models: Sequence[str] = DEFAULT_MODELS,
) -> list[dict[str, Any]]:
    split_path = project_root / "pre_data" / f"{split}.json"
    dialogues = read_json(split_path)
    if not isinstance(dialogues, list):
        raise ExperimentError(f"Split file must be a JSON list: {split_path}")
    llm_by_model = load_llm_annotations(project_root, split, models)
    emoji_list = [row["emoji"] for row in inventory_rows]
    inventory_set = set(emoji_list)
    candidates: list[dict[str, Any]] = []
    for dialogue in dialogues:
        if dialogue.get("split") != split:
            raise ExperimentError(f"Unexpected split in {split_path}: {dialogue.get('split')!r}")
        dialogue_id = str(dialogue.get("dialogue_id", ""))
        turns = dialogue.get("turns")
        if not dialogue_id or not isinstance(turns, list):
            raise ExperimentError(f"Malformed dialogue in {split_path}: {dialogue!r}")
        situation = clean_text(dialogue.get("situation"))
        if not situation:
            continue
        for turn_pos, turn in enumerate(turns):
            turn_index = int(turn.get("turn_index", turn.get("turn_id", turn_pos)))
            if turn_index < 1:
                continue
            target_text = clean_text(turn.get("utterance"))
            target_role = str(turn.get("role") or turn.get("speaker") or "")
            if not target_text or not target_role:
                continue
            previous_turns = []
            for prev in turns[:turn_pos]:
                prev_text = clean_text(prev.get("utterance"))
                prev_role = str(prev.get("role") or prev.get("speaker") or "")
                if prev_text and prev_role:
                    previous_turns.append(
                        {
                            "turn_index": int(prev.get("turn_index", prev.get("turn_id", len(previous_turns)))),
                            "role": prev_role,
                            "utterance": prev_text,
                        }
                    )
            if not previous_turns:
                continue
            previous_role = previous_turns[-1]["role"]
            annotations = []
            votes = []
            for model in models:
                model_records = llm_by_model[model]
                if dialogue_id not in model_records:
                    raise ExperimentError(f"Missing dialogue {dialogue_id} in {model} {split} annotations")
                ann = annotation_for_turn(model_records[dialogue_id], turn_index, model, dialogue_id)
                if ann["selected_emoji"] not in inventory_set:
                    raise ExperimentError(
                        f"{model} {dialogue_id} turn {turn_index} selected emoji outside inventory: {ann['selected_emoji']!r}"
                    )
                annotations.append(ann)
                votes.append((ann["selected_emoji"], float(ann["confidence"])))
            qe = q_from_weighted_votes(votes, emoji_list)
            raw_h, norm_h = normalized_vote_entropy(qe)
            qz = project_qz(qe, emoji_list, matrix, clusters)
            top1_region = top_label(qz)
            top1_emoji = top_label(qe)
            candidates.append(
                {
                    "dialogue_id": dialogue_id,
                    "target_turn_index": turn_index,
                    "target_role": target_role,
                    "previous_role": previous_role,
                    "role_transition": role_transition(previous_role, target_role),
                    "situation": situation,
                    "context_turns": previous_turns,
                    "target_turn": target_text,
                    "llm_annotations": annotations,
                    "llm_qE": qe,
                    "raw_entropy": raw_h,
                    "normalized_entropy": norm_h,
                    "top1_llm_emoji": top1_emoji,
                    "qZ": qz,
                    "top1_latent_region": top1_region,
                }
            )
    return candidates


def assign_tertiles(candidates: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["normalized_entropy"]),
            str(item["dialogue_id"]),
            int(item["target_turn_index"]),
        ),
    )
    n = len(ordered)
    base = n // 3
    remainder = n % 3
    sizes = [base + (1 if idx < remainder else 0) for idx in range(3)]
    result: dict[str, list[dict[str, Any]]] = {}
    start = 0
    for stratum, size in zip(STRATA, sizes):
        result[stratum] = [dict(item, disagreement_stratum=stratum) for item in ordered[start : start + size]]
        start += size
    return result


def select_stratified_samples(
    by_stratum: dict[str, list[dict[str, Any]]],
    n_items: int,
    seed: int,
) -> list[dict[str, Any]]:
    if n_items % 3 != 0:
        raise ExperimentError("--n-items must be divisible by three for low/medium/high strata")
    per_stratum = n_items // 3
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_dialogues: set[str] = set()
    region_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    emoji_counts: Counter[str] = Counter()

    for stratum in STRATA:
        pool = list(by_stratum[stratum])
        if len(pool) < per_stratum:
            raise ExperimentError(f"Stratum {stratum} has only {len(pool)} candidates, need {per_stratum}")
        random_rank = {id(item): rng.random() for item in pool}
        chosen: list[dict[str, Any]] = []
        while len(chosen) < per_stratum:
            available = [item for item in pool if item["dialogue_id"] not in used_dialogues and item not in chosen]
            if not available:
                raise ExperimentError(f"Could not select {per_stratum} unique-dialogue samples for {stratum}")

            def score(item: dict[str, Any]) -> tuple[float, int, int, int, float, str, int]:
                region = str(item["top1_latent_region"])
                trans = str(item["role_transition"])
                emoji = str(item["top1_llm_emoji"])
                return (
                    region_counts[region],
                    transition_counts[trans],
                    emoji_counts[emoji],
                    len([x for x in selected + chosen if x["disagreement_stratum"] == stratum and x["top1_latent_region"] == region]),
                    random_rank[id(item)],
                    str(item["dialogue_id"]),
                    int(item["target_turn_index"]),
                )

            item = min(available, key=score)
            chosen.append(item)
            used_dialogues.add(str(item["dialogue_id"]))
            region_counts[str(item["top1_latent_region"])] += 1
            transition_counts[str(item["role_transition"])] += 1
            emoji_counts[str(item["top1_llm_emoji"])] += 1
        selected.extend(chosen)

    for idx, item in enumerate(selected, 1):
        item["sample_id"] = f"hla_{idx:04d}"
    return selected


def public_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": item["sample_id"],
        "situation": item["situation"],
        "context_turns": item["context_turns"],
        "target_turn": item["target_turn"],
        "target_turn_index": item["target_turn_index"],
        "target_role": item["target_role"],
    }


def private_item(item: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_id",
        "dialogue_id",
        "target_turn_index",
        "target_role",
        "previous_role",
        "role_transition",
        "situation",
        "context_turns",
        "target_turn",
        "llm_annotations",
        "llm_qE",
        "raw_entropy",
        "normalized_entropy",
        "top1_llm_emoji",
        "qZ",
        "top1_latent_region",
        "disagreement_stratum",
    ]
    return {key: item[key] for key in keys}


def manifest_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": item["sample_id"],
        "dialogue_id": item["dialogue_id"],
        "target_turn_index": item["target_turn_index"],
        "target_role": item["target_role"],
        "previous_role": item["previous_role"],
        "role_transition": item["role_transition"],
        "disagreement_stratum": item["disagreement_stratum"],
        "normalized_entropy": f"{float(item['normalized_entropy']):.12f}",
        "top1_llm_emoji": item["top1_llm_emoji"],
        "top1_latent_region": item["top1_latent_region"],
    }


def summarize_sampling(
    candidates: Sequence[dict[str, Any]],
    by_stratum: dict[str, list[dict[str, Any]]],
    selected: Sequence[dict[str, Any]],
    seed: int,
    hashes: dict[str, str],
    inventory_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    entropies = [float(item["normalized_entropy"]) for item in selected]
    dialogue_counts = Counter(item["dialogue_id"] for item in selected)
    duplicate_dialogues = sum(1 for count in dialogue_counts.values() if count > 1)
    membership_sources = Counter(row.get("membership_source", "unknown") for row in inventory_rows)
    return {
        "candidate_count": len(candidates),
        "stratum_candidate_counts": {stratum: len(by_stratum[stratum]) for stratum in STRATA},
        "stratum_sample_counts": dict(Counter(item["disagreement_stratum"] for item in selected)),
        "entropy": {
            "min": min(entropies),
            "mean": statistics.fmean(entropies),
            "median": statistics.median(entropies),
            "max": max(entropies),
        },
        "latent_region_counts": dict(sorted(Counter(item["top1_latent_region"] for item in selected).items())),
        "role_transition_counts": dict(sorted(Counter(item["role_transition"] for item in selected).items())),
        "top1_emoji_frequency": dict(sorted(Counter(item["top1_llm_emoji"] for item in selected).items())),
        "duplicate_dialogue_count": duplicate_dialogues,
        "sampling_seed": seed,
        "input_artifact_hashes": hashes,
        "inventory_count": len(inventory_rows),
        "membership_source_counts": dict(sorted(membership_sources.items())),
        "unobserved_uniform_inventory_emojis": [
            row["emoji"] for row in inventory_rows if row.get("membership_source") == "unobserved_uniform"
        ],
    }


def build_experiment_payload(
    project_root: Path,
    split: str,
    n_items: int,
    n_annotators: int,
    sampling_seed: int,
    models: Sequence[str] = DEFAULT_MODELS,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    inventory_path = project_root / "data/face_emojis_0.json"
    membership_path = project_root / "src/name_free_emoji_clustering/outputs/soft_membership/emoji_cluster_membership.csv"
    split_path = project_root / "pre_data" / f"{split}.json"
    inventory = load_inventory(inventory_path)
    raw_membership, clusters = load_membership_rows(membership_path)
    inventory_with_a = make_inventory_with_membership(inventory, raw_membership, clusters)
    matrix = membership_matrix(inventory_with_a, clusters)
    candidates = build_candidate_pool(project_root, split, inventory_with_a, matrix, clusters, models)
    by_stratum = assign_tertiles(candidates)
    selected = select_stratified_samples(by_stratum, n_items, sampling_seed)
    annotator_seeds = list(DEFAULT_ANNOTATOR_SEEDS[:n_annotators])
    if len(annotator_seeds) != n_annotators:
        raise ExperimentError(f"Only {len(DEFAULT_ANNOTATOR_SEEDS)} annotator seeds are defined")
    hashes = {
        str(split_path.relative_to(project_root)): sha256_file(split_path),
        str(inventory_path.relative_to(project_root)): sha256_file(inventory_path),
        str(membership_path.relative_to(project_root)): sha256_file(membership_path),
    }
    for model in models:
        path = project_root / "data" / model / f"{split}_emoji_annotations.json"
        hashes[str(path.relative_to(project_root))] = sha256_file(path)
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "project_root": str(project_root),
        "split": split,
        "n_items": n_items,
        "n_annotators": n_annotators,
        "sampling_seed": sampling_seed,
        "annotator_seeds": annotator_seeds,
        "models": list(models),
        "clusters": list(clusters),
        "artifact_paths": {
            "test_split": str(split_path.relative_to(project_root)),
            "candidate_inventory": str(inventory_path.relative_to(project_root)),
            "membership_matrix_A": str(membership_path.relative_to(project_root)),
            "llm_annotations": [f"data/{model}/{split}_emoji_annotations.json" for model in models],
        },
        "input_artifact_hashes": hashes,
        "emoji_inventory": inventory_with_a,
        "public_items": [public_item(item) for item in selected],
        "private_items": [private_item(item) for item in selected],
        "manifest_rows": [manifest_row(item) for item in selected],
        "sampling_report": summarize_sampling(candidates, by_stratum, selected, sampling_seed, hashes, inventory_with_a),
    }


HTML_STYLE = """
:root {
  --bg: #f5f6f8;
  --panel: #ffffff;
  --ink: #1f2933;
  --muted: #667085;
  --line: #d8dee8;
  --soft: #eef2f6;
  --accent: #176b7a;
  --accent-soft: #e5f3f5;
  --done: #1f7a4d;
  --warn: #9a3412;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 Arial, sans-serif; }
.app { max-width: 1160px; margin: 0 auto; padding: 20px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; margin-bottom: 12px; }
.top { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
h1 { font-size: 22px; margin: 0 0 8px; }
h2 { font-size: 16px; margin: 0 0 8px; }
.small { font-size: 13px; }
.muted { color: var(--muted); }
.pill { border: 1px solid var(--line); border-radius: 999px; padding: 4px 10px; background: #fff; white-space: nowrap; }
.progressbar { height: 10px; background: #e5eaf0; border-radius: 999px; overflow: hidden; margin-top: 8px; }
.progressbar > div { height: 100%; width: 0; background: var(--accent); transition: width .15s ease; }
.question { font-size: 16px; font-weight: 700; }
.help { color: var(--muted); margin-top: 6px; }
.situation { background: #f9fafb; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
.turn { display: grid; grid-template-columns: 54px 1fr; gap: 10px; padding: 10px 0; border-bottom: 1px solid #edf0f4; }
.turn:last-child { border-bottom: 0; }
.role { font-weight: 700; color: #344054; }
.target { border: 2px solid var(--accent); background: var(--accent-soft); border-radius: 8px; padding: 13px; }
.target .role { color: var(--accent); }
.emoji-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(44px, 1fr)); gap: 6px; background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 10px; max-height: 330px; overflow: auto; }
.emoji-btn { min-height: 42px; border: 1px solid #d0d7e2; border-radius: 8px; background: #fff; font-size: 22px; cursor: pointer; }
.emoji-btn:hover { background: #f1f6f8; }
.emoji-btn.selected { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(23,107,122,.18); background: #e9f7f9; }
.confidence { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.confidence label { border: 1px solid #c7d1dc; border-radius: 8px; padding: 8px 12px; cursor: pointer; background: #fff; min-width: 42px; text-align: center; }
.confidence input { margin-right: 6px; }
.statusline { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.complete { color: var(--done); font-weight: 700; }
.incomplete { color: var(--warn); font-weight: 700; }
button, input[type="number"] { border: 1px solid #c5cfda; border-radius: 8px; background: #fff; color: var(--ink); font: inherit; padding: 8px 11px; }
button { cursor: pointer; }
button:hover { background: #eef4f7; }
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
button.danger { color: var(--warn); border-color: #f1b487; }
.nav { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; justify-content: space-between; }
.nav-group { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.item-map { display: grid; grid-template-columns: repeat(auto-fill, minmax(34px, 1fr)); gap: 5px; margin-top: 10px; }
.map-btn { border-radius: 6px; padding: 5px 0; font-size: 12px; }
.map-btn.done { background: #e7f6ee; border-color: #98d4b4; }
.map-btn.current { outline: 2px solid var(--accent); }
.selected-info { min-height: 22px; }
.hidden { display: none; }
@media (max-width: 760px) {
  .app { padding: 12px; }
  .turn { grid-template-columns: 42px 1fr; }
  .emoji-grid { grid-template-columns: repeat(auto-fill, minmax(40px, 1fr)); }
}
"""


def json_script(tag_id: str, payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    text = text.replace("</", "<\\/")
    return f'<script id="{html.escape(tag_id)}" type="application/json">{text}</script>'


HTML_SCRIPT = r"""
const ITEMS = JSON.parse(document.getElementById("items-data").textContent);
const EMOJIS = JSON.parse(document.getElementById("emoji-data").textContent);
const META = JSON.parse(document.getElementById("meta-data").textContent);
const STORAGE_KEY = [META.schema_version, META.experiment_id, META.annotator_id].join("::");
const state = loadState();
let current = 0;
let enteredAt = Date.now();

function byId(id) { return document.getElementById(id); }
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function codepoint(emoji) {
  return Array.from(emoji).map(ch => "U+" + ch.codePointAt(0).toString(16).toUpperCase().padStart(4, "0")).join(" ");
}
function freshState() {
  return {
    schema_version: META.schema_version,
    experiment_id: META.experiment_id,
    annotator_id: META.annotator_id,
    questionnaire_id: META.questionnaire_id,
    sample_count: ITEMS.length,
    created_at: new Date().toISOString(),
    answers: {}
  };
}
function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return freshState();
  try {
    const parsed = JSON.parse(raw);
    if (parsed.schema_version !== META.schema_version || parsed.experiment_id !== META.experiment_id || parsed.annotator_id !== META.annotator_id) {
      return freshState();
    }
    parsed.answers = parsed.answers || {};
    return parsed;
  } catch (_) {
    return freshState();
  }
}
function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  renderProgress();
}
function flushElapsed() {
  const item = ITEMS[current];
  if (!item) return;
  const sid = item.sample_id;
  const answer = state.answers[sid];
  if (answer) {
    answer.elapsed_ms = Math.max(0, Number(answer.elapsed_ms || 0)) + Math.max(0, Date.now() - enteredAt);
    enteredAt = Date.now();
    saveState();
  }
}
function isCompleteAnswer(answer) {
  return !!answer && !!answer.selected_emoji && Number.isInteger(answer.confidence) && answer.confidence >= 1 && answer.confidence <= 5;
}
function completedCount() {
  return ITEMS.filter(item => isCompleteAnswer(state.answers[item.sample_id])).length;
}
function currentAnswer() {
  const sid = ITEMS[current].sample_id;
  if (!state.answers[sid]) {
    state.answers[sid] = {
      sample_id: sid,
      selected_emoji: "",
      unicode: "",
      confidence: null,
      item_order: current + 1,
      elapsed_ms: 0
    };
  }
  return state.answers[sid];
}
function setAnswerEmoji(emoji) {
  const answer = currentAnswer();
  answer.selected_emoji = emoji;
  answer.unicode = codepoint(emoji);
  answer.item_order = current + 1;
  saveState();
  renderItem();
}
function setConfidence(value) {
  const answer = currentAnswer();
  answer.confidence = Number(value);
  answer.item_order = current + 1;
  saveState();
  renderItem();
}
function renderProgress() {
  const done = completedCount();
  byId("progress-text").textContent = `${current + 1} / ${ITEMS.length}`;
  byId("complete-text").textContent = done === ITEMS.length ? "Complete" : `${done} completed`;
  byId("complete-text").className = done === ITEMS.length ? "complete" : "incomplete";
  byId("progress-fill").style.width = `${100 * done / ITEMS.length}%`;
  byId("remaining-text").textContent = `${ITEMS.length - done} remaining`;
  renderMap();
}
function renderMap() {
  const map = byId("item-map");
  map.innerHTML = "";
  ITEMS.forEach((item, idx) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "map-btn" + (isCompleteAnswer(state.answers[item.sample_id]) ? " done" : "") + (idx === current ? " current" : "");
    button.textContent = String(idx + 1);
    button.title = isCompleteAnswer(state.answers[item.sample_id]) ? "Completed" : "Incomplete";
    button.addEventListener("click", () => goTo(idx));
    map.appendChild(button);
  });
}
function renderItem() {
  const item = ITEMS[current];
  const answer = state.answers[item.sample_id] || {};
  byId("sample-id").textContent = item.sample_id;
  byId("situation").textContent = item.situation;
  const context = byId("context");
  context.innerHTML = "";
  item.context_turns.forEach(turn => {
    const div = document.createElement("div");
    div.className = "turn";
    div.innerHTML = `<div class="role">${escapeHtml(turn.role)}</div><div>${escapeHtml(turn.utterance)}</div>`;
    context.appendChild(div);
  });
  byId("target").innerHTML = `<div class="role">${escapeHtml(item.target_role)} · turn ${escapeHtml(item.target_turn_index)}</div><div>${escapeHtml(item.target_turn)}</div>`;
  const grid = byId("emoji-grid");
  grid.innerHTML = "";
  EMOJIS.forEach(row => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "emoji-btn" + (answer.selected_emoji === row.emoji ? " selected" : "");
    button.textContent = row.emoji;
    button.title = row.unicode || codepoint(row.emoji);
    button.addEventListener("click", () => setAnswerEmoji(row.emoji));
    grid.appendChild(button);
  });
  document.querySelectorAll("input[name='confidence']").forEach(input => {
    input.checked = Number(input.value) === Number(answer.confidence);
  });
  byId("selected-info").textContent = answer.selected_emoji ? `Selected: ${answer.selected_emoji} (${answer.unicode || codepoint(answer.selected_emoji)})` : "No emoji selected";
  byId("item-status").textContent = isCompleteAnswer(answer) ? "This item is complete." : "This item is incomplete.";
  byId("item-status").className = isCompleteAnswer(answer) ? "complete" : "incomplete";
  byId("jump-input").value = String(current + 1);
  renderProgress();
}
function goTo(idx) {
  if (idx < 0 || idx >= ITEMS.length) return;
  flushElapsed();
  current = idx;
  enteredAt = Date.now();
  renderItem();
}
function firstIncomplete() {
  const idx = ITEMS.findIndex(item => !isCompleteAnswer(state.answers[item.sample_id]));
  if (idx >= 0) goTo(idx);
}
function nextItem() {
  const answer = state.answers[ITEMS[current].sample_id];
  if (!isCompleteAnswer(answer)) {
    window.alert("This item is not complete yet. You can skip it now, but it must have one emoji and one confidence score before final export validation will pass.");
  }
  goTo(Math.min(ITEMS.length - 1, current + 1));
}
function validateExportPayload(payload) {
  const errors = [];
  if (payload.schema_version !== META.schema_version) errors.push("schema_version mismatch");
  if (payload.experiment_id !== META.experiment_id) errors.push("experiment_id mismatch");
  if (payload.annotator_id !== META.annotator_id) errors.push("annotator_id mismatch");
  if (payload.sample_count !== ITEMS.length) errors.push("sample_count mismatch");
  const allowedSamples = new Set(ITEMS.map(item => item.sample_id));
  const allowedEmoji = new Set(EMOJIS.map(row => row.emoji));
  const seen = new Set();
  payload.answers.forEach((answer, idx) => {
    if (!allowedSamples.has(answer.sample_id)) errors.push(`answer ${idx + 1} has unknown sample_id`);
    if (seen.has(answer.sample_id)) errors.push(`duplicate sample_id ${answer.sample_id}`);
    seen.add(answer.sample_id);
    if (!allowedEmoji.has(answer.selected_emoji)) errors.push(`answer ${idx + 1} has emoji outside inventory`);
    if (!Number.isInteger(answer.confidence) || answer.confidence < 1 || answer.confidence > 5) errors.push(`answer ${idx + 1} has invalid confidence`);
  });
  if (payload.answers.length !== ITEMS.length) errors.push("answers length is not 120");
  return errors;
}
function makeExportPayload() {
  flushElapsed();
  const answers = ITEMS.map((item, idx) => {
    const answer = state.answers[item.sample_id] || {};
    return {
      sample_id: item.sample_id,
      selected_emoji: answer.selected_emoji || "",
      unicode: answer.unicode || (answer.selected_emoji ? codepoint(answer.selected_emoji) : ""),
      confidence: Number.isInteger(answer.confidence) ? answer.confidence : null,
      item_order: idx + 1,
      elapsed_ms: Math.max(0, Math.round(Number(answer.elapsed_ms || 0)))
    };
  });
  return {
    schema_version: META.schema_version,
    experiment_id: META.experiment_id,
    annotator_id: META.annotator_id,
    questionnaire_id: META.questionnaire_id,
    sample_count: ITEMS.length,
    completed_count: completedCount(),
    created_at: state.created_at,
    exported_at: new Date().toISOString(),
    answers
  };
}
function exportJson() {
  const payload = makeExportPayload();
  const remaining = ITEMS.length - payload.completed_count;
  if (remaining > 0 && !window.confirm(`${remaining} item(s) are incomplete. Export anyway?`)) return;
  const errors = validateExportPayload(payload);
  if (errors.length && !window.confirm(`Schema validation found ${errors.length} issue(s). Export anyway?\n\n${errors.slice(0, 6).join("\n")}`)) return;
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: "application/json;charset=utf-8"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${META.annotator_id}_annotations.json`;
  document.body.appendChild(link);
  link.click();
  URL.revokeObjectURL(link.href);
  link.remove();
}
function importJson(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const payload = JSON.parse(reader.result);
      const errors = validateExportPayload(payload);
      if (errors.length) {
        window.alert(`Import failed:\n${errors.slice(0, 10).join("\n")}`);
        return;
      }
      state.created_at = payload.created_at || state.created_at;
      state.answers = {};
      payload.answers.forEach((answer, idx) => {
        state.answers[answer.sample_id] = {
          sample_id: answer.sample_id,
          selected_emoji: answer.selected_emoji,
          unicode: answer.unicode || codepoint(answer.selected_emoji),
          confidence: answer.confidence,
          item_order: answer.item_order || idx + 1,
          elapsed_ms: Math.max(0, Number(answer.elapsed_ms || 0))
        };
      });
      saveState();
      renderItem();
      window.alert("Import complete.");
    } catch (err) {
      window.alert(`Import failed: ${err}`);
    }
  };
  reader.readAsText(file, "utf-8");
}
function resetAll() {
  if (!window.confirm("Clear all saved answers for this annotator?")) return;
  if (!window.confirm("This cannot be undone. Clear answers now?")) return;
  localStorage.removeItem(STORAGE_KEY);
  Object.assign(state, freshState());
  current = 0;
  enteredAt = Date.now();
  renderItem();
}
document.addEventListener("DOMContentLoaded", () => {
  byId("annotator-id").textContent = META.annotator_id;
  byId("questionnaire-id").textContent = META.questionnaire_id;
  byId("experiment-id").textContent = META.experiment_id;
  byId("question").textContent = META.question_text;
  byId("help").textContent = META.help_text;
  byId("prev").addEventListener("click", () => goTo(Math.max(0, current - 1)));
  byId("next").addEventListener("click", nextItem);
  byId("first-incomplete").addEventListener("click", firstIncomplete);
  byId("jump").addEventListener("click", () => goTo(Number(byId("jump-input").value) - 1));
  byId("export-json").addEventListener("click", exportJson);
  byId("import-json").addEventListener("click", () => byId("import-file").click());
  byId("import-file").addEventListener("change", event => {
    const file = event.target.files && event.target.files[0];
    if (file) importJson(file);
    event.target.value = "";
  });
  byId("reset").addEventListener("click", resetAll);
  document.querySelectorAll("input[name='confidence']").forEach(input => {
    input.addEventListener("change", () => setConfidence(input.value));
  });
  renderItem();
});
"""


def make_questionnaire(
    public_items: Sequence[dict[str, Any]],
    emoji_inventory: Sequence[dict[str, Any]],
    annotator_index: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    rng = random.Random(seed)
    items = [dict(item) for item in public_items]
    emojis = [{"emoji": row["emoji"], "unicode": row.get("unicode") or codepoint_string(row["emoji"])} for row in emoji_inventory]
    rng.shuffle(items)
    rng.shuffle(emojis)
    annotator_id = f"annotator_{annotator_index}"
    questionnaire_id = f"questionnaire_{annotator_index}"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "annotator_id": annotator_id,
        "questionnaire_id": questionnaire_id,
        "question_text": QUESTION_TEXT,
        "help_text": HELP_TEXT,
        "item_order": [item["sample_id"] for item in items],
        "emoji_order": [row["emoji"] for row in emojis],
    }
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM-Human Emoji Distribution Audit · {html.escape(annotator_id)}</title>
  <style>{HTML_STYLE}</style>
</head>
<body>
  <div class="app">
    <section class="panel top">
      <div>
        <h1>Emoji Annotation Audit</h1>
        <div class="statusline small">
          <span class="pill">Annotator: <strong id="annotator-id"></strong></span>
          <span class="pill">Questionnaire: <strong id="questionnaire-id"></strong></span>
          <span class="pill">Progress: <strong id="progress-text"></strong></span>
          <span class="pill"><strong id="complete-text"></strong></span>
          <span class="pill" id="remaining-text"></span>
        </div>
        <div class="progressbar"><div id="progress-fill"></div></div>
      </div>
      <div class="small muted">Anonymous ID only. Experiment: <span id="experiment-id"></span></div>
    </section>
    <section class="panel">
      <div class="question" id="question"></div>
      <div class="help" id="help"></div>
    </section>
    <section class="panel">
      <h2>Sample <span id="sample-id"></span></h2>
      <h2>Situation</h2>
      <div class="situation" id="situation"></div>
      <h2>Dialogue context</h2>
      <div id="context"></div>
      <h2>Target utterance</h2>
      <div class="target" id="target"></div>
    </section>
    <section class="panel">
      <h2>Emoji picker</h2>
      <div class="emoji-grid" id="emoji-grid"></div>
      <div class="selected-info small muted" id="selected-info"></div>
    </section>
    <section class="panel">
      <h2>Confidence</h2>
      <div class="confidence">
        <label><input type="radio" name="confidence" value="1">1</label>
        <label><input type="radio" name="confidence" value="2">2</label>
        <label><input type="radio" name="confidence" value="3">3</label>
        <label><input type="radio" name="confidence" value="4">4</label>
        <label><input type="radio" name="confidence" value="5">5</label>
        <span id="item-status" class="incomplete"></span>
      </div>
    </section>
    <section class="panel">
      <div class="nav">
        <div class="nav-group">
          <button type="button" id="prev">Previous</button>
          <button type="button" id="next" class="primary">Next</button>
          <button type="button" id="first-incomplete">First incomplete</button>
        </div>
        <div class="nav-group">
          <input id="jump-input" type="number" min="1" max="120" value="1" aria-label="Item number">
          <button type="button" id="jump">Jump</button>
        </div>
        <div class="nav-group">
          <button type="button" id="export-json">Export JSON</button>
          <button type="button" id="import-json">Import JSON</button>
          <button type="button" class="danger" id="reset">Clear/reset</button>
          <input id="import-file" type="file" accept="application/json,.json" class="hidden">
        </div>
      </div>
      <div class="item-map" id="item-map"></div>
    </section>
  </div>
  {json_script("items-data", items)}
  {json_script("emoji-data", emojis)}
  {json_script("meta-data", metadata)}
  <script>{HTML_SCRIPT}</script>
</body>
</html>
"""
    return html_text, metadata


def write_experiment(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("config", "data", "html", "exports", "scripts", "reports", "tests"):
        (output_dir / subdir).mkdir(exist_ok=True)
    written: list[Path] = []
    config = {
        key: payload[key]
        for key in (
            "experiment_id",
            "schema_version",
            "created_at",
            "project_root",
            "split",
            "n_items",
            "n_annotators",
            "sampling_seed",
            "annotator_seeds",
            "models",
            "clusters",
            "artifact_paths",
            "input_artifact_hashes",
        )
    }
    write_json(output_dir / "config/experiment.json", config)
    written.append(output_dir / "config/experiment.json")
    write_json(output_dir / "data/sampled_items_public.json", payload["public_items"])
    written.append(output_dir / "data/sampled_items_public.json")
    write_json(output_dir / "data/sampled_items_private.json", payload["private_items"])
    written.append(output_dir / "data/sampled_items_private.json")
    write_json(output_dir / "data/emoji_inventory.json", payload["emoji_inventory"])
    written.append(output_dir / "data/emoji_inventory.json")
    write_csv(
        output_dir / "data/sample_manifest.csv",
        payload["manifest_rows"],
        [
            "sample_id",
            "dialogue_id",
            "target_turn_index",
            "target_role",
            "previous_role",
            "role_transition",
            "disagreement_stratum",
            "normalized_entropy",
            "top1_llm_emoji",
            "top1_latent_region",
        ],
    )
    written.append(output_dir / "data/sample_manifest.csv")
    write_json(output_dir / "data/sampling_report.json", payload["sampling_report"])
    written.append(output_dir / "data/sampling_report.json")

    questionnaire_metadata = []
    for annotator_idx, seed in enumerate(payload["annotator_seeds"], 1):
        html_text, meta = make_questionnaire(payload["public_items"], payload["emoji_inventory"], annotator_idx, seed)
        html_path = output_dir / "html" / f"annotator_{annotator_idx}.html"
        html_path.write_text(html_text, encoding="utf-8")
        written.append(html_path)
        questionnaire_metadata.append(meta)
    write_json(output_dir / "data/questionnaire_metadata.json", questionnaire_metadata)
    written.append(output_dir / "data/questionnaire_metadata.json")
    (output_dir / "exports/.gitkeep").touch()
    written.append(output_dir / "exports/.gitkeep")
    return written


def load_experiment_dir(experiment_dir: Path) -> dict[str, Any]:
    return {
        "config": read_json(experiment_dir / "config/experiment.json"),
        "public_items": read_json(experiment_dir / "data/sampled_items_public.json"),
        "private_items": read_json(experiment_dir / "data/sampled_items_private.json"),
        "emoji_inventory": read_json(experiment_dir / "data/emoji_inventory.json"),
        "sampling_report": read_json(experiment_dir / "data/sampling_report.json"),
        "questionnaire_metadata": read_json(experiment_dir / "data/questionnaire_metadata.json"),
    }


def public_payload_leaks(public_items: Sequence[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for idx, item in enumerate(public_items):
        keys = set(item.keys())
        extra = keys - PUBLIC_ITEM_KEYS
        missing = PUBLIC_ITEM_KEYS - keys
        if extra:
            errors.append(f"public item {idx} has extra keys: {sorted(extra)}")
        if missing:
            errors.append(f"public item {idx} has missing keys: {sorted(missing)}")
        text = json.dumps(item, ensure_ascii=False)
        for token in PRIVATE_FIELD_TOKENS:
            if token in text:
                errors.append(f"public item {idx} contains private token {token!r}")
    return errors


def extract_json_script(html_text: str, tag_id: str) -> Any:
    pattern = re.compile(
        rf'<script\s+id="{re.escape(tag_id)}"\s+type="application/json">(.*?)</script>',
        re.DOTALL,
    )
    match = pattern.search(html_text)
    if not match:
        raise ExperimentError(f"Missing JSON script tag {tag_id}")
    return json.loads(match.group(1).replace("<\\/", "</"))


def html_private_leaks(html_text: str) -> list[str]:
    errors: list[str] = []
    for token in PRIVATE_FIELD_TOKENS:
        if token in html_text:
            errors.append(f"HTML contains private token {token!r}")
    external_patterns = [
        r"<script[^>]+src=['\"]https?://",
        r"<link[^>]+href=['\"]https?://",
        r"<img[^>]+src=['\"]https?://",
        r"cdn\.",
    ]
    for pattern in external_patterns:
        if re.search(pattern, html_text, flags=re.IGNORECASE):
            errors.append(f"HTML appears to reference an external resource: {pattern}")
    return errors


def validate_experiment_artifacts(experiment_dir: Path, *, check_reproducible: bool = True) -> tuple[bool, list[str], dict[str, Any]]:
    data = load_experiment_dir(experiment_dir)
    config = data["config"]
    public_items = data["public_items"]
    private_items = data["private_items"]
    inventory = data["emoji_inventory"]
    report = data["sampling_report"]
    metadata = data["questionnaire_metadata"]
    errors: list[str] = []

    if len(public_items) != 120 or len(private_items) != 120:
        errors.append(f"sampled items must be 120 public/private, got {len(public_items)}/{len(private_items)}")
    strata = Counter(item.get("disagreement_stratum") for item in private_items)
    for stratum in STRATA:
        if strata[stratum] != 40:
            errors.append(f"{stratum} stratum must have 40 samples, got {strata[stratum]}")
    dialogue_ids = [item.get("dialogue_id") for item in private_items]
    if len(set(dialogue_ids)) != 120:
        errors.append(f"dialogue_id must be 120 unique values, got {len(set(dialogue_ids))}")
    for item in private_items:
        if int(item.get("target_turn_index", -1)) < 1:
            errors.append(f"target_turn_index < 1 for {item.get('sample_id')}")
    if len(inventory) != 136:
        errors.append(f"emoji inventory must contain 136 entries, got {len(inventory)}")
    if not report.get("input_artifact_hashes"):
        errors.append("input artifact hashes are missing")
    if int(report.get("duplicate_dialogue_count", -1)) != 0:
        errors.append(f"duplicate dialogue count must be 0, got {report.get('duplicate_dialogue_count')}")
    errors.extend(public_payload_leaks(public_items))

    expected_ids = {item["sample_id"] for item in public_items}
    item_orders: list[tuple[str, ...]] = []
    emoji_orders: list[tuple[str, ...]] = []
    for idx in range(1, 4):
        html_path = experiment_dir / "html" / f"annotator_{idx}.html"
        if not html_path.exists():
            errors.append(f"missing HTML: {html_path}")
            continue
        html_text = html_path.read_text(encoding="utf-8")
        errors.extend(f"annotator_{idx}: {err}" for err in html_private_leaks(html_text))
        try:
            items = extract_json_script(html_text, "items-data")
            emojis = extract_json_script(html_text, "emoji-data")
            meta = extract_json_script(html_text, "meta-data")
        except Exception as exc:
            errors.append(f"annotator_{idx}: cannot parse embedded JSON: {exc}")
            continue
        ids = {item.get("sample_id") for item in items}
        if ids != expected_ids:
            errors.append(f"annotator_{idx}: embedded sample_id set differs from public payload")
        item_orders.append(tuple(item.get("sample_id") for item in items))
        emoji_orders.append(tuple(row.get("emoji") for row in emojis))
        if set(meta.get("item_order", [])) != expected_ids:
            errors.append(f"annotator_{idx}: metadata item_order set differs from public payload")
        if len(emojis) != 136:
            errors.append(f"annotator_{idx}: embedded emoji count must be 136, got {len(emojis)}")
        for row in emojis:
            if set(row.keys()) - {"emoji", "unicode"}:
                errors.append(f"annotator_{idx}: emoji payload contains non-public keys")
                break
    if len(set(item_orders)) != len(item_orders):
        errors.append("three item orders are not all distinct")
    if len(set(emoji_orders)) != len(emoji_orders):
        errors.append("three emoji orders are not all distinct")
    if len(metadata) != 3:
        errors.append(f"questionnaire metadata must contain 3 records, got {len(metadata)}")

    repro_summary: dict[str, Any] = {"checked": False}
    if check_reproducible:
        project_root = Path(config["project_root"])
        rebuilt = build_experiment_payload(
            project_root=project_root,
            split=config["split"],
            n_items=int(config["n_items"]),
            n_annotators=int(config["n_annotators"]),
            sampling_seed=int(config["sampling_seed"]),
            models=tuple(config["models"]),
        )
        old_ids = [item["sample_id"] + "|" + item["dialogue_id"] + "|" + str(item["target_turn_index"]) for item in private_items]
        new_ids = [
            item["sample_id"] + "|" + item["dialogue_id"] + "|" + str(item["target_turn_index"])
            for item in rebuilt["private_items"]
        ]
        repro_summary = {"checked": True, "matches": old_ids == new_ids}
        if old_ids != new_ids:
            errors.append("sampling is not reproducible from config seed and input artifacts")

    summary = {
        "valid": not errors,
        "sample_count": len(public_items),
        "stratum_counts": dict(strata),
        "unique_dialogues": len(set(dialogue_ids)),
        "emoji_count": len(inventory),
        "questionnaire_count": len(item_orders),
        "duplicate_dialogue_count": report.get("duplicate_dialogue_count"),
        "reproducibility": repro_summary,
        "private_field_leakage_errors": [err for err in errors if "private token" in err or "public item" in err],
        "errors": errors,
    }
    return not errors, errors, summary


def validate_export_payload(
    payload: dict[str, Any],
    *,
    expected_annotator_id: str | None,
    expected_sample_ids: set[str],
    emoji_set: set[str],
    schema_version: str = SCHEMA_VERSION,
    experiment_id: str = EXPERIMENT_ID,
) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != schema_version:
        errors.append("schema_version mismatch")
    if payload.get("experiment_id") != experiment_id:
        errors.append("experiment_id mismatch")
    annotator_id = payload.get("annotator_id")
    if expected_annotator_id and annotator_id != expected_annotator_id:
        errors.append(f"annotator_id must be {expected_annotator_id}, got {annotator_id}")
    if "name" in payload or "email" in payload:
        errors.append("export must not contain name/email fields")
    answers = payload.get("answers")
    if not isinstance(answers, list):
        return errors + ["answers must be a list"]
    if len(answers) != len(expected_sample_ids):
        errors.append(f"answers must contain {len(expected_sample_ids)} rows, got {len(answers)}")
    seen: set[str] = set()
    for idx, answer in enumerate(answers):
        if not isinstance(answer, dict):
            errors.append(f"answer {idx + 1} is not an object")
            continue
        sample_id = answer.get("sample_id")
        if sample_id not in expected_sample_ids:
            errors.append(f"answer {idx + 1} has unknown sample_id {sample_id!r}")
        if sample_id in seen:
            errors.append(f"duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        emoji = answer.get("selected_emoji")
        if emoji not in emoji_set:
            errors.append(f"answer {idx + 1} selected_emoji outside inventory: {emoji!r}")
        confidence = answer.get("confidence")
        if not isinstance(confidence, int) or confidence < 1 or confidence > 5:
            errors.append(f"answer {idx + 1} confidence must be integer 1..5")
        if not answer.get("unicode"):
            errors.append(f"answer {idx + 1} lacks unicode")
    missing = expected_sample_ids - seen
    if missing:
        errors.append(f"missing sample_id values: {sorted(missing)[:5]}")
    if payload.get("sample_count") != len(expected_sample_ids):
        errors.append("sample_count mismatch")
    if payload.get("completed_count") != len(expected_sample_ids):
        errors.append("completed_count must equal sample_count for final validation")
    return errors


def load_export_files(exports_dir: Path) -> list[Path]:
    return sorted(path for path in exports_dir.glob("*.json") if path.is_file())


def validate_exports(experiment_dir: Path, exports_dir: Path) -> tuple[bool, dict[str, Any], str]:
    data = load_experiment_dir(experiment_dir)
    sample_ids = {item["sample_id"] for item in data["public_items"]}
    emoji_set = {row["emoji"] for row in data["emoji_inventory"]}
    files = load_export_files(exports_dir)
    errors: list[str] = []
    expected_ids = {f"annotator_{idx}" for idx in range(1, 4)}
    payloads: dict[str, dict[str, Any]] = {}
    if len(files) != 3:
        errors.append(f"expected exactly 3 export JSON files, found {len(files)}")
    for path in files:
        try:
            payload = read_json(path)
        except Exception as exc:
            errors.append(f"{path.name}: cannot read JSON: {exc}")
            continue
        annotator_id = payload.get("annotator_id")
        if annotator_id in payloads:
            errors.append(f"duplicate annotator_id {annotator_id}")
        if annotator_id not in expected_ids:
            errors.append(f"{path.name}: unexpected annotator_id {annotator_id!r}")
        payloads[str(annotator_id)] = payload
        errors.extend(f"{path.name}: {err}" for err in validate_export_payload(
            payload,
            expected_annotator_id=str(annotator_id) if annotator_id in expected_ids else None,
            expected_sample_ids=sample_ids,
            emoji_set=emoji_set,
            schema_version=data["config"]["schema_version"],
            experiment_id=data["config"]["experiment_id"],
        ))
    if set(payloads) != expected_ids:
        errors.append(f"annotator_id set must be {sorted(expected_ids)}, got {sorted(payloads)}")
    sample_sets = {aid: {ans.get("sample_id") for ans in payload.get("answers", [])} for aid, payload in payloads.items()}
    if sample_sets and any(values != sample_ids for values in sample_sets.values()):
        errors.append("three export sample_id sets are not exactly the experiment sample_id set")
    total_judgments = sum(len(payload.get("answers", [])) for payload in payloads.values())
    if total_judgments != 360:
        errors.append(f"total judgments must be 360, got {total_judgments}")
    result = {
        "valid": not errors,
        "export_files": [str(path) for path in files],
        "annotator_ids": sorted(payloads),
        "sample_count_per_export": {aid: len(payload.get("answers", [])) for aid, payload in payloads.items()},
        "total_judgments": total_judgments,
        "errors": errors,
    }
    markdown = ["# Export Validation", "", f"- Valid: `{str(not errors).lower()}`", f"- Export files: {len(files)}", f"- Total judgments: {total_judgments}"]
    if errors:
        markdown.extend(["", "## Errors"])
        markdown.extend(f"- {err}" for err in errors)
    return not errors, result, "\n".join(markdown) + "\n"


def jsd_base2(p: Sequence[float], q: Sequence[float]) -> float:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if p_arr.shape != q_arr.shape:
        raise ExperimentError("JSD distributions must have the same shape")
    p_sum = float(p_arr.sum())
    q_sum = float(q_arr.sum())
    if p_sum <= 0 or q_sum <= 0:
        raise ExperimentError("JSD distributions must have positive mass")
    p_arr = p_arr / p_sum
    q_arr = q_arr / q_sum
    m = 0.5 * (p_arr + q_arr)

    def kl_base2(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    value = 0.5 * kl_base2(p_arr, m) + 0.5 * kl_base2(q_arr, m)
    return min(1.0, max(0.0, value))


def distribution_overlap(p: Sequence[float], q: Sequence[float]) -> float:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if p_arr.shape != q_arr.shape:
        raise ExperimentError("Overlap distributions must have the same shape")
    p_arr = p_arr / max(float(p_arr.sum()), 1e-12)
    q_arr = q_arr / max(float(q_arr.sum()), 1e-12)
    return float(np.minimum(p_arr, q_arr).sum())


def tie_aware_agreement(p: Sequence[float], q: Sequence[float]) -> int:
    return int(bool(top_set(p) & top_set(q)))


def human_distribution(
    answers: Sequence[dict[str, Any]],
    emojis: Sequence[str],
    *,
    confidence_weighted: bool,
) -> dict[str, float]:
    if len(answers) != 3:
        raise ExperimentError(f"Expected 3 human answers, got {len(answers)}")
    weights = []
    for answer in answers:
        weight = int(answer["confidence"]) if confidence_weighted else 1
        weights.append((answer["selected_emoji"], float(weight)))
    if confidence_weighted:
        return q_from_weighted_votes(weights, emojis)
    counts: dict[str, float] = defaultdict(float)
    for emoji, _ in weights:
        counts[emoji] += 1.0 / len(weights)
    return dict(sorted(counts.items()))


def leave_one_out_metrics(answers: Sequence[dict[str, Any]], emojis: Sequence[str], matrix: np.ndarray) -> dict[str, float]:
    if len(answers) != 3:
        raise ExperimentError("leave-one-out requires exactly three answers")
    emoji_vectors = []
    for answer in answers:
        emoji_vectors.append(distribution_vector({answer["selected_emoji"]: 1.0}, emojis))
    emoji_jsd = []
    emoji_overlap = []
    region_jsd = []
    region_overlap = []
    for idx in range(3):
        single = emoji_vectors[idx]
        pair = sum(emoji_vectors[j] for j in range(3) if j != idx) / 2.0
        single_z = single @ matrix
        pair_z = pair @ matrix
        emoji_jsd.append(jsd_base2(single, pair))
        emoji_overlap.append(distribution_overlap(single, pair))
        region_jsd.append(jsd_base2(single_z, pair_z))
        region_overlap.append(distribution_overlap(single_z, pair_z))
    return {
        "human_loo_emoji_jsd": statistics.fmean(emoji_jsd),
        "human_loo_region_jsd": statistics.fmean(region_jsd),
        "human_loo_emoji_overlap": statistics.fmean(emoji_overlap),
        "human_loo_region_overlap": statistics.fmean(region_overlap),
    }


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    values = [float(v) for v in values]
    n = len(values)
    boot = []
    for _ in range(n_boot):
        boot.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    return {
        "mean": statistics.fmean(values),
        "ci_low": percentile(boot, 0.025),
        "ci_high": percentile(boot, 0.975),
    }


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def ranks(vals: Sequence[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        result = [0.0] * len(vals)
        start = 0
        while start < len(vals):
            end = start
            while end + 1 < len(vals) and vals[order[end + 1]] == vals[order[start]]:
                end += 1
            rank = (start + end + 2) / 2.0
            for idx in range(start, end + 1):
                result[order[idx]] = rank
            start = end + 1
        return result

    rx = ranks(xs)
    ry = ranks(ys)
    mx = statistics.fmean(rx)
    my = statistics.fmean(ry)
    num = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in rx))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ry))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def sample_level_metrics(
    experiment_dir: Path,
    exports_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    valid, validation, _ = validate_exports(experiment_dir, exports_dir)
    if not valid:
        raise ExperimentError("Export validation failed; run validate_exports.py for details")
    data = load_experiment_dir(experiment_dir)
    private_by_id = {item["sample_id"]: item for item in data["private_items"]}
    inventory = data["emoji_inventory"]
    emojis = [row["emoji"] for row in inventory]
    clusters = data["config"]["clusters"]
    matrix = membership_matrix(inventory, clusters)
    exports = {read_json(path)["annotator_id"]: read_json(path) for path in load_export_files(exports_dir)}
    answers_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotator_id in sorted(exports):
        for answer in exports[annotator_id]["answers"]:
            row = dict(answer)
            row["annotator_id"] = annotator_id
            answers_by_sample[row["sample_id"]].append(row)

    metrics: list[dict[str, Any]] = []
    robustness: list[dict[str, Any]] = []
    for sample_id, private in private_by_id.items():
        answers = sorted(answers_by_sample[sample_id], key=lambda x: x["annotator_id"])
        llm_e = distribution_vector(private["llm_qE"], emojis)
        llm_z = llm_e @ matrix
        human_unweighted = human_distribution(answers, emojis, confidence_weighted=False)
        human_conf = human_distribution(answers, emojis, confidence_weighted=True)
        human_e = distribution_vector(human_unweighted, emojis)
        human_conf_e = distribution_vector(human_conf, emojis)
        human_z = human_e @ matrix
        human_conf_z = human_conf_e @ matrix
        loo = leave_one_out_metrics(answers, emojis, matrix)
        row = {
            "sample_id": sample_id,
            "disagreement_stratum": private["disagreement_stratum"],
            "llm_normalized_entropy": float(private["normalized_entropy"]),
            "emoji_jsd": jsd_base2(llm_e, human_e),
            "region_jsd": jsd_base2(llm_z, human_z),
            "emoji_overlap": distribution_overlap(llm_e, human_e),
            "region_overlap": distribution_overlap(llm_z, human_z),
            "top1_emoji_agreement": tie_aware_agreement(llm_e, human_e),
            "top1_region_agreement": tie_aware_agreement(llm_z, human_z),
            **loo,
        }
        row["delta_projection"] = row["region_jsd"] - row["emoji_jsd"]
        metrics.append(row)
        robustness.append(
            {
                "sample_id": sample_id,
                "disagreement_stratum": private["disagreement_stratum"],
                "human_aggregation": "human unweighted",
                "emoji_jsd": row["emoji_jsd"],
                "region_jsd": row["region_jsd"],
                "emoji_overlap": row["emoji_overlap"],
                "region_overlap": row["region_overlap"],
            }
        )
        robustness.append(
            {
                "sample_id": sample_id,
                "disagreement_stratum": private["disagreement_stratum"],
                "human_aggregation": "human confidence-weighted",
                "emoji_jsd": jsd_base2(llm_e, human_conf_e),
                "region_jsd": jsd_base2(llm_z, human_conf_z),
                "emoji_overlap": distribution_overlap(llm_e, human_conf_e),
                "region_overlap": distribution_overlap(llm_z, human_conf_z),
            }
        )
    return metrics, robustness, validation


def aggregate_metric_rows(rows: Sequence[dict[str, Any]], *, n_boot: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for key in ("emoji_jsd", "region_jsd", "emoji_overlap", "region_overlap", "delta_projection"):
        if key in rows[0]:
            ci = bootstrap_mean_ci([float(row[key]) for row in rows], n_boot=n_boot, seed=seed)
            out[f"{key}_mean"] = ci["mean"]
            out[f"{key}_ci_low"] = ci["ci_low"]
            out[f"{key}_ci_high"] = ci["ci_high"]
    for key in ("top1_emoji_agreement", "top1_region_agreement"):
        if key in rows[0]:
            successes = sum(int(row[key]) for row in rows)
            low, high = wilson_ci(successes, len(rows))
            out[key] = successes / len(rows)
            out[f"{key}_ci_low"] = low
            out[f"{key}_ci_high"] = high
    return out


def write_analysis_outputs(
    experiment_dir: Path,
    exports_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    metrics, robustness, validation = sample_level_metrics(experiment_dir, exports_dir)
    reports = experiment_dir / "reports"
    reports.mkdir(exist_ok=True)
    overall = aggregate_metric_rows(metrics, n_boot=bootstrap_samples, seed=seed)
    by_stratum: list[dict[str, Any]] = []
    for stratum in STRATA:
        rows = [row for row in metrics if row["disagreement_stratum"] == stratum]
        agg = aggregate_metric_rows(rows, n_boot=bootstrap_samples, seed=seed)
        by_stratum.append(
            {
                "stratum": stratum,
                "n": len(rows),
                "emoji_jsd": agg["emoji_jsd_mean"],
                "region_jsd": agg["region_jsd_mean"],
                "region_overlap": agg["region_overlap_mean"],
                "top1_region_agreement": agg["top1_region_agreement"],
            }
        )
    robustness_rows = []
    for label in ("human unweighted", "human confidence-weighted"):
        rows = [row for row in robustness if row["human_aggregation"] == label]
        agg = aggregate_metric_rows(rows, n_boot=bootstrap_samples, seed=seed)
        robustness_rows.append({"human_aggregation": label, **agg})
    loo_agg = {
        key: bootstrap_mean_ci([float(row[key]) for row in metrics], n_boot=bootstrap_samples, seed=seed)
        for key in ("human_loo_emoji_jsd", "human_loo_region_jsd", "human_loo_emoji_overlap", "human_loo_region_overlap")
    }
    summary = {
        "validation": validation,
        "overall": overall,
        "by_disagreement": by_stratum,
        "confidence_robustness": robustness_rows,
        "human_leave_one_out": loo_agg,
        "spearman": {
            "llm_entropy_vs_emoji_jsd": spearman(
                [row["llm_normalized_entropy"] for row in metrics],
                [row["emoji_jsd"] for row in metrics],
            ),
            "llm_entropy_vs_region_jsd": spearman(
                [row["llm_normalized_entropy"] for row in metrics],
                [row["region_jsd"] for row in metrics],
            ),
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }
    write_json(reports / "analysis_summary.json", summary)
    write_csv(
        reports / "sample_level_metrics.csv",
        metrics,
        [
            "sample_id",
            "disagreement_stratum",
            "llm_normalized_entropy",
            "emoji_jsd",
            "region_jsd",
            "emoji_overlap",
            "region_overlap",
            "top1_emoji_agreement",
            "top1_region_agreement",
            "delta_projection",
            "human_loo_emoji_jsd",
            "human_loo_region_jsd",
            "human_loo_emoji_overlap",
            "human_loo_region_overlap",
        ],
    )
    write_csv(
        reports / "main_results.csv",
        [
            {
                "comparison": "LLM confidence-weighted vs human unweighted",
                "emoji_jsd_mean": overall["emoji_jsd_mean"],
                "emoji_jsd_ci_low": overall["emoji_jsd_ci_low"],
                "emoji_jsd_ci_high": overall["emoji_jsd_ci_high"],
                "region_jsd_mean": overall["region_jsd_mean"],
                "region_jsd_ci_low": overall["region_jsd_ci_low"],
                "region_jsd_ci_high": overall["region_jsd_ci_high"],
                "emoji_overlap_mean": overall["emoji_overlap_mean"],
                "region_overlap_mean": overall["region_overlap_mean"],
                "top1_emoji_agreement": overall["top1_emoji_agreement"],
                "top1_region_agreement": overall["top1_region_agreement"],
            }
        ],
        [
            "comparison",
            "emoji_jsd_mean",
            "emoji_jsd_ci_low",
            "emoji_jsd_ci_high",
            "region_jsd_mean",
            "region_jsd_ci_low",
            "region_jsd_ci_high",
            "emoji_overlap_mean",
            "region_overlap_mean",
            "top1_emoji_agreement",
            "top1_region_agreement",
        ],
    )
    write_csv(
        reports / "results_by_disagreement.csv",
        by_stratum,
        ["stratum", "n", "emoji_jsd", "region_jsd", "region_overlap", "top1_region_agreement"],
    )
    write_csv(
        reports / "confidence_robustness.csv",
        robustness_rows,
        [
            "human_aggregation",
            "n",
            "emoji_jsd_mean",
            "emoji_jsd_ci_low",
            "emoji_jsd_ci_high",
            "region_jsd_mean",
            "region_jsd_ci_low",
            "region_jsd_ci_high",
            "emoji_overlap_mean",
            "emoji_overlap_ci_low",
            "emoji_overlap_ci_high",
            "region_overlap_mean",
            "region_overlap_ci_low",
            "region_overlap_ci_high",
        ],
    )
    loo_rows = [
        {"metric": metric, "mean": vals["mean"], "ci_low": vals["ci_low"], "ci_high": vals["ci_high"]}
        for metric, vals in loo_agg.items()
    ]
    write_csv(reports / "human_leave_one_out.csv", loo_rows, ["metric", "mean", "ci_low", "ci_high"])
    report_md = [
        "# LLM-Human Emoji Distribution Audit",
        "",
        f"N = {len(metrics)} samples; bootstrap samples = {bootstrap_samples}; seed = {seed}.",
        "",
        "Human leave-one-out disagreement is included only as contextual reference. It uses one human annotation versus the other two and is not a strictly equivalent baseline for LLM-vs-aggregated-human comparison.",
        "",
        f"- Emoji JSD: {overall['emoji_jsd_mean']:.4f} [{overall['emoji_jsd_ci_low']:.4f}, {overall['emoji_jsd_ci_high']:.4f}]",
        f"- Region JSD: {overall['region_jsd_mean']:.4f} [{overall['region_jsd_ci_low']:.4f}, {overall['region_jsd_ci_high']:.4f}]",
        f"- Delta_projection: {overall['delta_projection_mean']:.4f} [{overall['delta_projection_ci_low']:.4f}, {overall['delta_projection_ci_high']:.4f}]",
    ]
    (reports / "analysis_report.md").write_text("\n".join(report_md) + "\n", encoding="utf-8")
    snippet = (
        "We conducted a small-scale pilot direct comparison between the aggregated LLM emoji distribution and three independent human annotations on the same 120 test-split utterances. "
        f"The mean emoji-level Jensen-Shannon divergence was {overall['emoji_jsd_mean']:.3f} "
        f"(95% bootstrap CI [{overall['emoji_jsd_ci_low']:.3f}, {overall['emoji_jsd_ci_high']:.3f}]), "
        f"and the mean latent-region Jensen-Shannon divergence was {overall['region_jsd_mean']:.3f} "
        f"(95% bootstrap CI [{overall['region_jsd_ci_low']:.3f}, {overall['region_jsd_ci_high']:.3f}]). "
        "These annotations are not treated as gold labels, and the experiment is intended as a pilot diagnostic of possible LLM annotation bias rather than evidence of cultural universality or equivalence between LLM and human judgments."
    )
    (reports / "rebuttal_snippet.md").write_text(snippet + "\n", encoding="utf-8")
    return summary


def make_synthetic_exports(experiment_dir: Path, out_dir: Path) -> None:
    data = load_experiment_dir(experiment_dir)
    emojis = [row["emoji"] for row in data["emoji_inventory"]]
    private = data["private_items"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, 4):
        answers = []
        for order, item in enumerate(private, 1):
            top = item["top1_llm_emoji"]
            emoji = top if idx != 3 else emojis[(emojis.index(top) + 1) % len(emojis)]
            answers.append(
                {
                    "sample_id": item["sample_id"],
                    "selected_emoji": emoji,
                    "unicode": codepoint_string(emoji),
                    "confidence": idx + 2 if idx < 3 else 3,
                    "item_order": order,
                    "elapsed_ms": 1000 + order,
                }
            )
        payload = {
            "schema_version": data["config"]["schema_version"],
            "experiment_id": data["config"]["experiment_id"],
            "annotator_id": f"annotator_{idx}",
            "questionnaire_id": f"questionnaire_{idx}",
            "sample_count": 120,
            "completed_count": 120,
            "created_at": utc_now(),
            "exported_at": utc_now(),
            "answers": answers,
        }
        write_json(out_dir / f"annotator_{idx}_annotations.json", payload)
