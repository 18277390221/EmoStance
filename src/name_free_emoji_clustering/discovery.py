from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
CSV_SOFT_LABEL_COLUMNS = {
    "dialogue_id",
    "turn_id",
    "split",
    "role",
    "text",
    "emoji",
    "soft_prob",
    "top1_emoji",
    "top1_prob",
    "entropy",
    "agreement_type",
}


@dataclass(frozen=True)
class Candidate:
    kind: str
    paths: tuple[Path, ...]
    score: int
    fields: tuple[str, ...]
    rationale: str


@dataclass
class DiscoveryResult:
    root: Path
    soft_label_table: Candidate | None = None
    raw_multimodel_annotations: Candidate | None = None
    text_table: Candidate | None = None
    emoji_frequency_tables: list[Candidate] = field(default_factory=list)
    rejected_notes: list[str] = field(default_factory=list)


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except UnicodeDecodeError:
        return []


def inspect_csv(path: Path) -> Candidate | None:
    header = read_csv_header(path)
    if not header:
        return None

    columns = set(header)
    if CSV_SOFT_LABEL_COLUMNS.issubset(columns):
        score = 100 + len(CSV_SOFT_LABEL_COLUMNS.intersection(columns))
        if "mean_confidence" in columns:
            score += 8
        if "support_model_count" in columns:
            score += 8
        if "utterance_id" in columns:
            score += 4
        return Candidate(
            kind="standardized_soft_label_csv",
            paths=(path,),
            score=score,
            fields=tuple(header),
            rationale=(
                "CSV contains utterance metadata, one emoji column, soft probability, "
                "top1/entropy, and agreement columns."
            ),
        )

    if {"utterance_id", "text", "role", "split"}.issubset(columns):
        return Candidate(
            kind="utterance_text_csv",
            paths=(path,),
            score=80,
            fields=tuple(header),
            rationale="CSV contains utterance identifiers, text, role, and split.",
        )

    if {"dialogue_id", "turn_id", "text", "role", "split"}.issubset(columns):
        return Candidate(
            kind="utterance_text_csv",
            paths=(path,),
            score=75,
            fields=tuple(header),
            rationale="CSV contains reconstructable utterance identifiers, text, role, and split.",
        )

    if {"emoji", "count"}.issubset(columns):
        score = 60
        if "observed" in columns:
            score += 20
        if {"name", "name_en", "aliases", "unicode"}.intersection(columns):
            score -= 3
        return Candidate(
            kind="emoji_frequency_csv",
            paths=(path,),
            score=score,
            fields=tuple(header),
            rationale=(
                "CSV contains emoji and count columns"
                + (" plus an observed flag." if "observed" in columns else ".")
            ),
        )

    return None


def read_first_jsonl_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return obj if isinstance(obj, dict) else None
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    return None


def inspect_jsonl(path: Path) -> Candidate | None:
    obj = read_first_jsonl_object(path)
    if obj is None:
        return None

    keys = set(obj)
    if {"dialogue_id", "turn_id", "split", "role"}.issubset(keys) and (
        {"model_emojis", "model_confidences"}.issubset(keys)
        or {"soft_label", "models"}.issubset(keys)
    ):
        score = 105
        if "utterance" in keys or "text" in keys:
            score += 10
        return Candidate(
            kind="utterance_soft_label_jsonl",
            paths=(path,),
            score=score,
            fields=tuple(sorted(keys)),
            rationale=(
                "JSONL rows contain utterance metadata plus per-model emojis/confidences "
                "and soft-label fields."
            ),
        )
    return None


def read_json_sample(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None

    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return None


def inspect_json(path: Path) -> Candidate | None:
    sample = read_json_sample(path)
    if not isinstance(sample, dict):
        return None

    keys = set(sample)
    turns = sample.get("turns")
    first_turn = turns[0] if isinstance(turns, list) and turns else {}
    turn_keys = set(first_turn) if isinstance(first_turn, dict) else set()

    if (
        {"dialogue_id", "split", "turns"}.issubset(keys)
        and {"turn_id", "utterance"}.issubset(turn_keys)
        and isinstance(first_turn.get("emoji"), dict)
        and isinstance(first_turn.get("confidence"), dict)
    ):
        fields = tuple(sorted(keys | {f"turn.{key}" for key in turn_keys}))
        return Candidate(
            kind="multimodel_dialogue_json",
            paths=(path,),
            score=100,
            fields=fields,
            rationale=(
                "JSON dialogue records contain turns with per-model emoji and confidence maps. "
                "Transition keys, if present, are ignored by the clustering stage."
            ),
        )

    if (
        {"dialogue_id", "split", "turns"}.issubset(keys)
        and {"turn_id", "utterance", "emoji_annotation"}.issubset(turn_keys)
    ):
        fields = tuple(sorted(keys | {f"turn.{key}" for key in turn_keys}))
        return Candidate(
            kind="single_model_dialogue_json",
            paths=(path,),
            score=45,
            fields=fields,
            rationale="JSON dialogue records contain single-model annotations only.",
        )

    return None


def combine_multimodel_json_candidates(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None

    by_parent: dict[Path, list[Candidate]] = {}
    for candidate in candidates:
        by_parent.setdefault(candidate.paths[0].parent, []).append(candidate)

    groups = sorted(
        by_parent.values(),
        key=lambda group: (sum(item.score for item in group), len(group)),
        reverse=True,
    )
    best_group = groups[0]
    paths = tuple(sorted((item.paths[0] for item in best_group), key=lambda p: str(p)))
    fields = best_group[0].fields
    return Candidate(
        kind="multimodel_dialogue_json_group",
        paths=paths,
        score=sum(item.score for item in best_group),
        fields=fields,
        rationale=(
            "Best same-directory group of multi-model dialogue JSON files; together they "
            "cover split-level raw annotations that can reconstruct soft labels."
        ),
    )


def combine_single_model_json_candidates(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None

    paths = tuple(sorted((candidate.paths[0] for candidate in candidates), key=lambda p: str(p)))
    fields = candidates[0].fields
    return Candidate(
        kind="single_model_dialogue_json_group",
        paths=paths,
        score=70 + sum(item.score for item in candidates),
        fields=fields,
        rationale=(
            "Group of single-model dialogue JSON files; model names are inferred from "
            "their parent directories and merged by split/dialogue_id/turn_id."
        ),
    )


def discover_inputs(root: Path) -> DiscoveryResult:
    result = DiscoveryResult(root=root)
    csv_candidates: list[Candidate] = []
    jsonl_candidates: list[Candidate] = []
    multimodel_json_candidates: list[Candidate] = []
    single_model_json_candidates: list[Candidate] = []

    for path in iter_files(root, {".csv"}):
        candidate = inspect_csv(path)
        if candidate is not None:
            csv_candidates.append(candidate)

    for path in iter_files(root, {".jsonl"}):
        candidate = inspect_jsonl(path)
        if candidate is not None:
            jsonl_candidates.append(candidate)

    for path in iter_files(root, {".json"}):
        candidate = inspect_json(path)
        if candidate is None:
            continue
        if candidate.kind == "multimodel_dialogue_json":
            multimodel_json_candidates.append(candidate)
        elif candidate.kind == "single_model_dialogue_json":
            single_model_json_candidates.append(candidate)

    soft_csvs = [c for c in csv_candidates if c.kind == "standardized_soft_label_csv"]
    soft_jsonls = [c for c in jsonl_candidates if c.kind == "utterance_soft_label_jsonl"]
    text_tables = [c for c in csv_candidates if c.kind == "utterance_text_csv"]
    frequency_tables = [c for c in csv_candidates if c.kind == "emoji_frequency_csv"]

    if soft_csvs:
        result.soft_label_table = sorted(soft_csvs, key=lambda c: c.score, reverse=True)[0]
    if soft_jsonls:
        jsonl = sorted(soft_jsonls, key=lambda c: c.score, reverse=True)[0]
        if result.raw_multimodel_annotations is None or jsonl.score > result.raw_multimodel_annotations.score:
            result.raw_multimodel_annotations = jsonl

    multimodel_group = combine_multimodel_json_candidates(multimodel_json_candidates)
    if multimodel_group is not None:
        if (
            result.raw_multimodel_annotations is None
            or multimodel_group.score > result.raw_multimodel_annotations.score
        ):
            result.raw_multimodel_annotations = multimodel_group

    single_model_group = combine_single_model_json_candidates(single_model_json_candidates)
    if single_model_group is not None:
        if (
            result.raw_multimodel_annotations is None
            or single_model_group.score > result.raw_multimodel_annotations.score
        ):
            result.raw_multimodel_annotations = single_model_group

    if text_tables:
        result.text_table = sorted(text_tables, key=lambda c: c.score, reverse=True)[0]
    elif result.soft_label_table is not None:
        result.text_table = Candidate(
            kind="text_embedded_in_soft_label_table",
            paths=result.soft_label_table.paths,
            score=result.soft_label_table.score,
            fields=result.soft_label_table.fields,
            rationale="The selected soft-label table already contains utterance text, role, and split.",
        )

    result.emoji_frequency_tables = sorted(
        frequency_tables,
        key=lambda c: (c.score, -len(c.fields), str(c.paths[0])),
        reverse=True,
    )

    if result.soft_label_table is None and result.raw_multimodel_annotations is None:
        result.rejected_notes.append(
            "No standardized soft-label table or raw multi-model annotation source was discovered."
        )

    return result


def candidate_to_markdown(candidate: Candidate | None) -> str:
    if candidate is None:
        return "- Not found.\n"
    path_text = ", ".join(f"`{path}`" for path in candidate.paths)
    fields = ", ".join(f"`{field}`" for field in candidate.fields[:24])
    if len(candidate.fields) > 24:
        fields += ", ..."
    return (
        f"- Kind: `{candidate.kind}`\n"
        f"- Path(s): {path_text}\n"
        f"- Score: `{candidate.score}`\n"
        f"- Why selected: {candidate.rationale}\n"
        f"- Schema/content signals: {fields}\n"
    )


def write_discovery_report(result: DiscoveryResult, output_path: Path) -> None:
    lines = [
        "# Name-free Emoji Clustering Input Discovery",
        "",
        "The scan uses schema/content signals only. Emoji names, aliases, Unicode descriptions, "
        "and transition tables are not used as clustering features.",
        "",
        "## A. Multi-model utterance-level annotations",
        "",
        candidate_to_markdown(result.raw_multimodel_annotations),
        "## B. Standardized soft-label table",
        "",
        candidate_to_markdown(result.soft_label_table),
        "## C. Utterance text table",
        "",
        candidate_to_markdown(result.text_table),
        "## D. Emoji pool / observed-frequency tables",
        "",
    ]
    if result.emoji_frequency_tables:
        for candidate in result.emoji_frequency_tables[:12]:
            lines.append(candidate_to_markdown(candidate))
    else:
        lines.append("- Not found.\n")

    lines.extend(
        [
            "## Schema Assumptions",
            "",
            "- Utterance identity is `split + dialogue_id + turn_id` when no explicit `utterance_id` exists.",
            "- Soft-label rows are utterance-emoji rows; repeated utterance-level statistics are normalized into canonical columns.",
            "- Row-level confidence means confidence for models supporting that row's emoji; utterance-level mean confidence is reconstructed from support counts when available.",
            "- Frequency/pool files are optional and are read only through `emoji`, `count`, and `observed` columns when present.",
            "- Transition keys and transition-derived files are excluded from all clustering features.",
            "",
        ]
    )
    if result.rejected_notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in result.rejected_notes)
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
