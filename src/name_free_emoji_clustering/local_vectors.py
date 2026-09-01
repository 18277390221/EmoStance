from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .embeddings import HashedTfidfSentenceEncoder, cosine_matrix
from .soft_membership import cluster_sort_key


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
VALIDATION_TOLERANCE = 1e-8


@dataclass(frozen=True)
class MembershipEntry:
    emoji: str
    cluster_id: str
    membership: float
    source_type: str
    observed_count: int | None


@dataclass(frozen=True)
class MembershipArtifact:
    path: Path
    entries: tuple[MembershipEntry, ...]
    score: int
    membership_column: str
    rationale: str


@dataclass(frozen=True)
class EmbeddingArtifact:
    path: Path | None
    emojis: tuple[str, ...]
    vectors: np.ndarray
    vector_name: str
    source_mode: str
    score: int
    rationale: str


@dataclass(frozen=True)
class SimilarityArtifact:
    path: Path | None
    emojis: tuple[str, ...]
    matrix: np.ndarray
    source_mode: str
    score: int
    rationale: str


@dataclass(frozen=True)
class NeighborRow:
    emoji: str
    cluster_id: str
    neighbor_emoji: str
    rank: int
    similarity: float
    weight: float
    is_self: bool
    smoothing_fallback: bool


@dataclass(frozen=True)
class LocalVectorRow:
    emoji: str
    cluster_id: str
    membership: float
    source_type: str
    observed_count: int | None
    cluster_size: int
    neighbor_count: int
    smoothing_fallback: bool
    global_vector: np.ndarray
    local_vector: np.ndarray
    l2_delta: float
    cosine_similarity: float
    neighbor_emojis: tuple[str, ...]
    neighbor_weights: tuple[float, ...]
    neighbor_similarities: tuple[float, ...]


@dataclass(frozen=True)
class LocalVectorSummary:
    emoji_count: int
    membership_pair_count: int
    cluster_count: int
    average_neighbors: float
    too_small_clusters: tuple[str, ...]
    ambiguous_row_count: int
    ambiguous_mean_l2_delta: float
    ambiguous_max_l2_delta: float


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    except (OSError, UnicodeDecodeError):
        return []


def discover_membership_artifacts(root: Path, membership_column: str) -> list[MembershipArtifact]:
    artifacts: list[MembershipArtifact] = []
    for path in iter_files(root, {".csv"}):
        header = read_csv_header(path)
        columns = set(header)
        if not {"emoji", "cluster_id"}.issubset(columns):
            continue
        selected_column = membership_column
        if selected_column not in columns:
            if "membership_raw" in columns:
                selected_column = "membership_raw"
            elif "membership_prob" in columns:
                selected_column = "membership_prob"
            else:
                continue

        entries: list[MembershipEntry] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    emoji = row.get("emoji")
                    cluster_id = row.get("cluster_id")
                    probability = parse_float(row.get(selected_column))
                    if not emoji or not cluster_id or probability is None or probability <= 0:
                        continue
                    entries.append(
                        MembershipEntry(
                            emoji=emoji,
                            cluster_id=cluster_id,
                            membership=probability,
                            source_type=row.get("source_type", ""),
                            observed_count=parse_int(row.get("observed_count")),
                        )
                    )
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        if not entries:
            continue

        score = 100 + len(entries)
        if "membership_raw" in columns:
            score += 25
        if "membership_sharp" in columns:
            score += 10
        if "source_type" in columns:
            score += 5
        artifacts.append(
            MembershipArtifact(
                path=path,
                entries=tuple(entries),
                score=score,
                membership_column=selected_column,
                rationale=(
                    f"CSV has {len(entries)} positive emoji-cluster rows using "
                    f"`{selected_column}` as m(e,c)."
                ),
            )
        )

    return sorted(
        artifacts,
        key=lambda artifact: (
            artifact.score,
            artifact.path.stat().st_mtime if artifact.path.exists() else 0.0,
            str(artifact.path),
        ),
        reverse=True,
    )


def load_npz_embedding_artifact(path: Path) -> EmbeddingArtifact | None:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    keys = set(data.keys())
    if "emojis" not in keys:
        return None
    vector_name = ""
    score = 0
    if "shrunk_centroids" in keys:
        vector_name = "shrunk_centroids"
        score = 300
    elif "raw_centroids" in keys:
        vector_name = "raw_centroids"
        score = 250
    else:
        return None

    vectors = data[vector_name].astype(np.float32, copy=False)
    emojis = tuple(str(value) for value in data["emojis"])
    if vectors.ndim != 2 or len(emojis) != vectors.shape[0]:
        return None
    return EmbeddingArtifact(
        path=path,
        emojis=emojis,
        vectors=vectors,
        vector_name=vector_name,
        source_mode="intrinsic_embedding_npz",
        score=score + len(emojis),
        rationale=f"NPZ contains `emojis` and `{vector_name}` in the intrinsic clustering output.",
    )


def parse_vector_json(text: str) -> list[float] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    values = []
    for item in parsed:
        value = parse_float(item)
        if value is None:
            return None
        values.append(value)
    return values


def load_csv_embedding_artifact(path: Path) -> EmbeddingArtifact | None:
    header = read_csv_header(path)
    if "emoji" not in header:
        return None
    vector_column = ""
    score = 0
    if "shrunk_centroid_json" in header:
        vector_column = "shrunk_centroid_json"
        score = 200
    elif "raw_centroid_json" in header:
        vector_column = "raw_centroid_json"
        score = 175
    else:
        return None
    emojis: list[str] = []
    vectors: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                emoji = row.get("emoji")
                vector = parse_vector_json(row.get(vector_column, ""))
                if not emoji or vector is None:
                    continue
                emojis.append(emoji)
                vectors.append(vector)
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if not vectors:
        return None
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        return None
    return EmbeddingArtifact(
        path=path,
        emojis=tuple(emojis),
        vectors=np.array(vectors, dtype=np.float32),
        vector_name=vector_column,
        source_mode="intrinsic_embedding_csv",
        score=score + len(emojis),
        rationale=f"CSV contains `{vector_column}` vectors from intrinsic clustering outputs.",
    )


def discover_embedding_artifacts(root: Path) -> list[EmbeddingArtifact]:
    artifacts: list[EmbeddingArtifact] = []
    for path in iter_files(root, {".npz"}):
        artifact = load_npz_embedding_artifact(path)
        if artifact is not None:
            artifacts.append(artifact)
    for path in iter_files(root, {".csv"}):
        artifact = load_csv_embedding_artifact(path)
        if artifact is not None:
            artifacts.append(artifact)
    return sorted(
        artifacts,
        key=lambda artifact: (
            artifact.score,
            artifact.path.stat().st_mtime if artifact.path and artifact.path.exists() else 0.0,
            str(artifact.path),
        ),
        reverse=True,
    )


def matrix_artifact_from_npz(path: Path) -> SimilarityArtifact | None:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    keys = set(data.keys())
    if "emojis" not in keys:
        return None
    matrix_key = ""
    score = 0
    if "fused_affinity" in keys:
        matrix_key = "fused_affinity"
        score = 300
    elif "context_similarity" in keys:
        matrix_key = "context_similarity"
        score = 250
    elif "confusion_similarity" in keys:
        matrix_key = "confusion_similarity"
        score = 200
    else:
        return None
    emojis = tuple(str(value) for value in data["emojis"])
    matrix = data[matrix_key].astype(np.float32, copy=False)
    if matrix.shape != (len(emojis), len(emojis)):
        return None
    return SimilarityArtifact(
        path=path,
        emojis=emojis,
        matrix=matrix,
        source_mode=f"intrinsic_{matrix_key}_npz",
        score=score + len(emojis),
        rationale=f"NPZ contains square `{matrix_key}` matrix aligned to emojis.",
    )


def matrix_artifact_from_csv(path: Path) -> SimilarityArtifact | None:
    header = read_csv_header(path)
    if len(header) < 3 or header[0] not in {"emoji", "source_emoji", "source_cluster"}:
        return None
    if header[0] == "source_cluster":
        return None
    labels = tuple(header[1:])
    rows: list[list[float]] = []
    row_labels: list[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) != len(header):
                    return None
                row_labels.append(row[0])
                values = [parse_float(value) for value in row[1:]]
                if any(value is None for value in values):
                    return None
                rows.append([float(value) for value in values if value is not None])
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if tuple(row_labels) != labels:
        return None
    matrix = np.array(rows, dtype=np.float32)
    score = 250 if "fused" in path.name else 175
    return SimilarityArtifact(
        path=path,
        emojis=labels,
        matrix=matrix,
        source_mode="intrinsic_similarity_csv",
        score=score + len(labels),
        rationale="CSV is a square emoji-by-emoji similarity or affinity matrix.",
    )


def discover_similarity_artifacts(root: Path) -> list[SimilarityArtifact]:
    artifacts: list[SimilarityArtifact] = []
    for path in iter_files(root, {".npz"}):
        artifact = matrix_artifact_from_npz(path)
        if artifact is not None:
            artifacts.append(artifact)
    for path in iter_files(root, {".csv"}):
        artifact = matrix_artifact_from_csv(path)
        if artifact is not None:
            artifacts.append(artifact)
    return sorted(
        artifacts,
        key=lambda artifact: (
            artifact.score,
            artifact.path.stat().st_mtime if artifact.path and artifact.path.exists() else 0.0,
            str(artifact.path),
        ),
        reverse=True,
    )


def inspect_soft_label_csv(path: Path) -> bool:
    header = read_csv_header(path)
    columns = set(header)
    return {"emoji", "soft_prob", "text"}.issubset(columns) and (
        "utterance_id" in columns or {"dialogue_id", "turn_id", "split"}.issubset(columns)
    )


def make_utterance_id(row: dict[str, str]) -> str:
    if row.get("utterance_id"):
        return str(row["utterance_id"])
    return f"{row.get('split', '')}|{row.get('dialogue_id', '')}|{row.get('turn_id', '')}"


def reconstruct_embeddings_from_soft_labels(root: Path) -> EmbeddingArtifact:
    candidates = [
        path for path in iter_files(root, {".csv"})
        if inspect_soft_label_csv(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No intrinsic emoji embeddings were found, and no utterance soft-label table "
            "with `emoji`, `soft_prob`, and `text` could be discovered for reconstruction."
        )
    candidates.sort(
        key=lambda path: (
            "canonical" in path.name,
            path.stat().st_mtime if path.exists() else 0.0,
            str(path),
        ),
        reverse=True,
    )
    path = candidates[0]
    utterance_texts: OrderedDict[str, str] = OrderedDict()
    weights_by_utterance: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            utterance_id = make_utterance_id(row)
            utterance_texts.setdefault(utterance_id, row.get("text", ""))
            emoji = row.get("emoji")
            soft_prob = parse_float(row.get("soft_prob"))
            if emoji and soft_prob is not None and soft_prob > 0:
                weights_by_utterance[utterance_id][emoji] = (
                    weights_by_utterance[utterance_id].get(emoji, 0.0) + soft_prob
                )
    utterance_ids = list(utterance_texts)
    encoder = HashedTfidfSentenceEncoder(dim=256)
    utterance_vectors = encoder.fit_transform([utterance_texts[utterance_id] for utterance_id in utterance_ids])
    emoji_weights: dict[str, float] = defaultdict(float)
    emoji_vectors: dict[str, np.ndarray] = {}
    for row_index, utterance_id in enumerate(utterance_ids):
        vector = utterance_vectors[row_index]
        for emoji, weight in weights_by_utterance.get(utterance_id, {}).items():
            emoji_weights[emoji] += weight
            if emoji not in emoji_vectors:
                emoji_vectors[emoji] = np.zeros_like(vector)
            emoji_vectors[emoji] += weight * vector
    emojis = tuple(sorted(emoji_vectors, key=lambda emoji: (-emoji_weights[emoji], emoji)))
    vectors = np.vstack(
        [
            emoji_vectors[emoji] / max(emoji_weights[emoji], 1e-12)
            for emoji in emojis
        ]
    ).astype(np.float32)
    return EmbeddingArtifact(
        path=path,
        emojis=emojis,
        vectors=vectors,
        vector_name="reconstructed_hashed_tfidf_centroids",
        source_mode="reconstructed_from_soft_labels",
        score=0,
        rationale=(
            "Reconstructed emoji vectors from utterance text embeddings weighted by "
            "utterance-level soft emoji labels."
        ),
    )


def align_embeddings(
    embeddings: EmbeddingArtifact,
    required_emojis: Iterable[str],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    vector_by_emoji = {
        emoji: embeddings.vectors[index]
        for index, emoji in enumerate(embeddings.emojis)
    }
    missing = tuple(sorted(set(required_emojis) - set(vector_by_emoji)))
    if missing:
        preview = ", ".join(missing[:20])
        raise ValueError(f"Embedding artifact is missing membership emojis: {preview}")
    return vector_by_emoji, missing


def align_similarity(
    similarity: SimilarityArtifact | None,
    embeddings: EmbeddingArtifact,
    vector_by_emoji: dict[str, np.ndarray],
) -> SimilarityArtifact:
    if similarity is not None:
        return similarity
    labels = tuple(embeddings.emojis)
    matrix = cosine_matrix(
        np.vstack([vector_by_emoji[emoji] for emoji in labels]).astype(np.float32)
    ).astype(np.float32)
    return SimilarityArtifact(
        path=None,
        emojis=labels,
        matrix=matrix,
        source_mode="cosine_from_global_embeddings",
        score=0,
        rationale="No intrinsic similarity matrix found; using cosine similarity of global vectors.",
    )


def similarity_lookup(similarity: SimilarityArtifact) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}
    for row_index, source in enumerate(similarity.emojis):
        for col_index, target in enumerate(similarity.emojis):
            value = float(similarity.matrix[row_index, col_index])
            if math.isfinite(value):
                lookup[(source, target)] = value
    return lookup


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def stable_softmax(similarities: list[float], tau_nn: float) -> list[float]:
    if tau_nn <= 0:
        raise ValueError("tau_nn must be positive.")
    if not similarities:
        return []
    max_value = max(similarities)
    weights = [math.exp((value - max_value) / tau_nn) for value in similarities]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(similarities)] * len(similarities)
    return [weight / total for weight in weights]


def select_neighbors(
    emoji: str,
    cluster_emojis: list[str],
    similarity_values: dict[tuple[str, str], float],
    vector_by_emoji: dict[str, np.ndarray],
    max_neighbors: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    similarities = []
    for candidate in cluster_emojis:
        value = similarity_values.get((emoji, candidate))
        if value is None:
            value = cosine_similarity(vector_by_emoji[emoji], vector_by_emoji[candidate])
        similarities.append((candidate, value))
    ordered = sorted(
        similarities,
        key=lambda item: (item[0] != emoji, -item[1], item[0]),
    )[:max_neighbors]
    return tuple(item[0] for item in ordered), tuple(float(item[1]) for item in ordered)


def build_local_vectors(
    membership: MembershipArtifact,
    embeddings: EmbeddingArtifact,
    similarity: SimilarityArtifact | None,
    max_k: int,
    tau_nn: float,
    min_smoothing_cluster_size: int,
) -> tuple[list[LocalVectorRow], list[NeighborRow], LocalVectorSummary]:
    if max_k <= 0:
        raise ValueError("max_k must be positive.")
    if min_smoothing_cluster_size <= 0:
        raise ValueError("min_smoothing_cluster_size must be positive.")

    required_emojis = {entry.emoji for entry in membership.entries}
    vector_by_emoji, _ = align_embeddings(embeddings, required_emojis)
    similarity = align_similarity(similarity, embeddings, vector_by_emoji)
    similarity_values = similarity_lookup(similarity)

    entries_by_cluster: dict[str, list[MembershipEntry]] = defaultdict(list)
    entries_by_emoji: dict[str, list[MembershipEntry]] = defaultdict(list)
    for entry in membership.entries:
        entries_by_cluster[entry.cluster_id].append(entry)
        entries_by_emoji[entry.emoji].append(entry)

    local_rows: list[LocalVectorRow] = []
    neighbor_rows: list[NeighborRow] = []
    too_small_clusters: list[str] = []

    for cluster_id in sorted(entries_by_cluster, key=cluster_sort_key):
        cluster_entries = sorted(
            entries_by_cluster[cluster_id],
            key=lambda entry: (-(entry.observed_count or 0), entry.emoji),
        )
        cluster_emojis = [entry.emoji for entry in cluster_entries]
        cluster_size = len(cluster_emojis)
        fallback = cluster_size < min_smoothing_cluster_size
        if fallback:
            too_small_clusters.append(cluster_id)
        neighbor_limit = min(max_k, cluster_size)

        for entry in cluster_entries:
            global_vector = vector_by_emoji[entry.emoji]
            if fallback:
                neighbor_emojis = (entry.emoji,)
                neighbor_sims = (1.0,)
                neighbor_weights = (1.0,)
                local_vector = global_vector.copy()
            else:
                neighbor_emojis, neighbor_sims = select_neighbors(
                    entry.emoji,
                    cluster_emojis,
                    similarity_values,
                    vector_by_emoji,
                    neighbor_limit,
                )
                neighbor_weights = tuple(stable_softmax(list(neighbor_sims), tau_nn))
                local_vector = np.zeros_like(global_vector)
                for neighbor_emoji, weight in zip(neighbor_emojis, neighbor_weights):
                    local_vector += weight * vector_by_emoji[neighbor_emoji]

            l2_delta = float(np.linalg.norm(local_vector - global_vector))
            cos_sim = cosine_similarity(global_vector, local_vector)
            local_rows.append(
                LocalVectorRow(
                    emoji=entry.emoji,
                    cluster_id=entry.cluster_id,
                    membership=entry.membership,
                    source_type=entry.source_type,
                    observed_count=entry.observed_count,
                    cluster_size=cluster_size,
                    neighbor_count=len(neighbor_emojis),
                    smoothing_fallback=fallback,
                    global_vector=global_vector,
                    local_vector=local_vector,
                    l2_delta=l2_delta,
                    cosine_similarity=cos_sim,
                    neighbor_emojis=neighbor_emojis,
                    neighbor_weights=neighbor_weights,
                    neighbor_similarities=neighbor_sims,
                )
            )
            for rank, (neighbor_emoji, similarity_value, weight) in enumerate(
                zip(neighbor_emojis, neighbor_sims, neighbor_weights),
                start=1,
            ):
                neighbor_rows.append(
                    NeighborRow(
                        emoji=entry.emoji,
                        cluster_id=entry.cluster_id,
                        neighbor_emoji=neighbor_emoji,
                        rank=rank,
                        similarity=similarity_value,
                        weight=weight,
                        is_self=entry.emoji == neighbor_emoji,
                        smoothing_fallback=fallback,
                    )
                )

    ambiguous_rows = [
        row for row in local_rows
        if len(entries_by_emoji[row.emoji]) > 1 or row.source_type == "soft_candidate"
    ]
    summary = LocalVectorSummary(
        emoji_count=len(required_emojis),
        membership_pair_count=len(local_rows),
        cluster_count=len(entries_by_cluster),
        average_neighbors=(
            sum(row.neighbor_count for row in local_rows) / len(local_rows)
            if local_rows
            else 0.0
        ),
        too_small_clusters=tuple(sorted(too_small_clusters, key=cluster_sort_key)),
        ambiguous_row_count=len(ambiguous_rows),
        ambiguous_mean_l2_delta=(
            sum(row.l2_delta for row in ambiguous_rows) / len(ambiguous_rows)
            if ambiguous_rows
            else 0.0
        ),
        ambiguous_max_l2_delta=max((row.l2_delta for row in ambiguous_rows), default=0.0),
    )
    return local_rows, neighbor_rows, summary


def vector_json(vector: np.ndarray) -> str:
    return json.dumps([round(float(value), 8) for value in vector], ensure_ascii=False)


def tuple_json(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def write_local_vectors(path: Path, rows: list[LocalVectorRow]) -> None:
    fieldnames = [
        "emoji",
        "cluster_id",
        "membership",
        "source_type",
        "observed_count",
        "cluster_size",
        "neighbor_count",
        "smoothing_fallback",
        "l2_delta",
        "cosine_similarity_global_local",
        "neighbor_emojis_json",
        "neighbor_weights_json",
        "neighbor_similarities_json",
        "global_vector_json",
        "local_vector_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.emoji, cluster_sort_key(item.cluster_id))):
            writer.writerow(
                {
                    "emoji": row.emoji,
                    "cluster_id": row.cluster_id,
                    "membership": f"{row.membership:.15f}",
                    "source_type": row.source_type,
                    "observed_count": row.observed_count if row.observed_count is not None else "",
                    "cluster_size": row.cluster_size,
                    "neighbor_count": row.neighbor_count,
                    "smoothing_fallback": row.smoothing_fallback,
                    "l2_delta": f"{row.l2_delta:.12f}",
                    "cosine_similarity_global_local": f"{row.cosine_similarity:.12f}",
                    "neighbor_emojis_json": tuple_json(row.neighbor_emojis),
                    "neighbor_weights_json": tuple_json(round(float(value), 12) for value in row.neighbor_weights),
                    "neighbor_similarities_json": tuple_json(
                        round(float(value), 12) for value in row.neighbor_similarities
                    ),
                    "global_vector_json": vector_json(row.global_vector),
                    "local_vector_json": vector_json(row.local_vector),
                }
            )


def write_neighbors(path: Path, rows: list[NeighborRow]) -> None:
    fieldnames = [
        "emoji",
        "cluster_id",
        "neighbor_emoji",
        "rank",
        "similarity",
        "weight",
        "is_self",
        "smoothing_fallback",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.emoji, cluster_sort_key(item.cluster_id), item.rank)):
            writer.writerow(
                {
                    "emoji": row.emoji,
                    "cluster_id": row.cluster_id,
                    "neighbor_emoji": row.neighbor_emoji,
                    "rank": row.rank,
                    "similarity": f"{row.similarity:.12f}",
                    "weight": f"{row.weight:.12f}",
                    "is_self": row.is_self,
                    "smoothing_fallback": row.smoothing_fallback,
                }
            )


def frequent_ambiguous_rows(rows: list[LocalVectorRow], limit: int = 12) -> list[LocalVectorRow]:
    ambiguous = [
        row for row in rows
        if row.source_type == "soft_candidate"
    ]
    return sorted(
        ambiguous,
        key=lambda row: (-(row.observed_count or 0), -row.membership, row.emoji, cluster_sort_key(row.cluster_id)),
    )[:limit]


def largest_smoothing_changes(rows: list[LocalVectorRow], limit: int = 8) -> list[LocalVectorRow]:
    ambiguous = [
        row for row in rows
        if row.source_type == "soft_candidate"
    ]
    return sorted(
        ambiguous,
        key=lambda row: (-row.l2_delta, row.emoji, cluster_sort_key(row.cluster_id)),
    )[:limit]


def write_summary_json(
    path: Path,
    membership: MembershipArtifact,
    embeddings: EmbeddingArtifact,
    similarity: SimilarityArtifact,
    summary: LocalVectorSummary,
    max_k: int,
    tau_nn: float,
    min_smoothing_cluster_size: int,
) -> None:
    payload = {
        "membership_matrix": str(membership.path),
        "membership_column": membership.membership_column,
        "embedding_source": str(embeddings.path) if embeddings.path is not None else None,
        "embedding_source_mode": embeddings.source_mode,
        "embedding_vector_name": embeddings.vector_name,
        "similarity_source": str(similarity.path) if similarity.path is not None else None,
        "similarity_source_mode": similarity.source_mode,
        "max_k": max_k,
        "tau_nn": tau_nn,
        "min_smoothing_cluster_size": min_smoothing_cluster_size,
        "emoji_count": summary.emoji_count,
        "membership_pair_count": summary.membership_pair_count,
        "cluster_count": summary.cluster_count,
        "average_neighbors": summary.average_neighbors,
        "too_small_clusters": list(summary.too_small_clusters),
        "ambiguous_row_count": summary.ambiguous_row_count,
        "ambiguous_mean_l2_delta": summary.ambiguous_mean_l2_delta,
        "ambiguous_max_l2_delta": summary.ambiguous_max_l2_delta,
        "hard_constraints": {
            "emoji_names_used": False,
            "aliases_used": False,
            "unicode_names_used": False,
            "external_emoji_lexicon_used": False,
            "transition_features_used": False,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(
    path: Path,
    membership: MembershipArtifact,
    embeddings: EmbeddingArtifact,
    similarity: SimilarityArtifact,
    summary: LocalVectorSummary,
    rows: list[LocalVectorRow],
    output_paths: dict[str, Path],
) -> None:
    lines = [
        "# Cluster-Internal Local Emoji Vectors",
        "",
        "## Inputs",
        "",
        f"- Membership matrix: `{membership.path}` using `{membership.membership_column}`.",
        f"- Global embedding source: `{embeddings.path}` (`{embeddings.vector_name}`).",
        f"- Similarity source: `{similarity.path}` (`{similarity.source_mode}`).",
        "- Emoji names, aliases, Unicode names, external emoji lexicons, and transition information were not used.",
        "",
        "## Diagnostics",
        "",
        f"- Observed emojis: `{summary.emoji_count}`",
        f"- Emoji-cluster pairs: `{summary.membership_pair_count}`",
        f"- Clusters: `{summary.cluster_count}`",
        f"- Average intra-cluster neighbors used: `{summary.average_neighbors:.3f}`",
        f"- Too-small smoothing clusters: `{', '.join(summary.too_small_clusters) if summary.too_small_clusters else 'none'}`",
        f"- Ambiguous rows: `{summary.ambiguous_row_count}`",
        f"- Ambiguous mean L2 smoothing change: `{summary.ambiguous_mean_l2_delta:.6f}`",
        f"- Ambiguous max L2 smoothing change: `{summary.ambiguous_max_l2_delta:.6f}`",
        "",
        "## Frequent Ambiguous Emoji Neighbors",
        "",
        "| emoji | cluster | observed_count | membership | neighbors | weights |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in frequent_ambiguous_rows(rows):
        lines.append(
            f"| {row.emoji} | `{row.cluster_id}` | {row.observed_count or ''} | "
            f"{row.membership:.4f} | `{', '.join(row.neighbor_emojis)}` | "
            f"`{', '.join(f'{weight:.3f}' for weight in row.neighbor_weights)}` |"
        )

    lines.extend(
        [
            "",
            "## Largest Ambiguous Smoothing Changes",
            "",
            "| emoji | cluster | observed_count | L2 delta | cosine(global, local) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in largest_smoothing_changes(rows):
        lines.append(
            f"| {row.emoji} | `{row.cluster_id}` | {row.observed_count or ''} | "
            f"{row.l2_delta:.6f} | {row.cosine_similarity:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Local vectors: `{output_paths['local_vectors']}`",
            f"- Intra-cluster neighbors: `{output_paths['neighbors']}`",
            f"- Summary JSON: `{output_paths['summary_json']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def discover_required_artifacts(
    root: Path,
    membership_column: str,
) -> tuple[MembershipArtifact, EmbeddingArtifact, SimilarityArtifact]:
    membership_artifacts = discover_membership_artifacts(root, membership_column)
    if not membership_artifacts:
        raise FileNotFoundError(
            "No emoji-cluster membership matrix found. Expected CSV with `emoji`, "
            "`cluster_id`, and a positive membership column."
        )
    membership = membership_artifacts[0]
    embedding_artifacts = discover_embedding_artifacts(root)
    embeddings = embedding_artifacts[0] if embedding_artifacts else reconstruct_embeddings_from_soft_labels(root)
    similarity_artifacts = discover_similarity_artifacts(root)
    similarity = similarity_artifacts[0] if similarity_artifacts else None
    required_emojis = {entry.emoji for entry in membership.entries}
    vector_by_emoji, _ = align_embeddings(embeddings, required_emojis)
    aligned_similarity = align_similarity(similarity, embeddings, vector_by_emoji)
    return membership, embeddings, aligned_similarity


def build_and_write_local_vectors(
    root: Path,
    output_dir: Path | None,
    membership_column: str,
    max_k: int,
    tau_nn: float,
    min_smoothing_cluster_size: int,
) -> tuple[dict[str, Path], LocalVectorSummary]:
    membership, embeddings, similarity = discover_required_artifacts(root, membership_column)
    if output_dir is None:
        output_dir = membership.path.parent.parent / "local_vectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, neighbor_rows, summary = build_local_vectors(
        membership=membership,
        embeddings=embeddings,
        similarity=similarity,
        max_k=max_k,
        tau_nn=tau_nn,
        min_smoothing_cluster_size=min_smoothing_cluster_size,
    )
    output_paths = {
        "local_vectors": output_dir / "emoji_cluster_local_vectors.csv",
        "neighbors": output_dir / "cluster_intra_neighbors.csv",
        "summary_json": output_dir / "local_vector_summary.json",
        "report": output_dir / "local_vector_report.md",
    }
    write_local_vectors(output_paths["local_vectors"], rows)
    write_neighbors(output_paths["neighbors"], neighbor_rows)
    write_summary_json(
        output_paths["summary_json"],
        membership,
        embeddings,
        similarity,
        summary,
        max_k,
        tau_nn,
        min_smoothing_cluster_size,
    )
    write_report(output_paths["report"], membership, embeddings, similarity, summary, rows, output_paths)
    return output_paths, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cluster-internal neighbor-aggregated continuous emoji vectors."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root to search.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--membership-column",
        default="membership_raw",
        help="Membership column to use, default `membership_raw`.",
    )
    parser.add_argument("--max-k", type=int, default=5, help="Maximum intra-cluster neighbors.")
    parser.add_argument("--tau-nn", type=float, default=0.15, help="Neighbor softmax temperature.")
    parser.add_argument(
        "--min-smoothing-cluster-size",
        type=int,
        default=3,
        help="Clusters smaller than this fall back to unsmoothed global vectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else None
    )
    output_paths, summary = build_and_write_local_vectors(
        root=root,
        output_dir=output_dir,
        membership_column=args.membership_column,
        max_k=args.max_k,
        tau_nn=args.tau_nn,
        min_smoothing_cluster_size=args.min_smoothing_cluster_size,
    )
    print(f"Wrote local vectors: {output_paths['local_vectors']}")
    print(f"Wrote intra-cluster neighbors: {output_paths['neighbors']}")
    print(f"Wrote report: {output_paths['report']}")
    print(
        "Summary: "
        f"{summary.emoji_count} emojis, "
        f"{summary.membership_pair_count} emoji-cluster pairs, "
        f"average neighbors {summary.average_neighbors:.3f}."
    )


if __name__ == "__main__":
    main()
