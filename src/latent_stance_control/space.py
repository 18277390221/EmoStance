from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class StanceSpace:
    emojis: List[str]
    clusters: List[int]
    membership: np.ndarray
    emoji_vectors: np.ndarray

    @property
    def num_emojis(self) -> int:
        return len(self.emojis)

    @property
    def num_clusters(self) -> int:
        return len(self.clusters)

    @property
    def vector_dim(self) -> int:
        return int(self.emoji_vectors.shape[1])


def _as_cluster_id(value: Any) -> int:
    if isinstance(value, dict):
        for key in ("cluster", "cluster_id", "id"):
            if key in value:
                return _as_cluster_id(value[key])
    text = str(value)
    match = re.search(r"-?\d+", text)
    if not match:
        raise ValueError(f"Cannot parse cluster id from {value!r}")
    return int(match.group(0))


def _candidate_support(item: Any) -> Tuple[int, float]:
    if isinstance(item, dict):
        cluster = _as_cluster_id(item)
        support = float(item.get("support", item.get("score", item.get("weight", 1.0))))
        return cluster, support
    return _as_cluster_id(item), 1.0


def _json_vector(value: Any) -> np.ndarray | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=np.float32)
    return None


def load_emoji_vectors(vector_rows: List[Dict[str, Any]] | None) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    if not vector_rows:
        return vectors
    for row in vector_rows:
        emoji = str(row.get("emoji") or row.get("node") or row.get("label") or "").strip()
        if not emoji:
            continue
        vec = None
        for key in ("shrunk_centroid_json", "local_vector_json", "vector", "embedding", "z", "raw_centroid_json", "global_vector_json"):
            if key in row:
                vec = _json_vector(row.get(key))
                if vec is not None:
                    break
        if vec is not None:
            vectors[emoji] = vec.astype(np.float32)
    return vectors


def load_stance_space(cluster_rows: List[Dict[str, Any]], vector_rows: List[Dict[str, Any]] | None = None) -> StanceSpace:
    supports_by_emoji: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
    explicit_vectors = load_emoji_vectors(vector_rows)
    inline_vectors: Dict[str, np.ndarray] = {}

    for row in cluster_rows:
        emoji = str(row.get("emoji") or row.get("node") or row.get("label") or "").strip()
        if not emoji:
            continue

        if "cluster_id" in row and ("membership_raw" in row or "membership" in row):
            cluster = _as_cluster_id(row["cluster_id"])
            membership = row.get("membership_raw", row.get("membership", 0.0))
            supports_by_emoji[emoji][cluster] += max(float(membership), 0.0)
        elif any(k.startswith("raw_cluster_") for k in row):
            for key, value in row.items():
                if key.startswith("raw_cluster_"):
                    supports_by_emoji[emoji][_as_cluster_id(key)] += max(float(value or 0.0), 0.0)
        else:
            if "cluster" in row:
                supports_by_emoji[emoji][_as_cluster_id(row["cluster"])] += float(row.get("support", row.get("cluster_support", 1.0)))
            for item in row.get("candidateClusters", row.get("candidate_clusters", [])) or []:
                cluster, support = _candidate_support(item)
                supports_by_emoji[emoji][cluster] += max(float(support), 0.0)

        for key in ("vector", "embedding", "z"):
            if key in row:
                vec = _json_vector(row.get(key))
                if vec is not None:
                    inline_vectors[emoji] = vec.astype(np.float32)
                    break

    emojis = sorted(supports_by_emoji)
    clusters = sorted({cluster for supports in supports_by_emoji.values() for cluster in supports})
    if not emojis or not clusters:
        raise ValueError("Cluster artifact did not yield any emoji-cluster memberships.")
    cluster_index = {cluster: idx for idx, cluster in enumerate(clusters)}
    membership = np.zeros((len(emojis), len(clusters)), dtype=np.float32)
    for emoji_idx, emoji in enumerate(emojis):
        supports = supports_by_emoji[emoji]
        total = sum(supports.values()) or 1.0
        for cluster, support in supports.items():
            membership[emoji_idx, cluster_index[cluster]] = float(support / total)

    vectors = []
    if explicit_vectors or inline_vectors:
        dim = None
        for emoji in emojis:
            vec = explicit_vectors.get(emoji, inline_vectors.get(emoji))
            if vec is not None:
                dim = int(vec.shape[0])
                break
        if dim is not None:
            for emoji_idx, emoji in enumerate(emojis):
                vec = explicit_vectors.get(emoji, inline_vectors.get(emoji))
                if vec is None:
                    vec = membership[emoji_idx]
                    vec = np.pad(vec, (0, max(dim - vec.shape[0], 0)))[:dim]
                vectors.append(vec.astype(np.float32))
    emoji_vectors = np.stack(vectors).astype(np.float32) if vectors else membership.copy()
    return StanceSpace(emojis=emojis, clusters=clusters, membership=membership, emoji_vectors=emoji_vectors)


def emoji_distribution(votes: Dict[str, float], emojis: Sequence[str], eps: float = 1e-12) -> np.ndarray:
    index = {emoji: idx for idx, emoji in enumerate(emojis)}
    dist = np.zeros(len(emojis), dtype=np.float32)
    for emoji, weight in votes.items():
        idx = index.get(str(emoji))
        if idx is not None:
            dist[idx] += max(float(weight), 0.0)
    total = float(dist.sum())
    if total <= eps:
        dist[:] = 1.0 / max(len(dist), 1)
    else:
        dist /= total
    return dist


def cluster_distribution(q_emoji: np.ndarray, membership: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q_emoji, dtype=np.float32) @ np.asarray(membership, dtype=np.float32)
    q /= max(float(q.sum()), eps)
    return q.astype(np.float32)


def stance_vector(q_emoji: np.ndarray, emoji_vectors: np.ndarray) -> np.ndarray:
    return (np.asarray(q_emoji, dtype=np.float32) @ np.asarray(emoji_vectors, dtype=np.float32)).astype(np.float32)
