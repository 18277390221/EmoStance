from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class EmbeddingCache:
    utterance_ids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, object]


class HashedTfidfSentenceEncoder:
    """Deterministic, dataset-intrinsic sentence encoder.

    This is intentionally name-free: it uses only utterance text and stable feature
    hashing over token and token-bigram features.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.dim = dim
        self.idf: np.ndarray | None = None

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def features(self, text: str) -> list[str]:
        tokens = self.tokenize(text)
        feats = [f"tok:{token}" for token in tokens]
        feats.extend(f"bigram:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        return feats

    def hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % self.dim
        sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
        return index, sign

    def fit(self, texts: list[str]) -> None:
        doc_freq = np.zeros(self.dim, dtype=np.float32)
        for text in texts:
            seen: set[int] = set()
            for feature in self.features(text):
                index, _ = self.hash_feature(feature)
                seen.add(index)
            for index in seen:
                doc_freq[index] += 1.0
        n_docs = max(len(texts), 1)
        self.idf = np.log((n_docs + 1.0) / (doc_freq + 1.0)) + 1.0

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise ValueError("Encoder must be fit before transform.")

        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for feature in self.features(text):
                index, sign = self.hash_feature(feature)
                matrix[row_index, index] += sign
            matrix[row_index] *= self.idf
            norm = float(np.linalg.norm(matrix[row_index]))
            if norm > 0:
                matrix[row_index] /= norm
        return matrix

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


def corpus_digest(utterance_ids: list[str], texts: list[str], dim: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(dim).encode("utf-8"))
    for utterance_id, text in zip(utterance_ids, texts):
        hasher.update(utterance_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(text.encode("utf-8", errors="replace"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_embedding_cache(npz_path: Path, metadata_path: Path, digest: str) -> EmbeddingCache | None:
    if not npz_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("corpus_digest") != digest:
        return None
    data = np.load(npz_path, allow_pickle=False)
    utterance_ids = [str(value) for value in data["utterance_ids"]]
    embeddings = data["embeddings"].astype(np.float32, copy=False)
    return EmbeddingCache(utterance_ids=utterance_ids, embeddings=embeddings, metadata=metadata)


def build_or_load_embeddings(
    utterance_ids: list[str],
    texts: list[str],
    output_npz_path: Path,
    metadata_path: Path,
    dim: int,
    force: bool = False,
) -> EmbeddingCache:
    digest = corpus_digest(utterance_ids, texts, dim)
    if not force:
        cached = load_embedding_cache(output_npz_path, metadata_path, digest)
        if cached is not None:
            return cached

    encoder = HashedTfidfSentenceEncoder(dim=dim)
    embeddings = encoder.fit_transform(texts)
    np.savez_compressed(
        output_npz_path,
        utterance_ids=np.array(utterance_ids),
        embeddings=embeddings,
        idf=encoder.idf if encoder.idf is not None else np.array([], dtype=np.float32),
    )
    metadata: dict[str, object] = {
        "encoder": "deterministic_hashed_tfidf",
        "embedding_dim": dim,
        "utterance_count": len(utterance_ids),
        "corpus_digest": digest,
        "feature_template": "token unigrams + adjacent token bigrams",
        "normalization": "hashed TF-IDF vectors are L2-normalized",
        "external_lexicon_used": False,
        "emoji_names_used": False,
        "justification": (
            "No pretrained sentence-transformers/transformers/sklearn encoder is installed in "
            "the project environment; this stable in-repo encoder uses only dataset utterance text."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return EmbeddingCache(utterance_ids=utterance_ids, embeddings=embeddings, metadata=metadata)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
    matrix = safe @ safe.T
    return np.clip(matrix, -1.0, 1.0)


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0
