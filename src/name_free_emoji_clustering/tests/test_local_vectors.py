from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from name_free_emoji_clustering.local_vectors import (
    EmbeddingArtifact,
    MembershipArtifact,
    MembershipEntry,
    SimilarityArtifact,
    build_local_vectors,
    stable_softmax,
)


class LocalVectorTests(unittest.TestCase):
    def test_stable_softmax_normalizes_weights(self) -> None:
        weights = stable_softmax([1.0, 0.5, 0.0], tau_nn=0.15)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_small_cluster_falls_back_to_global_vector(self) -> None:
        membership = MembershipArtifact(
            path=Path("membership.csv"),
            entries=(MembershipEntry("🙂", "cluster_00", 1.0, "onehot", 10),),
            score=1,
            membership_column="membership_raw",
            rationale="test",
        )
        embeddings = EmbeddingArtifact(
            path=None,
            emojis=("🙂",),
            vectors=np.array([[1.0, 0.0]], dtype=np.float32),
            vector_name="test",
            source_mode="test",
            score=1,
            rationale="test",
        )

        rows, neighbors, summary = build_local_vectors(
            membership=membership,
            embeddings=embeddings,
            similarity=None,
            max_k=5,
            tau_nn=0.15,
            min_smoothing_cluster_size=3,
        )

        self.assertTrue(rows[0].smoothing_fallback)
        self.assertTrue(np.allclose(rows[0].global_vector, rows[0].local_vector))
        self.assertEqual(neighbors[0].neighbor_emoji, "🙂")
        self.assertEqual(summary.too_small_clusters, ("cluster_00",))

    def test_local_smoothing_uses_cluster_neighbors(self) -> None:
        membership = MembershipArtifact(
            path=Path("membership.csv"),
            entries=(
                MembershipEntry("a", "cluster_00", 1.0, "onehot", 10),
                MembershipEntry("b", "cluster_00", 1.0, "onehot", 9),
                MembershipEntry("c", "cluster_00", 1.0, "onehot", 8),
            ),
            score=1,
            membership_column="membership_raw",
            rationale="test",
        )
        embeddings = EmbeddingArtifact(
            path=None,
            emojis=("a", "b", "c"),
            vectors=np.array(
                [
                    [1.0, 0.0],
                    [0.5, 0.5],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            vector_name="test",
            source_mode="test",
            score=1,
            rationale="test",
        )
        similarity = SimilarityArtifact(
            path=None,
            emojis=("a", "b", "c"),
            matrix=np.array(
                [
                    [1.0, 0.9, 0.1],
                    [0.9, 1.0, 0.8],
                    [0.1, 0.8, 1.0],
                ],
                dtype=np.float32,
            ),
            source_mode="test",
            score=1,
            rationale="test",
        )

        rows, neighbors, summary = build_local_vectors(
            membership=membership,
            embeddings=embeddings,
            similarity=similarity,
            max_k=2,
            tau_nn=0.15,
            min_smoothing_cluster_size=3,
        )

        row_by_emoji = {row.emoji: row for row in rows}
        self.assertFalse(row_by_emoji["a"].smoothing_fallback)
        self.assertEqual(row_by_emoji["a"].neighbor_emojis, ("a", "b"))
        self.assertEqual(summary.average_neighbors, 2.0)
        neighbor_pairs = {(row.emoji, row.neighbor_emoji) for row in neighbors}
        self.assertIn(("a", "a"), neighbor_pairs)
        self.assertIn(("a", "b"), neighbor_pairs)


if __name__ == "__main__":
    unittest.main()
