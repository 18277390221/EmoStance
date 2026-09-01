from __future__ import annotations

import unittest

import numpy as np

from name_free_emoji_clustering.clustering import kmeans, ordered_cluster_labels
from name_free_emoji_clustering.graphing import build_symmetric_knn_edges
from name_free_emoji_clustering.similarity import normalize_similarity_for_fusion
from name_free_emoji_clustering.soft_labels import (
    compute_entropy,
    effective_nonzero_from_entropy,
    normalize_weights,
)


class PureFunctionTests(unittest.TestCase):
    def test_normalize_weights(self) -> None:
        normalized = normalize_weights({"a": 2.0, "b": 1.0, "c": 0.0})
        self.assertAlmostEqual(sum(normalized.values()), 1.0)
        self.assertEqual(list(normalized), ["a", "b"])

    def test_entropy_effective_nonzero(self) -> None:
        entropy = compute_entropy([0.5, 0.5])
        self.assertAlmostEqual(entropy, 1.0)
        self.assertAlmostEqual(effective_nonzero_from_entropy(entropy), 2.0)

    def test_normalize_similarity_constant_off_diagonal(self) -> None:
        matrix = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
        normalized = normalize_similarity_for_fusion(matrix)
        self.assertTrue(np.allclose(np.diag(normalized), 1.0))
        self.assertAlmostEqual(float(normalized[0, 1]), 0.0)

    def test_knn_edges_are_symmetric_union(self) -> None:
        labels = ["a", "b", "c"]
        affinity = np.array(
            [
                [1.0, 0.9, 0.1],
                [0.9, 1.0, 0.8],
                [0.1, 0.8, 1.0],
            ],
            dtype=np.float32,
        )
        edges = build_symmetric_knn_edges(labels, affinity, k=1)
        pairs = {(edge["source_emoji"], edge["target_emoji"]) for edge in edges}
        self.assertEqual(pairs, {("a", "b"), ("b", "c")})

    def test_ordered_cluster_labels_are_anonymous(self) -> None:
        assignments, communities = ordered_cluster_labels([["b"], ["a", "c"]])
        self.assertEqual(assignments["a"], "cluster_00")
        self.assertEqual(assignments["b"], "cluster_01")
        self.assertEqual(communities[0], ["a", "c"])

    def test_kmeans_returns_requested_number_of_labels(self) -> None:
        points = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)
        labels = kmeans(points, k=2, seed=1, n_init=2)
        self.assertEqual(set(labels.tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
