import unittest

import numpy as np

from ..utterance_cluster_projection import (
    WorkingUtterance,
    build_projection,
    reliability_weight,
    summarize_projection,
    validate_projection,
)


class UtteranceClusterProjectionTests(unittest.TestCase):
    def test_projection_builds_cluster_probs_and_vectors(self) -> None:
        utterances = [
            WorkingUtterance(
                utterance_id="u1",
                dialogue_id="d1",
                turn_id="0",
                turn_index="0",
                split="train",
                role="A",
                emoji_probs={"😀": 0.5, "😢": 0.5},
                direct_mean_confidence=5.0,
            )
        ]
        membership = {
            "😀": {"cluster_00": 0.8, "cluster_01": 0.2},
            "😢": {"cluster_01": 1.0},
        }
        local_vectors = {
            ("😀", "cluster_00"): np.asarray([1.0, 0.0]),
            ("😀", "cluster_01"): np.asarray([0.5, 0.5]),
            ("😢", "cluster_01"): np.asarray([0.0, 1.0]),
        }

        result = build_projection(
            utterances,
            membership,
            local_vectors,
            ("cluster_00", "cluster_01"),
            dimension=2,
            eps=0.0,
            top_contributors=2,
        )
        validate_projection(result)

        projected = result.utterances[0]
        self.assertAlmostEqual(projected.cluster_probs["cluster_00"], 0.4)
        self.assertAlmostEqual(projected.cluster_probs["cluster_01"], 0.6)
        self.assertEqual(projected.top1_cluster, "cluster_01")
        self.assertEqual(projected.nonzero_cluster_count, 2)

        vectors = {
            row.cluster_id: row
            for row in result.continuous_vectors
        }
        np.testing.assert_allclose(vectors["cluster_00"].vector, np.asarray([1.0, 0.0]))
        np.testing.assert_allclose(
            vectors["cluster_01"].vector,
            np.asarray([0.08333333333333333, 0.9166666666666667]),
        )
        self.assertEqual(vectors["cluster_01"].top_contributing_emojis, ("😢", "😀"))
        self.assertAlmostEqual(sum(vectors["cluster_01"].top_contribution_weights), 1.0)

    def test_missing_local_vector_raises(self) -> None:
        utterances = [
            WorkingUtterance(
                utterance_id="u1",
                dialogue_id="d1",
                turn_id="0",
                turn_index="0",
                split="train",
                role="A",
                emoji_probs={"😀": 1.0},
            )
        ]
        membership = {"😀": {"cluster_00": 1.0}}

        with self.assertRaisesRegex(ValueError, "local vectors are missing"):
            build_projection(
                utterances,
                membership,
                {},
                ("cluster_00",),
                dimension=2,
            )

    def test_reliability_and_summary_concentration(self) -> None:
        self.assertAlmostEqual(reliability_weight(0.0, 3, None), 1.0)
        self.assertAlmostEqual(reliability_weight(0.0, 3, 2.5), 0.5)

        utterances = [
            WorkingUtterance(
                utterance_id="a",
                dialogue_id="d",
                turn_id="0",
                turn_index="0",
                split="train",
                role="A",
                emoji_probs={"😀": 0.5, "😢": 0.5},
            ),
            WorkingUtterance(
                utterance_id="b",
                dialogue_id="d",
                turn_id="1",
                turn_index="1",
                split="train",
                role="B",
                emoji_probs={"😀": 1.0},
            ),
        ]
        membership = {
            "😀": {"cluster_00": 1.0},
            "😢": {"cluster_01": 1.0},
        }
        local_vectors = {
            ("😀", "cluster_00"): np.asarray([1.0]),
            ("😢", "cluster_01"): np.asarray([0.0]),
        }

        result = build_projection(
            utterances,
            membership,
            local_vectors,
            ("cluster_00", "cluster_01"),
            dimension=1,
            eps=0.0,
        )
        summary = summarize_projection(result)

        self.assertTrue(summary.b_more_concentrated_than_a)
        self.assertLess(summary.role_average_entropy["B"], summary.role_average_entropy["A"])


if __name__ == "__main__":
    unittest.main()
