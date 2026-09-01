from __future__ import annotations

import math
import unittest

from name_free_emoji_clustering.cluster_transitions import (
    WorkingUtterance,
    build_conditional_rows,
    build_transition_counts,
    project_utterances_to_clusters,
    reliability_weight,
)


class ClusterTransitionTests(unittest.TestCase):
    def test_project_utterance_soft_emojis_to_clusters(self) -> None:
        utterance = WorkingUtterance(
            utterance_id="train|d1|0",
            dialogue_id="d1",
            turn_id="0",
            turn_index="0",
            split="train",
            role="A",
            emoji_probs={"🙂": 0.5, "😊": 0.5},
            direct_mean_confidence=5.0,
        )
        membership = {
            "🙂": {"cluster_00": 1.0},
            "😊": {"cluster_00": 0.25, "cluster_01": 0.75},
        }

        projection = project_utterances_to_clusters(
            [utterance],
            membership,
            ("cluster_00", "cluster_01"),
        )

        projected = projection.utterances[0]
        self.assertAlmostEqual(projected.cluster_probs["cluster_00"], 0.625)
        self.assertAlmostEqual(projected.cluster_probs["cluster_01"], 0.375)
        self.assertEqual(projected.top1_cluster, "cluster_00")

    def test_reliability_uses_confidence_and_entropy(self) -> None:
        value = reliability_weight(math.log(2), 4, 2.5)
        self.assertAlmostEqual(value, 0.25)

    def test_a2b_transition_counts_use_outer_product_and_reliability(self) -> None:
        source = WorkingUtterance(
            utterance_id="train|d1|0",
            dialogue_id="d1",
            turn_id="0",
            turn_index="0",
            split="train",
            role="A",
            emoji_probs={"🙂": 1.0},
            direct_mean_confidence=5.0,
        )
        target = WorkingUtterance(
            utterance_id="train|d1|1",
            dialogue_id="d1",
            turn_id="1",
            turn_index="1",
            split="train",
            role="B",
            emoji_probs={"😊": 1.0},
            direct_mean_confidence=5.0,
        )
        projection = project_utterances_to_clusters(
            [source, target],
            {"🙂": {"cluster_00": 1.0}, "😊": {"cluster_01": 1.0}},
            ("cluster_00", "cluster_01"),
        )

        transitions = build_transition_counts(projection.utterances)

        self.assertEqual(transitions.transition_instances["A2B"], 1)
        self.assertAlmostEqual(
            transitions.weighted_counts["A2B"][("cluster_00", "cluster_01")],
            1.0,
        )

    def test_conditional_rows_are_smoothed_per_source(self) -> None:
        rows = build_conditional_rows(
            "A2B",
            {("cluster_00", "cluster_01"): 2.0},
            {("cluster_00", "cluster_01"): 2.0},
            ("cluster_00", "cluster_01"),
            alpha=0.1,
        )
        source_zero_rows = [row for row in rows if row.source_cluster == "cluster_00"]

        self.assertAlmostEqual(sum(row.conditional_prob for row in source_zero_rows), 1.0)
        self.assertGreater(
            next(row for row in source_zero_rows if row.target_cluster == "cluster_01").conditional_prob,
            0.9,
        )


if __name__ == "__main__":
    unittest.main()
