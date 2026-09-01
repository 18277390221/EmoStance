import unittest

import numpy as np

from ..transition_operators import (
    LocalVectorArtifact,
    UtteranceRecord,
    WeightedSample,
    build_conditional_rows,
    build_transition_samples,
    fit_operator_from_samples,
    output_distribution_for_cluster,
    validate_conditional_rows,
)


class TransitionOperatorTests(unittest.TestCase):
    def test_weighted_transition_counts_and_conditionals(self) -> None:
        utterances = [
            UtteranceRecord(
                utterance_id="u0",
                dialogue_id="d",
                turn_id="0",
                turn_index="0",
                split="train",
                role="A",
                cluster_probs={"cluster_00": 0.6, "cluster_01": 0.4},
                reliability_weight=0.25,
            ),
            UtteranceRecord(
                utterance_id="u1",
                dialogue_id="d",
                turn_id="1",
                turn_index="1",
                split="train",
                role="B",
                cluster_probs={"cluster_01": 1.0},
                reliability_weight=1.0,
            ),
        ]
        transitions, weighted_counts, unweighted_counts, _ = build_transition_samples(utterances)

        self.assertEqual(len(transitions["A2B"]), 1)
        self.assertAlmostEqual(weighted_counts["A2B"][("cluster_00", "cluster_01")], 0.3)
        self.assertAlmostEqual(weighted_counts["A2B"][("cluster_01", "cluster_01")], 0.2)
        self.assertAlmostEqual(unweighted_counts["A2B"][("cluster_00", "cluster_01")], 0.6)

        rows = build_conditional_rows(
            "A2B",
            weighted_counts["A2B"],
            unweighted_counts["A2B"],
            ("cluster_00", "cluster_01"),
            alpha=0.1,
        )
        validate_conditional_rows(rows, ("cluster_00", "cluster_01"))
        by_pair = {(row.source_cluster, row.target_cluster): row for row in rows}
        self.assertAlmostEqual(by_pair[("cluster_00", "cluster_01")].conditional_prob, 0.8)
        self.assertGreater(by_pair[("cluster_00", "cluster_01")].lift, 0.0)

    def test_weighted_ridge_recovers_affine_map(self) -> None:
        utterances = []
        for index, source_value in enumerate([0.0, 1.0, 2.0, 3.0]):
            target_value = 2.0 * source_value + 1.0
            utterances.append(
                UtteranceRecord(
                    utterance_id=f"s{index}",
                    dialogue_id="d",
                    turn_id=str(index * 2),
                    turn_index=str(index * 2),
                    split="train",
                    role="A",
                    vectors={"cluster_00": np.asarray([source_value])},
                )
            )
            utterances.append(
                UtteranceRecord(
                    utterance_id=f"t{index}",
                    dialogue_id="d",
                    turn_id=str(index * 2 + 1),
                    turn_index=str(index * 2 + 1),
                    split="train",
                    role="B",
                    vectors={"cluster_01": np.asarray([target_value])},
                )
            )
        samples = [
            WeightedSample(source_index=index * 2, target_index=index * 2 + 1, weight=1.0)
            for index in range(4)
        ]

        fit = fit_operator_from_samples(
            "A2B",
            "cluster_00",
            "cluster_01",
            tuple(utterances),
            samples,
            ridge_lambda=1e-8,
            status="learned_pair_operator",
            backoff_type="none",
        )

        self.assertAlmostEqual(fit.operator_matrix[0, 0], 2.0, places=5)
        self.assertAlmostEqual(fit.bias_vector[0], 1.0, places=5)
        self.assertLess(fit.weighted_train_mse, 1e-8)

    def test_output_distribution_remains_soft_within_cluster(self) -> None:
        local_vectors = LocalVectorArtifact(
            path=None,  # type: ignore[arg-type]
            vectors_by_cluster={
                "cluster_00": {
                    "😀": np.asarray([1.0, 0.0]),
                    "🙂": np.asarray([0.8, 0.2]),
                    "😢": np.asarray([0.0, 1.0]),
                }
            },
            dimension=2,
            score=0,
            rationale="test",
        )

        emojis, weights, affect_vector = output_distribution_for_cluster(
            "cluster_00",
            np.asarray([1.0, 0.0]),
            local_vectors,
            tau_out=0.5,
            top_k=2,
        )

        self.assertEqual(len(emojis), 2)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertFalse(np.allclose(affect_vector, np.asarray([1.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
