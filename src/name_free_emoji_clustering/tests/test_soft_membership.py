from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from name_free_emoji_clustering.soft_membership import (
    CandidateCluster,
    EmojiClusterNode,
    build_membership_rows,
    discover_current_cluster_artifact,
    parse_html_artifact,
    validate_membership_rows,
)


class SoftMembershipTests(unittest.TestCase):
    def test_onehot_membership_for_primary_only_node(self) -> None:
        rows = build_membership_rows(
            [
                EmojiClusterNode(
                    emoji="🙂",
                    primary_cluster="cluster_00",
                    multi_cluster=False,
                    candidate_clusters=(),
                    observed_count=5,
                )
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].cluster_id, "cluster_00")
        self.assertEqual(rows[0].membership_raw, 1.0)
        self.assertEqual(rows[0].membership_sharp, 1.0)
        self.assertEqual(rows[0].source_type, "onehot")

    def test_soft_membership_normalizes_candidate_supports(self) -> None:
        rows = build_membership_rows(
            [
                EmojiClusterNode(
                    emoji="🙂",
                    primary_cluster="cluster_00",
                    multi_cluster=True,
                    candidate_clusters=(
                        CandidateCluster("cluster_00", 3.0),
                        CandidateCluster("cluster_01", 1.0),
                    ),
                    observed_count=5,
                )
            ]
        )

        probabilities = {row.cluster_id: row.membership_raw for row in rows}
        self.assertAlmostEqual(probabilities["cluster_00"], 0.75)
        self.assertAlmostEqual(probabilities["cluster_01"], 0.25)
        sharp_probabilities = {row.cluster_id: row.membership_sharp for row in rows}
        self.assertGreater(sharp_probabilities["cluster_00"], probabilities["cluster_00"])
        self.assertTrue(all(row.source_type == "soft_candidate" for row in rows))

    def test_primary_cluster_is_injected_when_missing(self) -> None:
        rows = build_membership_rows(
            [
                EmojiClusterNode(
                    emoji="🙂",
                    primary_cluster="cluster_00",
                    multi_cluster=True,
                    candidate_clusters=(CandidateCluster("cluster_01", 2.0),),
                    observed_count=None,
                )
            ]
        )

        probabilities = {row.cluster_id: row.membership_raw for row in rows}
        self.assertEqual(set(probabilities), {"cluster_00", "cluster_01"})
        self.assertAlmostEqual(probabilities["cluster_00"], 0.5)
        self.assertAlmostEqual(probabilities["cluster_01"], 0.5)
        sharp_probabilities = {row.cluster_id: row.membership_sharp for row in rows}
        self.assertAlmostEqual(sharp_probabilities["cluster_00"], 0.5)
        self.assertAlmostEqual(sharp_probabilities["cluster_01"], 0.5)

    def test_html_payload_parses_node_data(self) -> None:
        html = """
        <script>
        const data = {"nodes": [
          {"emoji": "🙂", "cluster": "cluster_00",
           "candidateClusters": [{"cluster": "cluster_01", "support": 2.0}],
           "multiCluster": true, "count": 7}
        ], "edges": []};
        </script>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cluster.html"
            path.write_text(html, encoding="utf-8")
            artifact = parse_html_artifact(path)

        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(len(artifact.nodes), 1)
        self.assertEqual(artifact.nodes[0].emoji, "🙂")
        self.assertEqual(artifact.nodes[0].candidate_clusters[0].cluster_id, "cluster_01")

    def test_discovery_prefers_html_with_candidate_supports_over_csv_onehot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "assignments.csv"
            csv_path.write_text(
                "emoji,cluster_id,observed_count\n🙂,cluster_00,7\n",
                encoding="utf-8",
            )
            html_path = root / "cluster.html"
            html_path.write_text(
                """
                <script>
                const data = {"nodes": [
                  {"emoji": "🙂", "cluster": "cluster_00",
                   "candidateClusters": [{"cluster": "cluster_01", "support": 2.0}],
                   "multiCluster": true, "count": 7}
                ]};
                </script>
                """,
                encoding="utf-8",
            )

            artifact = discover_current_cluster_artifact(root)

        self.assertEqual(artifact.path.name, "cluster.html")

    def test_validation_reports_soft_counts_and_entropy(self) -> None:
        rows = build_membership_rows(
            [
                EmojiClusterNode("🙂", "cluster_00", False, (), 5),
                EmojiClusterNode(
                    "😊",
                    "cluster_00",
                    True,
                    (
                        CandidateCluster("cluster_00", 1.0),
                        CandidateCluster("cluster_01", 1.0),
                    ),
                    4,
                ),
            ]
        )
        validation = validate_membership_rows(rows)

        self.assertEqual(validation.observed_emoji_count, 2)
        self.assertEqual(validation.onehot_emoji_count, 1)
        self.assertEqual(validation.soft_multi_cluster_emoji_count, 1)
        self.assertEqual(validation.top_ambiguous_raw[0]["emoji"], "😊")
        self.assertEqual(validation.top_ambiguous_sharp[0]["emoji"], "😊")


if __name__ == "__main__":
    unittest.main()
