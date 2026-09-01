from pathlib import Path

from latent_stance_control.prepare_data import build_examples


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_prepare_data() -> None:
    examples, meta = build_examples(
        [ROOT / "examples" / "tiny_annotations.jsonl"],
        ROOT / "examples" / "tiny_clusters.json",
    )
    assert len(examples) == 2
    assert {row["split"] for row in examples} == {"train", "dev"}
    assert meta["num_clusters"] == 3
    assert meta["vector_dim"] == 3
    assert all(abs(sum(row["target_cluster"]) - 1.0) < 1e-6 for row in examples)
