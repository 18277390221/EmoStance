from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_lib import (  # noqa: E402
    EXPERIMENT_ID,
    SCHEMA_VERSION,
    distribution_overlap,
    entropy_natural,
    jsd_base2,
    leave_one_out_metrics,
    public_payload_leaks,
    q_from_weighted_votes,
    select_stratified_samples,
    tie_aware_agreement,
    validate_export_payload,
)


def test_qe_confidence_aggregation() -> None:
    qe = q_from_weighted_votes([("🙂", 1), ("😢", 3), ("🙂", 2)], ["🙂", "😢"])
    assert qe == {"🙂": 0.5, "😢": 0.5}


def test_entropy_normalized_four_votes() -> None:
    assert entropy_natural([1.0, 0.0]) == 0.0
    assert math.isclose(entropy_natural([0.25, 0.25, 0.25, 0.25]) / math.log(4), 1.0)


def test_jsd_base2_properties() -> None:
    assert jsd_base2([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert math.isclose(jsd_base2([1, 0], [0, 1]), 1.0)
    assert math.isclose(jsd_base2([0.8, 0.2], [0.1, 0.9]), jsd_base2([0.1, 0.9], [0.8, 0.2]))
    assert 0.0 <= jsd_base2([0.8, 0.2], [0.1, 0.9]) <= 1.0


def test_distribution_overlap() -> None:
    assert math.isclose(distribution_overlap([0.7, 0.3], [0.2, 0.8]), 0.5)


def test_tie_aware_top1_agreement() -> None:
    assert tie_aware_agreement([0.5, 0.5, 0.0], [0.0, 1.0, 0.0]) == 1
    assert tie_aware_agreement([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == 0


def test_leave_one_out_human_distributions() -> None:
    emojis = ["🙂", "😢"]
    matrix = np.eye(2)
    answers = [
        {"selected_emoji": "🙂", "confidence": 5},
        {"selected_emoji": "🙂", "confidence": 3},
        {"selected_emoji": "😢", "confidence": 2},
    ]
    metrics = leave_one_out_metrics(answers, emojis, matrix)
    assert set(metrics) == {
        "human_loo_emoji_jsd",
        "human_loo_region_jsd",
        "human_loo_emoji_overlap",
        "human_loo_region_overlap",
    }
    assert 0 <= metrics["human_loo_emoji_jsd"] <= 1


def test_export_schema_validation() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "annotator_id": "annotator_1",
        "questionnaire_id": "questionnaire_1",
        "sample_count": 2,
        "completed_count": 2,
        "created_at": "2026-01-01T00:00:00Z",
        "exported_at": "2026-01-01T01:00:00Z",
        "answers": [
            {"sample_id": "s1", "selected_emoji": "🙂", "unicode": "U+1F642", "confidence": 5, "item_order": 1, "elapsed_ms": 10},
            {"sample_id": "s2", "selected_emoji": "😢", "unicode": "U+1F622", "confidence": 1, "item_order": 2, "elapsed_ms": 20},
        ],
    }
    errors = validate_export_payload(
        payload,
        expected_annotator_id="annotator_1",
        expected_sample_ids={"s1", "s2"},
        emoji_set={"🙂", "😢"},
    )
    assert errors == []


def test_deterministic_sampling() -> None:
    by_stratum = {}
    for stratum_idx, stratum in enumerate(("low", "medium", "high")):
        rows = []
        for idx in range(60):
            rows.append(
                {
                    "dialogue_id": f"{stratum}_{idx}",
                    "target_turn_index": 1,
                    "disagreement_stratum": stratum,
                    "top1_latent_region": f"cluster_{idx % 9:02d}",
                    "role_transition": "A->B" if idx % 2 else "B->A",
                    "top1_llm_emoji": ["🙂", "😢", "😮"][idx % 3],
                    "normalized_entropy": stratum_idx + idx / 1000,
                }
            )
        by_stratum[stratum] = rows
    first = select_stratified_samples(by_stratum, 120, 42)
    second = select_stratified_samples(by_stratum, 120, 42)
    assert [(x["dialogue_id"], x["sample_id"]) for x in first] == [(x["dialogue_id"], x["sample_id"]) for x in second]
    assert len({x["dialogue_id"] for x in first}) == 120


def test_public_private_payload_leakage_check() -> None:
    clean = [
        {
            "sample_id": "s1",
            "situation": "A situation",
            "context_turns": [],
            "target_turn": "Target",
            "target_turn_index": 1,
            "target_role": "B",
        }
    ]
    assert public_payload_leaks(clean) == []
    leaky = [dict(clean[0], llm_qE={"🙂": 1.0})]
    assert public_payload_leaks(leaky)
