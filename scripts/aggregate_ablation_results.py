from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from collections import defaultdict

import numpy as np

from ablation_utils import (
    DATASET_METADATA,
    MAIN_RUN,
    append_command,
    bootstrap_ci,
    cluster_metrics,
    ece,
    environment_snapshot,
    fmt_mean_std,
    fmt_num,
    fuse_log_probs,
    gated_fusion,
    generation_rows_metrics,
    graph_prediction,
    labels,
    latex_escape,
    load_git_commit,
    load_prepared_rows,
    mean_std,
    npz_arrays,
    read_json,
    read_jsonl,
    rel,
    sanitize,
    soft_ce_per_example,
    vector_cosine,
    vector_cosine_per_example,
    vector_mse,
    write_csv,
    write_json,
)


SEEDS = [13, 21, 42]


def result(
    *,
    group: str,
    ablation_id: str,
    method: str,
    split: str = "test",
    deployable: bool | str | None = None,
    seeds: Sequence[int] | None = None,
    metrics: Dict[str, Any] | None = None,
    source_type: str = "reused",
    artifact_path: str | None = None,
    config_verified: bool = True,
    notes: str = "",
    dataset_size: int | None = None,
) -> Dict[str, Any]:
    return sanitize(
        {
            "experiment_group": group,
            "ablation_id": ablation_id,
            "method": method,
            "deployable": deployable,
            "split": split,
            "dataset_size": dataset_size or DATASET_METADATA["test_examples"],
            "seeds": list(seeds or []),
            "run_type": "mean±std" if seeds and len(seeds) > 1 else ("single-run" if seeds else "not_run"),
            "metrics": metrics or {},
            "source": {
                "type": source_type,
                "artifact_path": artifact_path,
                "config_verified": config_verified,
            },
            "notes": notes,
        }
    )


def load_stance_result(run_dir: Path, method: str, ablation_id: str, notes: str) -> Dict[str, Any]:
    arrays = npz_arrays(run_dir / "test_predictions.npz")
    target = cluster_metrics(arrays["gold_target_cluster"], arrays["pred_target_cluster"])
    source = cluster_metrics(arrays["gold_source_cluster"], arrays["pred_source_cluster"])
    metrics = {
        "target_ce": target["soft_ce"],
        "target_jsd": target["jsd"],
        "target_kl": target["kl"],
        "target_acc": target["accuracy"],
        "target_macro_f1": target["macro_f1"],
        "target_ece": target["ece"],
        "target_brier": target["brier"],
        "target_per_cluster_f1": target.get("per_cluster_f1"),
        "target_entropy_confidence_breakdown": target.get("entropy_confidence_breakdown"),
        "source_ce": source["soft_ce"],
        "source_acc": source["accuracy"],
        "source_macro_f1": source["macro_f1"],
        "source_ece": source["ece"],
        "target_vector_cosine": vector_cosine(arrays["gold_target_vector"], arrays["pred_target_vector"]),
        "target_vector_mse": vector_mse(arrays["gold_target_vector"], arrays["pred_target_vector"]),
        "source_vector_cosine": vector_cosine(arrays["gold_source_vector"], arrays["pred_source_vector"]),
        "source_vector_mse": vector_mse(arrays["gold_source_vector"], arrays["pred_source_vector"]),
    }
    config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    return result(
        group="stance_prediction",
        ablation_id=ablation_id,
        method=method,
        seeds=[int(config.get("seed", 13))],
        metrics=metrics,
        artifact_path=rel(run_dir / "test_predictions.npz"),
        notes=notes,
    )


def load_executed_stance_ablation(dir_prefix: str, method: str, ablation_id: str, notes: str) -> Dict[str, Any]:
    run_dirs = sorted((Path("runs/emnlp_ablation_report/stance")).glob(f"{dir_prefix}_seed*"))
    run_dirs = [d for d in run_dirs if (d / "test_predictions.npz").exists()]
    if not run_dirs:
        return result(
            group="stance_prediction",
            ablation_id=ablation_id,
            method=method,
            source_type="missing",
            config_verified=False,
            notes=notes,
        )
    per_seed: Dict[str, List[float]] = defaultdict(list)
    diagnostics_by_seed: List[Dict[str, Any]] = []
    seeds: List[int] = []
    dataset_size = None
    for run_dir in run_dirs:
        arrays = npz_arrays(run_dir / "test_predictions.npz")
        dataset_size = int(arrays["gold_target_cluster"].shape[0])
        target = cluster_metrics(arrays["gold_target_cluster"], arrays["pred_target_cluster"])
        source = cluster_metrics(arrays["gold_source_cluster"], arrays["pred_source_cluster"])
        vals = {
            "target_ce": target["soft_ce"],
            "target_jsd": target["jsd"],
            "target_kl": target["kl"],
            "target_acc": target["accuracy"],
            "target_macro_f1": target["macro_f1"],
            "target_ece": target["ece"],
            "target_brier": target["brier"],
            "source_ce": source["soft_ce"],
            "source_acc": source["accuracy"],
            "source_macro_f1": source["macro_f1"],
            "source_ece": source["ece"],
            "target_vector_cosine": vector_cosine(arrays["gold_target_vector"], arrays["pred_target_vector"]),
            "target_vector_mse": vector_mse(arrays["gold_target_vector"], arrays["pred_target_vector"]),
            "source_vector_cosine": vector_cosine(arrays["gold_source_vector"], arrays["pred_source_vector"]),
            "source_vector_mse": vector_mse(arrays["gold_source_vector"], arrays["pred_source_vector"]),
        }
        for key, value in vals.items():
            per_seed[key].append(float(value))
        config = read_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
        seed = config.get("seed")
        if seed is not None:
            seeds.append(int(seed))
        diagnostics_by_seed.append({
            "seed": int(seed) if seed is not None else None,
            "target_per_cluster_f1": target.get("per_cluster_f1"),
            "target_entropy_confidence_breakdown": target.get("entropy_confidence_breakdown"),
        })
    metrics: Dict[str, Any] = {}
    for key, values in per_seed.items():
        metrics[key] = values[0] if len(values) == 1 else mean_std(values)
    if len(diagnostics_by_seed) == 1:
        metrics.update({k: v for k, v in diagnostics_by_seed[0].items() if k != "seed"})
    else:
        metrics["target_diagnostics_by_seed"] = diagnostics_by_seed
    return result(
        group="stance_prediction",
        ablation_id=ablation_id,
        method=method,
        seeds=seeds,
        metrics=metrics,
        source_type="new",
        artifact_path=", ".join(rel(d / "test_predictions.npz") for d in run_dirs),
        config_verified=True,
        notes=notes + " Recomputed CE/JSD/Brier from saved full-test prediction distributions.",
        dataset_size=dataset_size,
    )


def graph_ablation_results() -> List[Dict[str, Any]]:
    stance_arrays = npz_arrays(MAIN_RUN / "stance_fp32" / "test_predictions.npz")
    rows = load_prepared_rows(MAIN_RUN / "prepared", "test")
    transitions = [str(row.get("transition", "A->B")) for row in rows]
    matrices = read_json(MAIN_RUN / "ablations" / "transition_matrices.json")
    graph = graph_prediction(stance_arrays["pred_source_cluster"], transitions, matrices)
    text = stance_arrays["pred_target_cluster"]
    gold = stance_arrays["gold_target_cluster"]
    ablation_metrics = read_json(MAIN_RUN / "ablations" / "metrics.json")
    beta = float(ablation_metrics["test"]["text_graph_calibrated"].get("beta", 0.0))
    calibrated = gated_fusion(text, graph, beta)
    out: List[Dict[str, Any]] = []
    for ablation_id, method, pred in [
        ("A5a", "graph-only", graph),
        ("A5b", "calibrated graph fusion", calibrated),
    ]:
        m = cluster_metrics(gold, pred)
        out.append(
            result(
                group="stance_prediction",
                ablation_id=ablation_id,
                method=method,
                seeds=[13],
                metrics={
                    "target_ce": m["soft_ce"],
                    "target_jsd": m["jsd"],
                    "target_kl": m["kl"],
                    "target_acc": m["accuracy"],
                    "target_macro_f1": m["macro_f1"],
                    "target_ece": m["ece"],
                    "target_brier": m["brier"],
                    "target_per_cluster_f1": m.get("per_cluster_f1"),
                    "target_entropy_confidence_breakdown": m.get("entropy_confidence_breakdown"),
                    "beta": beta if method == "calibrated graph fusion" else None,
                },
                artifact_path=rel(MAIN_RUN / "ablations" / "metrics.json"),
                notes=(
                    "Computed from text-only stance predictions plus transition graph. "
                    "Calibrated fusion selected beta=0.0 on this artifact, so it matches text-only predictions."
                    if method == "calibrated graph fusion"
                    else "Graph prior computed from predicted source cluster and role transition matrix."
                ),
            )
        )
    return out


def vector_results() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ab = read_json(MAIN_RUN / "ablations" / "metrics.json")["test"]
    text_metrics = read_json(MAIN_RUN / "stance_fp32" / "metrics.json")["test"]
    for ablation_id, method, source_key, note in [
        ("V0", "prototype reconstruction", "target_vector_cluster_prototype", "predicted target-cluster distribution multiplied by train-set cluster prototypes"),
        ("V1", "direct vector regression", "target_vector_text", "direct target-vector regression head from text-only stance predictor"),
    ]:
        m = ab[source_key]
        out.append(
            result(
                group="continuous_vector",
                ablation_id=ablation_id,
                method=method,
                seeds=[13],
                metrics={
                    "target_ce": text_metrics["target_soft_ce"],
                    "target_acc": text_metrics["target_accuracy"],
                    "target_macro_f1": text_metrics["target_macro_f1"],
                    "target_vector_cosine": m["vector_cosine"],
                    "target_vector_mse": m["vector_mse"],
                    "source_vector_cosine": None,
                },
                artifact_path=rel(MAIN_RUN / "ablations" / "metrics.json"),
                notes=note,
            )
        )

    src_ab = read_json(MAIN_RUN / "source_vector_feature_ablation" / "metrics.json")["runs"]
    id_map = {"none": "V2", "direct": "V3", "prototype": "V4"}
    method_map = {
        "none": "no source vector feature",
        "direct": "direct source vector feature",
        "prototype": "prototype source vector feature",
    }
    for run in src_ab:
        name = run["name"]
        m = run["metrics"]["test"]
        out.append(
            result(
                group="continuous_vector",
                ablation_id=id_map[name],
                method=method_map[name],
                seeds=[13],
                metrics={
                    "target_ce": m["target_soft_ce"],
                    "target_acc": m["target_accuracy"],
                    "target_macro_f1": m["target_macro_f1"],
                    "target_vector_cosine": m["target_vector_cosine"],
                    "target_vector_mse": m["target_vector_mse"],
                    "source_vector_cosine": m["source_vector_cosine"],
                    "source_vector_mse": m["source_vector_mse"],
                },
                artifact_path=rel(MAIN_RUN / "source_vector_feature_ablation" / "metrics.json"),
                notes=f"source_vector_feature_mode={run['source_vector_feature_mode']}; single seed.",
            )
        )
    return out


def metric_mean(metric_obj: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = metric_obj.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return {"mean": value, "std": None, "values": []}


def extra_generation_metrics_from_dirs(dirs: Sequence[Path], *, control_type: str | None = None, selection_type: str | None = None) -> Dict[str, Any]:
    per_seed: Dict[str, List[float]] = defaultdict(list)
    total_rows = 0
    for directory in dirs:
        file_name = "selected_test.jsonl" if selection_type else "generations_test.scored.jsonl"
        path = directory / file_name
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if control_type:
            rows = [r for r in rows if r.get("control_type") == control_type]
        if selection_type:
            rows = [r for r in rows if r.get("selection_type") == selection_type]
        total_rows += len(rows)
        m = generation_rows_metrics(rows)
        for key, value in m.items():
            if value is not None:
                per_seed[key].append(float(value))
    output = {key: mean_std(values) for key, values in per_seed.items()}
    output["rows_total"] = total_rows
    return output


def generation_results() -> List[Dict[str, Any]]:
    summary = read_json(MAIN_RUN / "main_results_summary" / "metrics.json")
    rows = {row["method"]: row for row in summary["rows"]}
    control_dirs = [MAIN_RUN / f"generator_control_eval_512_seed{s}" for s in SEEDS]
    role_c7_dirs = [MAIN_RUN / f"generator_control_eval_role_weight025_c7mix050_512_seed{s}" for s in SEEDS]
    rerank_dirs = [MAIN_RUN / f"rerank_c7mix050_512_seed{s}" for s in SEEDS]

    specs = [
        ("G0", "zero control", "zero control", True, control_dirs, {"control_type": "zero"}, "Control baseline; deployable."),
        ("G1", "shuffled control", "shuffled control", False, control_dirs, {"control_type": "shuffled"}, "Negative control; not deployable because control is intentionally mismatched."),
        ("G2", "gold control", "gold control", "no, upper-reference", control_dirs, {"control_type": "gold"}, "Uses gold target stance vector; upper-reference only."),
        ("G3", "predicted control baseline", "predicted control baseline", True, control_dirs, {"control_type": "predicted"}, "Deployable predicted-control baseline."),
        ("G4", "role-aware predicted control", "role+c7mix050", True, role_c7_dirs, {"control_type": "predicted"}, "Deployable role-aware/c7mix050 predicted control without reranking."),
        ("G5", "role-aware predicted control + rerank", "role+c7mix050 + rerank_control", True, rerank_dirs, {"selection_type": "rerank_control"}, "Final deployable automatic-generation diagnostic; 512-example subset."),
        ("G6", "oracle gold selection", "oracle_gold candidate selection", "no, oracle upper bound", rerank_dirs, {"selection_type": "oracle_gold"}, "Oracle candidate selection using gold stance CE; upper bound only."),
    ]
    out: List[Dict[str, Any]] = []
    for ablation_id, method, summary_name, deployable, dirs, filt, note in specs:
        split = rows[summary_name]["splits"]["test"]
        extra = extra_generation_metrics_from_dirs(dirs, **filt)
        metrics = {
            "vs_gold_ce": metric_mean(split, "vs_gold_ce"),
            "vs_gold_jsd": extra.get("vs_gold_jsd", {"mean": None, "std": None, "values": []}),
            "vs_gold_acc": metric_mean(split, "vs_gold_acc"),
            "vs_gold_macro_f1": metric_mean(split, "vs_gold_macro_f1"),
            "vs_control_ce": metric_mean(split, "vs_control_ce"),
            "bert_score_f1": {"mean": None, "std": None, "values": []},
            "distinct_1": extra.get("distinct_1", {"mean": None, "std": None, "values": []}),
            "distinct_2": extra.get("distinct_2", {"mean": None, "std": None, "values": []}),
            "mean_length": metric_mean(split, "mean_words"),
            "generic_rate": extra.get("generic_rate", {"mean": None, "std": None, "values": []}),
            "repetition_rate": extra.get("repetition_rate", {"mean": None, "std": None, "values": []}),
            "too_short_rate": extra.get("too_short_rate", {"mean": None, "std": None, "values": []}),
            "empty_rate": metric_mean(split, "empty_rate"),
        }
        out.append(
            result(
                group="generation_control",
                ablation_id=ablation_id,
                method=method,
                deployable=deployable,
                seeds=SEEDS,
                dataset_size=512,
                metrics=metrics,
                artifact_path=rel(MAIN_RUN / "main_results_summary" / "metrics.json"),
                notes=note + " BERTScore-F1 is not present in existing artifacts and was not fabricated.",
            )
        )
    return out


def rerank_results() -> List[Dict[str, Any]]:
    rerank_dirs = [MAIN_RUN / f"rerank_c7mix050_512_seed{s}" for s in SEEDS]
    summary = read_json(MAIN_RUN / "main_results_summary" / "metrics.json")
    summary_rows = {row["method"]: row for row in summary["rows"]}
    raw_extra = extra_generation_metrics_from_dirs(rerank_dirs, selection_type="raw_first")
    rerank_extra = extra_generation_metrics_from_dirs(rerank_dirs, selection_type="rerank_control")
    oracle_extra = extra_generation_metrics_from_dirs(rerank_dirs, selection_type="oracle_gold")
    specs = [
        ("R0", "first candidate", 1, "role+c7mix050", raw_extra, 1.0),
        ("R1", "rerank by control CE", 4, "role+c7mix050 + rerank_control", rerank_extra, 4.0),
        ("R2", "oracle gold selection", 4, "oracle_gold candidate selection", oracle_extra, 4.0),
    ]
    out: List[Dict[str, Any]] = []
    for ablation_id, method, cand, summary_name, extra, cost in specs:
        split = summary_rows[summary_name]["splits"]["test"] if summary_name in summary_rows else {}
        out.append(
            result(
                group="reranking",
                ablation_id=ablation_id,
                method=method,
                deployable=(method != "oracle gold selection"),
                seeds=SEEDS,
                dataset_size=512,
                metrics={
                    "candidate_count": cand,
                    "vs_gold_ce": metric_mean(split, "vs_gold_ce") if split else extra.get("vs_gold_ce"),
                    "vs_gold_acc": metric_mean(split, "vs_gold_acc") if split else extra.get("vs_gold_acc"),
                    "vs_gold_macro_f1": metric_mean(split, "vs_gold_macro_f1") if split else extra.get("vs_gold_macro_f1"),
                    "mean_length": metric_mean(split, "mean_words") if split else extra.get("mean_length"),
                    "latency_per_example": None,
                    "cost_multiplier": cost,
                },
                artifact_path=rel(MAIN_RUN / "rerank_c7mix050_512_seed13" / "metrics.json"),
                notes="512-example test subset; latency was not logged in artifacts.",
            )
        )
    return out


def paired_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    # Full EmoStance vs text-only.
    full = npz_arrays(MAIN_RUN / "stance_role_aware_weight025" / "test_predictions.npz")
    text = npz_arrays(MAIN_RUN / "stance_fp32" / "test_predictions.npz")
    n = min(len(full["gold_target_cluster"]), len(text["gold_target_cluster"]))
    gold = full["gold_target_cluster"][:n]
    full_pred = full["pred_target_cluster"][:n]
    text_pred = text["pred_target_cluster"][:n]
    stats["Full EmoStance vs text-only DeBERTa"] = {
        "metric_direction": "negative CE/JSD delta favors Full EmoStance; positive acc delta favors Full EmoStance",
        "target_ce_delta_full_minus_text": bootstrap_ci(soft_ce_per_example(gold, full_pred), soft_ce_per_example(gold, text_pred)),
        "target_jsd_delta_full_minus_text": bootstrap_ci(
            # Imported lazily to keep this block explicit.
            __import__("ablation_utils").jsd_per_example(gold, full_pred),
            __import__("ablation_utils").jsd_per_example(gold, text_pred),
        ),
        "target_acc_delta_full_minus_text": bootstrap_ci(
            (labels(gold) == labels(full_pred)).astype(float),
            (labels(gold) == labels(text_pred)).astype(float),
        ),
    }

    for label, path in [
        ("Full EmoStance vs w/o role-aware transition", Path("runs/emnlp_ablation_report/stance/no_role_transition_seed13/test_predictions.npz")),
        ("Full EmoStance vs hard-label training", Path("runs/emnlp_ablation_report/stance/hard_label_seed13/test_predictions.npz")),
    ]:
        if path.exists():
            ablated = npz_arrays(path)
            n2 = min(len(gold), len(ablated["gold_target_cluster"]))
            ablated_pred = ablated["pred_target_cluster"][:n2]
            gold2 = gold[:n2]
            full_pred2 = full_pred[:n2]
            stats[label] = {
                "metric_direction": "negative CE/JSD delta favors Full EmoStance when reported as full minus ablation; positive acc delta favors Full EmoStance",
                "target_ce_delta_full_minus_ablation": bootstrap_ci(soft_ce_per_example(gold2, full_pred2), soft_ce_per_example(gold2, ablated_pred)),
                "target_acc_delta_full_minus_ablation": bootstrap_ci((labels(gold2) == labels(full_pred2)).astype(float), (labels(gold2) == labels(ablated_pred)).astype(float)),
            }
        else:
            stats[label] = {
                "status": "not_computed",
                "reason": "No verified artifact found; not fabricated.",
            }

    # Prototype reconstruction vs direct vector regression.
    prototypes = np.asarray(read_json(MAIN_RUN / "ablations" / "cluster_prototypes.json"), dtype=np.float64)
    proto_vec = text["pred_target_cluster"] @ prototypes
    direct_vec = text["pred_target_vector"]
    gold_vec = text["gold_target_vector"]
    stats["prototype reconstruction vs direct vector regression"] = {
        "metric_direction": "positive cosine delta favors prototype; negative MSE delta favors prototype",
        "cosine_delta_proto_minus_direct": bootstrap_ci(vector_cosine_per_example(gold_vec, proto_vec), vector_cosine_per_example(gold_vec, direct_vec)),
        "mse_delta_proto_minus_direct": bootstrap_ci(
            ((gold_vec - proto_vec) ** 2).mean(axis=-1),
            ((gold_vec - direct_vec) ** 2).mean(axis=-1),
        ),
    }

    # Rerank vs raw first, context-level over 512-example subsets across seeds.
    rerank_ce: List[float] = []
    raw_ce: List[float] = []
    for seed in SEEDS:
        rows = read_jsonl(MAIN_RUN / f"rerank_c7mix050_512_seed{seed}" / "selected_test.jsonl")
        by_type: Dict[str, Dict[int, Mapping[str, Any]]] = {"raw_first": {}, "rerank_control": {}}
        for row in rows:
            st = row.get("selection_type")
            if st in by_type:
                by_type[st][int(row["example_index"])] = row
        for idx in sorted(set(by_type["raw_first"]) & set(by_type["rerank_control"])):
            raw_ce.append(float(by_type["raw_first"][idx]["generated_vs_gold_soft_ce"]))
            rerank_ce.append(float(by_type["rerank_control"][idx]["generated_vs_gold_soft_ce"]))
    stats["role-aware control + rerank vs role-aware control w/o rerank"] = {
        "metric_direction": "negative CE delta favors reranking",
        "vs_gold_ce_delta_rerank_minus_first": bootstrap_ci(rerank_ce, raw_ce),
    }

    # Predicted vs shuffled.
    pred_ce: List[float] = []
    shuf_ce: List[float] = []
    for seed in SEEDS:
        rows = read_jsonl(MAIN_RUN / f"generator_control_eval_512_seed{seed}" / "generations_test.scored.jsonl")
        by_type: Dict[str, Dict[int, Mapping[str, Any]]] = {"predicted": {}, "shuffled": {}}
        for row in rows:
            ct = row.get("control_type")
            if ct in by_type:
                by_type[ct][int(row["example_index"])] = row
        for idx in sorted(set(by_type["predicted"]) & set(by_type["shuffled"])):
            pred_ce.append(float(by_type["predicted"][idx]["generated_vs_gold_soft_ce"]))
            shuf_ce.append(float(by_type["shuffled"][idx]["generated_vs_gold_soft_ce"]))
    stats["predicted control vs shuffled control"] = {
        "metric_direction": "negative CE delta favors predicted control",
        "vs_gold_ce_delta_predicted_minus_shuffled": bootstrap_ci(pred_ce, shuf_ce),
    }
    return sanitize(stats)


def missing_experiments() -> List[Dict[str, str]]:
    stance_dir = Path("runs/emnlp_ablation_report/stance")

    def seed_values(prefix: str) -> List[int]:
        values: List[int] = []
        for path in sorted(stance_dir.glob(f"{prefix}_seed*/test_predictions.npz")):
            try:
                values.append(int(path.parent.name.rsplit("_seed", 1)[1]))
            except Exception:
                pass
        return sorted(set(values))

    def stance_status(prefix: str) -> str:
        values = seed_values(prefix)
        if len(values) >= 3:
            return "found (seeds 13/21/42 mean +/- std)"
        if values:
            return "found (seed " + ",".join(str(v) for v in values) + ")"
        return "missing"

    def stance_next(prefix: str, default: str) -> str:
        values = seed_values(prefix)
        if len(values) >= 3:
            return "complete for automatic table; keep human-eval claims pending"
        if values:
            missing = [s for s in [13, 21, 42] if s not in values]
            return "run missing seeds " + ",".join(str(v) for v in missing) + " for mean +/- std"
        return default

    return [
        {"ablation": "hard-label training", "required": "yes", "current_status": stance_status("hard_label"), "next_action": stance_next("hard_label", "run stance predictor with argmax hard target labels; do not reuse no_focal artifact")},
        {"ablation": "w/o role-aware transition", "required": "yes", "current_status": stance_status("no_role_transition"), "next_action": stance_next("no_role_transition", "train/evaluate role-aware architecture with current role, next role, transition prior removed")},
        {"ablation": "w/o gated transition prior", "required": "yes", "current_status": stance_status("no_gated_prior"), "next_action": stance_next("no_gated_prior", "train/evaluate role-aware features with graph_prior_weight=0 or saved target_text_logits")},
        {"ablation": "one-hot emoji membership", "required": "optional", "current_status": "missing", "next_action": "appendix only if budget allows"},
        {"ablation": "raw vs sharpened membership", "required": "optional", "current_status": "missing", "next_action": "appendix only if budget allows"},
        {"ablation": "full-test rerank sweep", "required": "optional", "current_status": "missing; current rerank is 512-example subset", "next_action": "run if generation budget allows"},
        {"ablation": "human eval for final ablations", "required": "yes", "current_status": "pending", "next_action": "prepare blind annotation batches"},
    ]


def human_eval_pending() -> List[Dict[str, str]]:
    return [
        {"comparison": "Full EmoStance vs w/o stance prefix", "reason": "Tests whether internal stance prefix improves perceived response quality.", "status": "pending"},
        {"comparison": "Full EmoStance vs w/o rerank", "reason": "Tests whether reranking improves not only stance score but also human preference.", "status": "pending"},
        {"comparison": "Full EmoStance vs w/o role-aware transition", "reason": "Tests whether role-aware stance prediction improves listener-appropriate response.", "status": "pending"},
        {"comparison": "Full EmoStance vs direct vector regression", "reason": "Tests whether prototype vector stability translates into better generated replies.", "status": "pending"},
        {"comparison": "Full EmoStance vs emotion-prompt baseline", "reason": "Tests whether latent stance control is better than textual emotion prompting.", "status": "pending"},
    ]


def build_results() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    results.append(
        load_stance_result(
            MAIN_RUN / "stance_role_aware_weight025",
            "Full EmoStance stance predictor",
            "A0",
            "Role-aware predictor with entropy-gated graph prior; single seed, reused from verified artifact.",
        )
    )
    results.append(
        load_stance_result(
            MAIN_RUN / "stance_fp32",
            "text-only DeBERTa",
            "A1",
            "Text-only stance predictor; prompt uses situation and dialogue context only.",
        )
    )
    results.append(
        load_executed_stance_ablation(
            "no_role_transition",
            "w/o role-aware transition",
            "A2",
            "New run: role/current-role/next-role/transition embeddings are zeroed and graph prior is disabled.",
        )
    )
    results.append(
        load_executed_stance_ablation(
            "no_gated_prior",
            "w/o gated transition prior",
            "A3",
            "New run: role features are kept but graph_prior_weight=0 removes the gated log-prior.",
        )
    )
    results.append(
        load_executed_stance_ablation(
            "hard_label",
            "hard-label training",
            "A4",
            "New run: source and target CE losses use argmax hard labels instead of soft weak distributions.",
        )
    )
    results.extend(graph_ablation_results())
    results.extend(vector_results())
    results.extend(generation_results())
    results.extend(rerank_results())

    metadata = {
        "date": __import__("datetime").date.today().isoformat(),
        "git_commit": load_git_commit(),
        "dataset": "EMOJIDIALOGUE",
        "test_examples": DATASET_METADATA["test_examples"],
        "train_examples": DATASET_METADATA["train_examples"],
        "dev_examples": DATASET_METADATA["dev_examples"],
        "observed_emoji": DATASET_METADATA["observed_emoji"],
        "stance_clusters": DATASET_METADATA["stance_clusters"],
        "stance_vector_dim": DATASET_METADATA["stance_vector_dim"],
        "input_restriction": "Deployable systems use only situation, dialogue history up to current turn, and speaker-role markers at inference time.",
    }
    return sanitize(
        {
            "metadata": metadata,
            "results": results,
            "statistical_tests": paired_stats(),
            "missing_experiments": missing_experiments(),
            "human_eval_pending": human_eval_pending(),
            "metric_notes": {
                "generation": "Automatic generation metrics are stance-consistency diagnostics, not final human-quality evidence.",
                "bert_score": "BERTScore-F1 was requested but is absent from existing artifacts and no local BERTScore dependency/model was available in this aggregation.",
                "single_seed": "Stance/vector predictor results reused here are single-run artifacts unless otherwise indicated.",
            },
        }
    )


def flatten_for_csv(results: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        row: Dict[str, Any] = {
            "experiment_group": item.get("experiment_group"),
            "ablation_id": item.get("ablation_id"),
            "method": item.get("method"),
            "deployable": item.get("deployable"),
            "split": item.get("split"),
            "dataset_size": item.get("dataset_size"),
            "seeds": ",".join(str(s) for s in item.get("seeds", [])),
            "run_type": item.get("run_type"),
            "source_type": item.get("source", {}).get("type"),
            "artifact_path": item.get("source", {}).get("artifact_path"),
            "notes": item.get("notes"),
        }
        for key, value in (item.get("metrics") or {}).items():
            if isinstance(value, Mapping):
                row[f"{key}_mean"] = value.get("mean")
                row[f"{key}_std"] = value.get("std")
            else:
                row[key] = value
        rows.append(sanitize(row))
    return rows


def write_latex_tables(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for item in results:
        groups.setdefault(str(item.get("experiment_group")), []).append(item)

    lines: List[str] = ["% Auto-generated by scripts/aggregate_ablation_results.py"]
    stance = groups.get("stance_prediction", [])
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & CE$\downarrow$ & JSD$\downarrow$ & Acc$\uparrow$ & F1$\uparrow$ & ECE$\downarrow$ & Brier$\downarrow$ \\")
    lines.append(r"\midrule")
    for item in stance:
        m = item.get("metrics") or {}
        lines.append(
            f"{latex_escape(item.get('method'))} & {fmt_num(m.get('target_ce'))} & {fmt_num(m.get('target_jsd'))} & {fmt_num(m.get('target_acc'))} & {fmt_num(m.get('target_macro_f1'))} & {fmt_num(m.get('target_ece'))} & {fmt_num(m.get('target_brier'))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    gen = groups.get("generation_control", [])
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & CE$\downarrow$ & JSD$\downarrow$ & Acc$\uparrow$ & F1$\uparrow$ & Len \\")
    lines.append(r"\midrule")
    for item in gen:
        m = item.get("metrics") or {}
        lines.append(
            f"{latex_escape(item.get('method'))} & {fmt_mean_std(m.get('vs_gold_ce'))} & {fmt_mean_std(m.get('vs_gold_jsd'))} & {fmt_mean_std(m.get('vs_gold_acc'))} & {fmt_mean_std(m.get('vs_gold_macro_f1'))} & {fmt_mean_std(m.get('mean_length'))} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing_md(path: Path, missing: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Missing or Incomplete Ablations", ""]
    lines.append("| Ablation | Required? | Current status | Next action |")
    lines.append("|---|---:|---|---|")
    for item in missing:
        lines.append(f"| {item['ablation']} | {item['required']} | {item['current_status']} | {item['next_action']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_human_pending_md(path: Path, pending: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Human Evaluation Pending",
        "",
        "Automatic generation metrics are stance-consistency diagnostics. Human evaluation is still required for emotional appropriateness, empathy, naturalness, and perceived response quality.",
        "",
        "| Comparison | Why human evaluation is needed | Status |",
        "|---|---|---|",
    ]
    for item in pending:
        lines.append(f"| {item['comparison']} | {item['reason']} | {item['status']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate reused and newly computed ablation results.")
    parser.add_argument("--inventory", default="runs/emnlp_ablation_report/result_inventory.json")
    parser.add_argument("--new_results", default="runs/emnlp_ablation_report")
    parser.add_argument("--out_dir", default="runs/emnlp_ablation_report")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_command(out_dir, f"python scripts/aggregate_ablation_results.py --inventory {args.inventory} --new_results {args.new_results} --out_dir {args.out_dir}")

    payload = build_results()
    write_json(out_dir / "environment.json", sanitize(environment_snapshot()))
    write_json(out_dir / "ablation_results.json", payload)
    write_csv(out_dir / "ablation_results.csv", flatten_for_csv(payload["results"]))
    write_latex_tables(out_dir / "ablation_tables.tex", payload["results"])
    write_missing_md(out_dir / "missing_experiments.md", payload["missing_experiments"])
    write_human_pending_md(out_dir / "human_eval_pending.md", payload["human_eval_pending"])


if __name__ == "__main__":
    main()
