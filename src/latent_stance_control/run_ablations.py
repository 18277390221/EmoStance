from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data import load_prepared_split, read_json, write_json
from .metrics import evaluate_cluster_predictions, evaluate_vectors, normalize_prob
from .operators import ridge_affine_operator
from .transitions import build_transition_matrices, graph_prior


def row_array(rows: List[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float32)


def load_stance_predictions(stance_dir: Path, split: str, rows: List[dict], allow_gold_fallback: bool = False) -> Dict[str, np.ndarray]:
    path = stance_dir / f"{split}_predictions.npz"
    if path.exists():
        with np.load(path) as data:
            result = {key: data[key] for key in data.files}
        bad = [key for key, value in result.items() if key.startswith("pred_") and not np.isfinite(value).all()]
        if bad:
            raise ValueError(
                f"Non-finite predictions found in {path}: {bad}. "
                "Retrain the stance predictor after the fp32 fix, then rerun ablations."
            )
        return result
    if not allow_gold_fallback:
        raise FileNotFoundError(f"Missing stance predictions: {path}. Run train_stance_predictor first, or pass --allow-gold-fallback for a smoke-test upper-bound run.")
    return {
        "gold_source_cluster": row_array(rows, "source_cluster"),
        "gold_target_cluster": row_array(rows, "target_cluster"),
        "gold_source_vector": row_array(rows, "source_vector"),
        "gold_target_vector": row_array(rows, "target_vector"),
        "pred_source_cluster": row_array(rows, "source_cluster"),
        "pred_target_cluster": row_array(rows, "target_cluster"),
        "pred_source_vector": row_array(rows, "source_vector"),
        "pred_target_vector": row_array(rows, "target_vector"),
    }


def fuse_log_probs(text_prob: np.ndarray, graph_prob: np.ndarray, beta: float) -> np.ndarray:
    logits = np.log(normalize_prob(text_prob)) + beta * np.log(normalize_prob(graph_prob))
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def gated_fusion(text_prob: np.ndarray, graph_prob: np.ndarray, beta: float) -> np.ndarray:
    entropy = -(normalize_prob(text_prob) * np.log(normalize_prob(text_prob))).sum(axis=-1)
    gate = np.clip(entropy / np.log(text_prob.shape[-1]), 0.0, 1.0)[:, None]
    return fuse_log_probs(text_prob, graph_prob, beta * gate)


def cluster_prototypes(rows: List[dict], num_clusters: int, vector_dim: int) -> np.ndarray:
    sums = np.zeros((num_clusters, vector_dim), dtype=np.float64)
    weights = np.zeros(num_clusters, dtype=np.float64)
    for row in rows:
        q = np.asarray(row["target_cluster"], dtype=np.float64)
        z = np.asarray(row["target_vector"], dtype=np.float64)
        sums += q[:, None] * z[None, :]
        weights += q
    global_mean = row_array(rows, "target_vector").mean(axis=0)
    prototypes = sums / np.maximum(weights[:, None], 1e-12)
    prototypes[weights <= 1e-12] = global_mean
    return prototypes.astype(np.float32)


def fit_role_affine(rows: List[dict], vector_dim: int) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    operators: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    global_matrix, global_bias = ridge_affine_operator(row_array(rows, "source_vector"), row_array(rows, "target_vector"))
    for transition in sorted({row.get("transition", "A->B") for row in rows}):
        subset = [row for row in rows if row.get("transition", "A->B") == transition]
        if len(subset) < max(4, vector_dim // 2):
            operators[transition] = (global_matrix, global_bias)
        else:
            operators[transition] = ridge_affine_operator(row_array(subset, "source_vector"), row_array(subset, "target_vector"))
    operators["__global__"] = (global_matrix, global_bias)
    return operators


def apply_role_affine(source_vectors: np.ndarray, rows: List[dict], operators: Dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    outputs = []
    global_op = operators["__global__"]
    for idx, row in enumerate(rows):
        matrix, bias = operators.get(row.get("transition", "A->B"), global_op)
        outputs.append(matrix @ source_vectors[idx] + bias)
    return np.stack(outputs).astype(np.float32)


def evaluate_split(rows: List[dict], predictions: Dict[str, np.ndarray], matrices: Dict[str, np.ndarray], prototypes: np.ndarray, role_ops: Dict[str, tuple[np.ndarray, np.ndarray]], beta_grid: List[float]) -> Dict[str, Dict[str, float]]:
    gold = row_array(rows, "target_cluster")
    text_target = predictions["pred_target_cluster"]
    graph = np.stack([graph_prior(predictions["pred_source_cluster"][i], rows[i].get("transition", "A->B"), matrices) for i in range(len(rows))])
    results: Dict[str, Dict[str, float]] = {
        "text_only": evaluate_cluster_predictions(gold, text_target),
        "graph_only": evaluate_cluster_predictions(gold, graph),
    }
    best_poe = text_target
    best_poe_metric = evaluate_cluster_predictions(gold, text_target)["soft_ce"]
    best_poe_beta = 0.0
    best_cal = text_target
    best_cal_metric = best_poe_metric
    best_cal_beta = 0.0
    for beta in beta_grid:
        poe = fuse_log_probs(text_target, graph, beta)
        cal = gated_fusion(text_target, graph, beta)
        poe_metric = evaluate_cluster_predictions(gold, poe)["soft_ce"]
        cal_metric = evaluate_cluster_predictions(gold, cal)["soft_ce"]
        if np.isfinite(poe_metric) and poe_metric < best_poe_metric:
            best_poe_metric, best_poe, best_poe_beta = poe_metric, poe, beta
        if np.isfinite(cal_metric) and cal_metric < best_cal_metric:
            best_cal_metric, best_cal, best_cal_beta = cal_metric, cal, beta
    results["text_graph_poe"] = evaluate_cluster_predictions(gold, best_poe)
    results["text_graph_poe"]["beta"] = float(best_poe_beta)
    results["text_graph_calibrated"] = evaluate_cluster_predictions(gold, best_cal)
    results["text_graph_calibrated"]["beta"] = float(best_cal_beta)
    results["target_vector_text"] = evaluate_vectors(predictions["gold_target_vector"], predictions["pred_target_vector"])
    results["target_vector_cluster_prototype"] = evaluate_vectors(predictions["gold_target_vector"], predictions["pred_target_cluster"] @ prototypes)
    results["target_vector_graph_prototype"] = evaluate_vectors(predictions["gold_target_vector"], graph @ prototypes)
    results["target_vector_role_affine_operator"] = evaluate_vectors(predictions["gold_target_vector"], apply_role_affine(predictions["pred_source_vector"], rows, role_ops))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stance-level ablations.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--stance-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--betas", default="-1.0,-0.5,0.0,0.25,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--allow-gold-fallback", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_json(Path(args.prepared) / "meta.json")
    train_rows = load_prepared_split(args.prepared, "train")
    matrices = build_transition_matrices(train_rows, meta["num_clusters"], alpha=args.alpha)
    prototypes = cluster_prototypes(train_rows, meta["num_clusters"], meta["vector_dim"])
    role_ops = fit_role_affine(train_rows, meta["vector_dim"])
    write_json(out / "transition_matrices.json", {k: v.tolist() for k, v in matrices.items()})
    write_json(out / "cluster_prototypes.json", prototypes.tolist())
    write_json(out / "role_affine_operators.json", {k: {"matrix": v[0].tolist(), "bias": v[1].tolist()} for k, v in role_ops.items()})

    beta_grid = [float(x) for x in args.betas.split(",") if x.strip()]
    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    stance_dir = Path(args.stance_dir)
    for split in ("dev", "test"):
        rows = load_prepared_split(args.prepared, split)
        if not rows:
            continue
        preds = load_stance_predictions(stance_dir, split, rows, allow_gold_fallback=args.allow_gold_fallback)
        all_metrics[split] = evaluate_split(rows, preds, matrices, prototypes, role_ops, beta_grid)
    write_json(out / "metrics.json", all_metrics)


if __name__ == "__main__":
    main()
