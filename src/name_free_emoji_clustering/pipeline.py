from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .clustering import (
    ClusterSolution,
    choose_candidate_solutions,
    run_leiden_or_fallback,
    run_spectral_scan,
    write_solution_summary,
)
from .discovery import DiscoveryResult, discover_inputs, write_discovery_report
from .embeddings import build_or_load_embeddings
from .frequency import load_frequency_pool
from .graphing import (
    build_symmetric_knn_edges,
    degree_rows,
    graph_from_edges,
    write_degree_csv,
    write_edge_csv,
)
from .reporting import write_analysis_report, write_html_visualization
from .similarity import (
    build_centroids,
    build_confusion_similarity,
    build_soft_label_matrix,
    context_similarity,
    fuse_similarities,
    top_neighbors,
    write_centroids,
    write_neighbor_csv,
    write_square_matrix_csv,
)
from .soft_labels import (
    CanonicalData,
    build_from_existing_soft_csv,
    build_from_raw_annotations,
    filter_canonical_data_by_splits,
    split_counts,
    write_canonical_outputs,
)


@dataclass
class PipelineConfig:
    root: Path
    output_dir: Path
    tau: float = 50.0
    knn_k: int = 8
    embedding_dim: int = 256
    lambda_ctx: float = 0.65
    lambda_conf: float = 0.35
    leiden_resolutions: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
    spectral_k_min: int = 6
    spectral_k_max: int = 16
    force_embeddings: bool = False
    cluster_splits: tuple[str, ...] = ("train",)


@dataclass
class PipelineResult:
    output_dir: Path
    analysis_report_path: Path
    html_path: Path
    solutions: list[ClusterSolution]


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "tables": output_dir / "tables",
        "embeddings": output_dir / "embeddings",
        "matrices": output_dir / "matrices",
        "neighbors": output_dir / "neighbors",
        "graph": output_dir / "graph",
        "clusters": output_dir / "clusters",
        "reports": output_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def build_canonical_data(config: PipelineConfig, discovery: DiscoveryResult) -> CanonicalData:
    if discovery.soft_label_table is not None:
        try:
            return build_from_existing_soft_csv(discovery.soft_label_table.paths[0])
        except Exception as exc:
            if discovery.raw_multimodel_annotations is None:
                raise ValueError(
                    "Discovered a soft-label table but it could not be validated, and no "
                    f"raw multi-model fallback was found. Validation error: {exc}"
                ) from exc

    if discovery.raw_multimodel_annotations is not None:
        return build_from_raw_annotations(discovery.raw_multimodel_annotations)

    raise ValueError(
        "Could not find required inputs. Expected either a standardized utterance-emoji "
        "soft-label table or raw multi-model annotations with dialogue_id, turn_id, role, "
        "text/utterance, emoji maps, confidence maps, model identifiers, and split."
    )


def write_pipeline_metadata(config: PipelineConfig, data: CanonicalData, path: Path) -> None:
    metadata = {
        "root": str(config.root),
        "tau": config.tau,
        "knn_k": config.knn_k,
        "embedding_dim": config.embedding_dim,
        "lambda_ctx": config.lambda_ctx,
        "lambda_conf": config.lambda_conf,
        "leiden_resolutions": list(config.leiden_resolutions),
        "spectral_k_values": list(range(config.spectral_k_min, config.spectral_k_max + 1)),
        "cluster_splits": list(config.cluster_splits),
        "split_counts_after_filtering": split_counts(data),
        "source_mode": data.source_mode,
        "source_path": data.source_path,
        "hard_constraints": {
            "emoji_names_used": False,
            "aliases_used": False,
            "unicode_descriptions_used": False,
            "external_emoji_lexicon_used": False,
            "transition_features_used": False,
            "zero_frequency_emojis_clustered": False,
        },
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    dirs = ensure_dirs(config.output_dir)
    discovery = discover_inputs(config.root)
    write_discovery_report(discovery, dirs["reports"] / "input_discovery.md")
    data = build_canonical_data(config, discovery)
    data = filter_canonical_data_by_splits(data, config.cluster_splits)
    frequency_summary = load_frequency_pool(
        discovery.emoji_frequency_tables,
        data.emojis,
        dirs["reports"] / "observed_frequency_validation.json",
    )
    write_pipeline_metadata(config, data, dirs["reports"] / "pipeline_metadata.json")
    if frequency_summary.clustered_zero_frequency_emojis:
        raise ValueError(
            "Frequency-pool validation found zero-frequency emojis in cluster set: "
            f"{frequency_summary.clustered_zero_frequency_emojis[:20]}"
        )

    canonical_table = dirs["tables"] / "canonical_soft_label_table.csv"
    validation_json = dirs["tables"] / "soft_label_validation.json"
    validation_report = dirs["reports"] / "soft_label_validation.md"
    write_canonical_outputs(data, canonical_table, validation_json, validation_report)

    utterance_ids = [utterance.utterance_id for utterance in data.utterances]
    texts = [utterance.text for utterance in data.utterances]
    embedding_cache = build_or_load_embeddings(
        utterance_ids=utterance_ids,
        texts=texts,
        output_npz_path=dirs["embeddings"] / "utterance_embeddings.npz",
        metadata_path=dirs["embeddings"] / "utterance_embeddings_metadata.json",
        dim=config.embedding_dim,
        force=config.force_embeddings,
    )
    embeddings = embedding_cache.embeddings

    q, _, _ = build_soft_label_matrix(data)
    raw_centroids, shrunk_centroids, effective_counts, alpha = build_centroids(
        data=data,
        q=q,
        embeddings=embeddings,
        tau=config.tau,
    )
    write_centroids(
        path=dirs["tables"] / "emoji_centroids.csv",
        npz_path=dirs["embeddings"] / "emoji_centroids.npz",
        emojis=data.emojis,
        observed_counts=data.observed_counts,
        effective_counts=effective_counts,
        alpha=alpha,
        raw_centroids=raw_centroids,
        shrunk_centroids=shrunk_centroids,
    )

    raw_confusion, confusion_similarity = build_confusion_similarity(q)
    context_sim = context_similarity(shrunk_centroids)
    fused, context_norm, confusion_norm = fuse_similarities(
        context_similarity=context_sim,
        confusion_similarity=confusion_similarity,
        lambda_ctx=config.lambda_ctx,
        lambda_conf=config.lambda_conf,
    )

    write_square_matrix_csv(dirs["matrices"] / "soft_confusion_raw.csv", data.emojis, raw_confusion)
    write_square_matrix_csv(
        dirs["matrices"] / "soft_confusion_similarity.csv",
        data.emojis,
        confusion_similarity,
    )
    write_square_matrix_csv(
        dirs["matrices"] / "soft_confusion_similarity_for_fusion.csv",
        data.emojis,
        confusion_norm,
    )
    write_square_matrix_csv(dirs["matrices"] / "context_similarity.csv", data.emojis, context_sim)
    write_square_matrix_csv(
        dirs["matrices"] / "context_similarity_for_fusion.csv",
        data.emojis,
        context_norm,
    )
    write_square_matrix_csv(dirs["matrices"] / "fused_affinity.csv", data.emojis, fused)
    np.savez_compressed(
        dirs["matrices"] / "similarity_matrices.npz",
        emojis=np.array(data.emojis),
        raw_confusion=raw_confusion,
        confusion_similarity=confusion_similarity,
        context_similarity=context_sim,
        fused_affinity=fused,
    )

    write_neighbor_csv(
        dirs["neighbors"] / "top_confusion_neighbors.csv",
        top_neighbors(data.emojis, confusion_similarity, data.observed_counts),
    )
    write_neighbor_csv(
        dirs["neighbors"] / "top_context_neighbors.csv",
        top_neighbors(data.emojis, context_sim, data.observed_counts),
    )
    fused_neighbor_path = dirs["neighbors"] / "top_fused_neighbors.csv"
    write_neighbor_csv(
        fused_neighbor_path,
        top_neighbors(data.emojis, fused, data.observed_counts),
    )

    edges = build_symmetric_knn_edges(data.emojis, fused, config.knn_k)
    write_edge_csv(dirs["graph"] / "emoji_knn_edges.csv", edges)
    graph = graph_from_edges(data.emojis, edges)
    write_degree_csv(dirs["graph"] / "emoji_knn_degrees.csv", degree_rows(graph, data.emojis))

    leiden_solutions = run_leiden_or_fallback(
        graph=graph,
        labels=data.emojis,
        observed_counts=data.observed_counts,
        resolutions=list(config.leiden_resolutions),
        output_dir=dirs["clusters"],
    )
    spectral_solutions = run_spectral_scan(
        affinity=fused,
        labels=data.emojis,
        graph=graph,
        observed_counts=data.observed_counts,
        k_values=list(range(config.spectral_k_min, config.spectral_k_max + 1)),
        output_dir=dirs["clusters"],
    )
    solutions = [*leiden_solutions, *spectral_solutions]
    write_solution_summary(dirs["clusters"] / "clustering_solution_summary.csv", solutions)

    candidates = choose_candidate_solutions(solutions, observed_count=len(data.emojis))
    analysis_report_path = dirs["reports"] / "analysis_summary.md"
    write_analysis_report(
        path=analysis_report_path,
        observed_count=len(data.emojis),
        fused_neighbor_path=fused_neighbor_path,
        leiden_solutions=leiden_solutions,
        all_solutions=solutions,
        candidate_solutions=candidates,
        graph=graph,
    )

    selected = candidates[0] if candidates else solutions[0]
    html_path = config.output_dir / "cluster_visualization.html"
    write_html_visualization(
        path=html_path,
        labels=data.emojis,
        affinity=fused,
        graph=graph,
        solution=selected,
        observed_counts=data.observed_counts,
    )
    return PipelineResult(
        output_dir=config.output_dir,
        analysis_report_path=analysis_report_path,
        html_path=html_path,
        solutions=solutions,
    )
