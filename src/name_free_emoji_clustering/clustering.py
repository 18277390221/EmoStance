from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np


@dataclass
class ClusterSolution:
    method: str
    parameter: str
    algorithm_used: str
    assignments: dict[str, str]
    communities: list[list[str]]
    cluster_count: int
    singleton_count: int
    max_cluster_size: int
    min_cluster_size: int
    modularity: float | None
    output_path: Path


def ordered_cluster_labels(communities: Iterable[Iterable[str]]) -> tuple[dict[str, str], list[list[str]]]:
    ordered = [
        sorted([str(item) for item in community])
        for community in communities
        if list(community)
    ]
    ordered.sort(key=lambda community: (-len(community), community[0]))
    assignments: dict[str, str] = {}
    named_communities: list[list[str]] = []
    for idx, community in enumerate(ordered):
        cluster_id = f"cluster_{idx:02d}"
        for emoji in community:
            assignments[emoji] = cluster_id
        named_communities.append(community)
    return assignments, named_communities


def modularity_or_none(graph: nx.Graph, communities: list[list[str]], resolution: float = 1.0) -> float | None:
    if graph.number_of_edges() == 0:
        return None
    try:
        return float(
            nx.algorithms.community.modularity(
                graph,
                [set(community) for community in communities],
                weight="weight",
                resolution=resolution,
            )
        )
    except Exception:
        return None


def write_assignments(path: Path, assignments: dict[str, str], observed_counts: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["emoji", "cluster_id", "observed_count"])
        writer.writeheader()
        for emoji, cluster_id in sorted(
            assignments.items(),
            key=lambda item: (item[1], -observed_counts.get(item[0], 0), item[0]),
        ):
            writer.writerow(
                {
                    "emoji": emoji,
                    "cluster_id": cluster_id,
                    "observed_count": observed_counts.get(emoji, 0),
                }
            )


def make_solution(
    method: str,
    parameter: str,
    algorithm_used: str,
    communities: Iterable[Iterable[str]],
    graph: nx.Graph,
    observed_counts: dict[str, int],
    output_path: Path,
    modularity_resolution: float = 1.0,
) -> ClusterSolution:
    assignments, ordered = ordered_cluster_labels(communities)
    sizes = [len(community) for community in ordered]
    modularity = modularity_or_none(graph, ordered, resolution=modularity_resolution)
    write_assignments(output_path, assignments, observed_counts)
    return ClusterSolution(
        method=method,
        parameter=parameter,
        algorithm_used=algorithm_used,
        assignments=assignments,
        communities=ordered,
        cluster_count=len(ordered),
        singleton_count=sum(1 for size in sizes if size == 1),
        max_cluster_size=max(sizes) if sizes else 0,
        min_cluster_size=min(sizes) if sizes else 0,
        modularity=modularity,
        output_path=output_path,
    )


def run_leiden_or_fallback(
    graph: nx.Graph,
    labels: list[str],
    observed_counts: dict[str, int],
    resolutions: list[float],
    output_dir: Path,
    seed: int = 13,
) -> list[ClusterSolution]:
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions: list[ClusterSolution] = []

    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        index = {label: idx for idx, label in enumerate(labels)}
        edges = [(index[u], index[v]) for u, v in graph.edges()]
        weights = [float(graph[u][v].get("weight", 1.0)) for u, v in graph.edges()]
        ig_graph = ig.Graph(n=len(labels), edges=edges, directed=False)
        ig_graph.vs["name"] = labels
        ig_graph.es["weight"] = weights
        for resolution in resolutions:
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=resolution,
                seed=seed,
            )
            communities = [[labels[idx] for idx in community] for community in partition]
            output_path = output_dir / f"leiden_resolution_{str(resolution).replace('.', '_')}.csv"
            solutions.append(
                make_solution(
                    method="leiden",
                    parameter=f"resolution={resolution}",
                    algorithm_used="leidenalg.RBConfigurationVertexPartition",
                    communities=communities,
                    graph=graph,
                    observed_counts=observed_counts,
                    output_path=output_path,
                    modularity_resolution=resolution,
                )
            )
        return solutions
    except ImportError:
        pass

    for resolution in resolutions:
        if hasattr(nx.algorithms.community, "louvain_communities"):
            communities = nx.algorithms.community.louvain_communities(
                graph,
                weight="weight",
                resolution=resolution,
                seed=seed,
            )
            algorithm_used = "networkx_louvain_fallback_leiden_unavailable"
        else:
            communities = nx.algorithms.community.greedy_modularity_communities(
                graph,
                weight="weight",
                resolution=resolution,
            )
            algorithm_used = "networkx_greedy_modularity_fallback_leiden_unavailable"

        output_path = output_dir / f"leiden_resolution_{str(resolution).replace('.', '_')}.csv"
        solutions.append(
            make_solution(
                method="leiden",
                parameter=f"resolution={resolution}",
                algorithm_used=algorithm_used,
                communities=communities,
                graph=graph,
                observed_counts=observed_counts,
                output_path=output_path,
                modularity_resolution=resolution,
            )
        )
    return solutions


def kmeans(points: np.ndarray, k: int, seed: int = 13, max_iter: int = 100, n_init: int = 8) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive.")
    if points.shape[0] < k:
        raise ValueError("k cannot exceed the number of points.")

    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_inertia = math.inf
    n = points.shape[0]

    for _ in range(n_init):
        first = int(rng.integers(0, n))
        centers = [points[first]]
        while len(centers) < k:
            center_matrix = np.vstack(centers)
            distances = ((points[:, None, :] - center_matrix[None, :, :]) ** 2).sum(axis=2)
            closest = distances.min(axis=1)
            total = float(closest.sum())
            if total <= 0:
                next_idx = int(rng.integers(0, n))
            else:
                probs = closest / total
                next_idx = int(rng.choice(n, p=probs))
            centers.append(points[next_idx])
        center_array = np.vstack(centers)
        labels = np.zeros(n, dtype=np.int32)

        for _iteration in range(max_iter):
            distances = ((points[:, None, :] - center_array[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster_idx in range(k):
                mask = labels == cluster_idx
                if mask.any():
                    center_array[cluster_idx] = points[mask].mean(axis=0)
                else:
                    farthest = int(distances.min(axis=1).argmax())
                    center_array[cluster_idx] = points[farthest]

        final_distances = ((points - center_array[labels]) ** 2).sum(axis=1)
        inertia = float(final_distances.sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    if best_labels is None:
        raise ValueError("K-means failed to produce labels.")
    return best_labels


def spectral_embedding(affinity: np.ndarray, k: int) -> np.ndarray:
    matrix = np.maximum(affinity.astype(np.float64, copy=True), 0.0)
    np.fill_diagonal(matrix, 0.0)
    degrees = matrix.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
    normalized = inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    top = eigenvectors[:, np.argsort(eigenvalues)[-k:]]
    norms = np.linalg.norm(top, axis=1, keepdims=True)
    return np.divide(top, norms, out=np.zeros_like(top), where=norms > 0)


def run_spectral_scan(
    affinity: np.ndarray,
    labels: list[str],
    graph: nx.Graph,
    observed_counts: dict[str, int],
    k_values: list[int],
    output_dir: Path,
    seed: int = 13,
) -> list[ClusterSolution]:
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions: list[ClusterSolution] = []
    for k_value in k_values:
        if k_value > len(labels):
            continue
        embedding = spectral_embedding(affinity, k_value)
        label_ids = kmeans(embedding, k_value, seed=seed)
        communities = [
            [labels[idx] for idx in range(len(labels)) if int(label_ids[idx]) == cluster_idx]
            for cluster_idx in range(k_value)
        ]
        output_path = output_dir / f"spectral_k_{k_value:02d}.csv"
        solutions.append(
            make_solution(
                method="spectral",
                parameter=f"K={k_value}",
                algorithm_used="numpy_normalized_spectral_clustering",
                communities=communities,
                graph=graph,
                observed_counts=observed_counts,
                output_path=output_path,
            )
        )
    return solutions


def write_solution_summary(path: Path, solutions: list[ClusterSolution]) -> None:
    fieldnames = [
        "method",
        "parameter",
        "algorithm_used",
        "cluster_count",
        "singleton_count",
        "max_cluster_size",
        "min_cluster_size",
        "modularity",
        "assignment_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for solution in solutions:
            writer.writerow(
                {
                    "method": solution.method,
                    "parameter": solution.parameter,
                    "algorithm_used": solution.algorithm_used,
                    "cluster_count": solution.cluster_count,
                    "singleton_count": solution.singleton_count,
                    "max_cluster_size": solution.max_cluster_size,
                    "min_cluster_size": solution.min_cluster_size,
                    "modularity": (
                        f"{solution.modularity:.10f}" if solution.modularity is not None else ""
                    ),
                    "assignment_file": str(solution.output_path),
                }
            )


def choose_candidate_solutions(solutions: list[ClusterSolution], observed_count: int) -> list[ClusterSolution]:
    leiden_solutions = [solution for solution in solutions if solution.method == "leiden"]
    pool = leiden_solutions or solutions
    target_cluster_count = min(12, max(6, round(math.sqrt(max(observed_count, 1)))))

    def score(solution: ClusterSolution) -> tuple[float, float, int]:
        singleton_rate = solution.singleton_count / max(observed_count, 1)
        max_share = solution.max_cluster_size / max(observed_count, 1)
        granularity_penalty = abs(solution.cluster_count - target_cluster_count) / target_cluster_count
        coarse_penalty = max(0, 5 - solution.cluster_count) * 0.25
        oversized_penalty = max(0.0, max_share - 0.35) * 2.0
        modularity_bonus = solution.modularity if solution.modularity is not None else 0.0
        total = (
            0.5 * modularity_bonus
            - 2.0 * singleton_rate
            - granularity_penalty
            - coarse_penalty
            - oversized_penalty
        )
        return (total, -(singleton_rate), -solution.cluster_count)

    return sorted(pool, key=score, reverse=True)[:3]
