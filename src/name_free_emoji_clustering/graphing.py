from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import numpy as np


def build_symmetric_knn_edges(
    labels: list[str],
    affinity: np.ndarray,
    k: int,
) -> list[dict[str, object]]:
    if k <= 0:
        raise ValueError("k must be positive.")
    edge_weights: dict[tuple[int, int], float] = {}
    n = len(labels)
    for i in range(n):
        order = sorted(
            (j for j in range(n) if j != i and np.isfinite(affinity[i, j])),
            key=lambda j: (-float(affinity[i, j]), labels[j]),
        )[: min(k, max(n - 1, 0))]
        for j in order:
            a, b = sorted((i, j))
            weight = float(affinity[i, j])
            edge_weights[(a, b)] = max(edge_weights.get((a, b), 0.0), weight)

    rows: list[dict[str, object]] = []
    for (i, j), weight in sorted(edge_weights.items(), key=lambda item: (item[0][0], item[0][1])):
        rows.append(
            {
                "source_emoji": labels[i],
                "target_emoji": labels[j],
                "weight": f"{weight:.10f}",
            }
        )
    return rows


def graph_from_edges(labels: list[str], edges: list[dict[str, object]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    for edge in edges:
        graph.add_edge(
            str(edge["source_emoji"]),
            str(edge["target_emoji"]),
            weight=float(edge["weight"]),
        )
    return graph


def degree_rows(graph: nx.Graph, labels: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in labels:
        weighted_degree = sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(label, data=True))
        rows.append(
            {
                "emoji": label,
                "degree": int(graph.degree(label)),
                "weighted_degree": f"{weighted_degree:.10f}",
            }
        )
    return rows


def write_edge_csv(path: Path, edges: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_emoji", "target_emoji", "weight"])
        writer.writeheader()
        writer.writerows(edges)


def write_degree_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["emoji", "degree", "weighted_degree"])
        writer.writeheader()
        writer.writerows(rows)
