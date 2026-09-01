from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from .clustering import ClusterSolution


def read_neighbor_rows(path: Path, source_limit: int = 10, neighbor_limit: int = 5) -> dict[str, list[dict[str, str]]]:
    rows_by_source: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row["source_emoji"]
            rows_by_source.setdefault(source, [])
            if len(rows_by_source[source]) < neighbor_limit:
                rows_by_source[source].append(row)
            if len(rows_by_source) >= source_limit and source not in rows_by_source:
                break
    return dict(list(rows_by_source.items())[:source_limit])


def write_analysis_report(
    path: Path,
    observed_count: int,
    fused_neighbor_path: Path,
    leiden_solutions: list[ClusterSolution],
    all_solutions: list[ClusterSolution],
    candidate_solutions: list[ClusterSolution],
    graph: nx.Graph,
) -> None:
    neighbors = read_neighbor_rows(fused_neighbor_path, source_limit=10, neighbor_limit=5)
    singleton_rates = [
        solution.singleton_count / max(observed_count, 1)
        for solution in leiden_solutions
    ]
    too_many_singletons = bool(singleton_rates and max(singleton_rates) > 0.25)
    isolated_nodes = [node for node in graph.nodes if graph.degree(node) == 0]
    fallback_algorithms = sorted(
        {
            solution.algorithm_used
            for solution in leiden_solutions
            if "fallback" in solution.algorithm_used
        }
    )

    lines = [
        "# Name-free Emoji Clustering Summary",
        "",
        f"1. Observed emojis clustered: `{observed_count}`.",
        "",
        "2. Fused nearest neighbors for 10 frequent emojis:",
        "",
    ]
    for source, rows in neighbors.items():
        neighbor_text = ", ".join(
            f"{row['neighbor_emoji']} ({float(row['similarity']):.3f})" for row in rows
        )
        lines.append(f"- {source}: {neighbor_text}")

    lines.extend(["", "3. Leiden resolution scan:", ""])
    for solution in leiden_solutions:
        lines.append(
            f"- `{solution.parameter}` -> `{solution.cluster_count}` clusters, "
            f"`{solution.singleton_count}` singletons, max size `{solution.max_cluster_size}`."
        )

    lines.extend(
        [
            "",
            f"4. Singleton check: `{'too many' if too_many_singletons else 'not excessive'}` "
            f"(max singleton rate `{max(singleton_rates) if singleton_rates else 0.0:.3f}`).",
            "",
            "5. Obvious anomalies:",
            "",
        ]
    )
    if isolated_nodes:
        lines.append(f"- Isolated graph nodes exist: `{len(isolated_nodes)}`.")
    else:
        lines.append("- No isolated kNN graph nodes.")
    if fallback_algorithms:
        lines.append(
            "- Real Leiden dependencies were unavailable in this environment; "
            f"used `{', '.join(fallback_algorithms)}` for the Leiden-named scan."
        )
    largest = max(all_solutions, key=lambda solution: solution.max_cluster_size, default=None)
    if largest is not None and largest.max_cluster_size / max(observed_count, 1) > 0.6:
        lines.append(
            f"- A large cluster appears in `{largest.method} {largest.parameter}` "
            f"with `{largest.max_cluster_size}` emojis."
        )
    else:
        lines.append("- No solution has a single cluster covering more than 60% of observed emojis.")

    lines.extend(["", "6. Promising candidate solutions for the next step:", ""])
    for solution in candidate_solutions:
        modularity = "NA" if solution.modularity is None else f"{solution.modularity:.4f}"
        lines.append(
            f"- `{solution.method}` `{solution.parameter}`: `{solution.cluster_count}` clusters, "
            f"`{solution.singleton_count}` singletons, modularity `{modularity}`."
        )

    lines.extend(
        [
            "",
            "No emoji names, aliases, Unicode descriptions, external emoji lexicons, or transition "
            "features were used in the clustering features.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def spectral_2d_layout(affinity: np.ndarray) -> np.ndarray:
    if affinity.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if affinity.shape[0] == 1:
        return np.array([[0.5, 0.5]], dtype=np.float32)
    matrix = np.maximum(affinity.astype(np.float64, copy=True), 0.0)
    np.fill_diagonal(matrix, 0.0)
    degrees = matrix.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
    normalized = inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[-3:]
    coords = eigenvectors[:, order[:2]] if len(order) >= 2 else eigenvectors[:, order[-1:]]
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0])])
    mins = coords.min(axis=0)
    spans = coords.max(axis=0) - mins
    coords = np.divide(coords - mins, spans, out=np.zeros_like(coords), where=spans > 1e-12)
    return coords.astype(np.float32)


def cluster_grid_layout(
    labels: list[str],
    solution: ClusterSolution,
    observed_counts: dict[str, int],
) -> dict[str, tuple[float, float]]:
    clusters: dict[str, list[str]] = {}
    for label in labels:
        clusters.setdefault(solution.assignments.get(label, "unassigned"), []).append(label)
    for members in clusters.values():
        members.sort(key=lambda emoji: (-observed_counts.get(emoji, 0), emoji))

    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {}

    cols = math.ceil(math.sqrt(len(cluster_ids)))
    rows = math.ceil(len(cluster_ids) / cols)
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    coords: dict[str, tuple[float, float]] = {}

    for cluster_index, cluster_id in enumerate(cluster_ids):
        col = cluster_index % cols
        row = cluster_index // cols
        center_x = (col + 0.5) * cell_w
        center_y = (row + 0.5) * cell_h
        members = clusters[cluster_id]
        if len(members) == 1:
            coords[members[0]] = (center_x, center_y)
            continue

        coords[members[0]] = (center_x, center_y)
        remaining = members[1:]
        per_ring = 8
        ring_count = max(1, math.ceil(len(remaining) / per_ring))
        max_radius = 0.36 * min(cell_w, cell_h)

        for ring_index in range(ring_count):
            start = ring_index * per_ring
            ring_members = remaining[start : start + per_ring]
            radius = max_radius * (ring_index + 1) / ring_count
            angle_offset = (ring_index % 2) * math.pi / max(len(ring_members), 1)
            for member_index, emoji in enumerate(ring_members):
                angle = angle_offset + 2.0 * math.pi * member_index / len(ring_members)
                coords[emoji] = (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                )
    return coords


def cluster_internal_edges(
    labels: list[str],
    affinity: np.ndarray,
    solution: ClusterSolution,
    observed_counts: dict[str, int],
) -> list[dict[str, object]]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    clusters: dict[str, list[str]] = {}
    for label in labels:
        clusters.setdefault(solution.assignments.get(label, "unassigned"), []).append(label)

    edges: list[dict[str, object]] = []
    for cluster_id, members in sorted(clusters.items()):
        members = sorted(members, key=lambda emoji: (-observed_counts.get(emoji, 0), emoji))
        if len(members) < 2:
            continue

        connected = {members[0]}
        remaining = set(members[1:])
        while remaining:
            best: tuple[float, str, str] | None = None
            for source in connected:
                source_idx = label_index[source]
                for target in remaining:
                    target_idx = label_index[target]
                    weight = float(affinity[source_idx, target_idx])
                    if not math.isfinite(weight):
                        weight = 0.0
                    candidate = (weight, source, target)
                    if best is None or candidate > best:
                        best = candidate
            if best is None:
                source = sorted(connected)[0]
                target = sorted(remaining)[0]
                weight = 0.0
            else:
                weight, source, target = best
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "cluster": cluster_id,
                    "weight": weight,
                }
            )
            connected.add(target)
            remaining.remove(target)
    return edges


def multi_cluster_candidates(
    labels: list[str],
    affinity: np.ndarray,
    solution: ClusterSolution,
) -> dict[str, list[dict[str, object]]]:
    label_index = {label: idx for idx, label in enumerate(labels)}
    clusters: dict[str, list[str]] = {}
    for label in labels:
        clusters.setdefault(solution.assignments.get(label, "unassigned"), []).append(label)

    candidates: dict[str, list[dict[str, object]]] = {}
    for label in labels:
        own_cluster = solution.assignments.get(label, "unassigned")
        source_idx = label_index[label]
        support_by_cluster: list[dict[str, object]] = []

        for cluster_id, members in sorted(clusters.items()):
            member_scores = [
                float(affinity[source_idx, label_index[member]])
                for member in members
                if member != label and np.isfinite(affinity[source_idx, label_index[member]])
            ]
            support = max(member_scores) if member_scores else 0.0
            support_by_cluster.append({"cluster": cluster_id, "support": support})

        own_support = next(
            (
                float(item["support"])
                for item in support_by_cluster
                if item["cluster"] == own_cluster
            ),
            0.0,
        )
        cross_threshold = max(0.70, own_support * 0.85)
        cross_clusters = [
            item
            for item in support_by_cluster
            if item["cluster"] != own_cluster and float(item["support"]) >= cross_threshold
        ]
        if not cross_clusters:
            continue

        selected = [
            {"cluster": own_cluster, "support": own_support},
            *sorted(cross_clusters, key=lambda item: float(item["support"]), reverse=True),
        ]
        candidates[label] = selected[:4]
    return candidates


def write_html_visualization(
    path: Path,
    labels: list[str],
    affinity: np.ndarray,
    graph: nx.Graph,
    solution: ClusterSolution,
    observed_counts: dict[str, int],
) -> None:
    coords = cluster_grid_layout(labels, solution, observed_counts)
    if not coords:
        spectral_coords = spectral_2d_layout(affinity)
        coords = {
            emoji: (float(spectral_coords[idx, 0]), float(spectral_coords[idx, 1]))
            for idx, emoji in enumerate(labels)
        }
    nodes = []
    overlap_candidates = multi_cluster_candidates(labels, affinity, solution)
    for idx, emoji in enumerate(labels):
        x, y = coords[emoji]
        candidate_clusters = overlap_candidates.get(emoji, [])
        nodes.append(
            {
                "emoji": emoji,
                "cluster": solution.assignments.get(emoji, "unassigned"),
                "candidateClusters": candidate_clusters,
                "multiCluster": bool(candidate_clusters),
                "count": observed_counts.get(emoji, 0),
                "x": x,
                "y": y,
            }
        )
    edges = [
        {
            "source": str(u),
            "target": str(v),
            "weight": float(data.get("weight", 1.0)),
        }
        for u, v, data in graph.edges(data=True)
    ]
    internal_edges = cluster_internal_edges(labels, affinity, solution, observed_counts)
    clusters: dict[str, list[dict[str, object]]] = {}
    for node in nodes:
        clusters.setdefault(str(node["cluster"]), []).append(node)
    for members in clusters.values():
        members.sort(key=lambda item: (-int(item["count"]), str(item["emoji"])))

    data_json = json.dumps(
        {"nodes": nodes, "edges": edges, "clusterEdges": internal_edges},
        ensure_ascii=False,
    )
    cluster_sections = []
    for cluster_id in sorted(clusters):
        members = clusters[cluster_id]
        chip_parts = []
        for member in members:
            candidate_clusters = ", ".join(
                str(item["cluster"]) for item in member["candidateClusters"]
            )
            title = f"{member['emoji']} · count {int(member['count'])}"
            if member["multiCluster"]:
                title += f" · candidate clusters: {candidate_clusters}"
            chip_class = "emoji-chip multi-cluster" if member["multiCluster"] else "emoji-chip"
            chip_parts.append(
                f"<span class='{chip_class}' title='{html.escape(title)}'>"
                f"{html.escape(str(member['emoji']))}"
                f"<small>{int(member['count'])}</small></span>"
            )
        chips = "\n".join(chip_parts)
        cluster_sections.append(
            f"<section class='cluster'><h2>{html.escape(cluster_id)} "
            f"<span>{len(members)}</span></h2><div class='chips'>{chips}</div></section>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Name-free Emoji Clusters</title>
<style>
html {{
  height: 100%;
}}
body {{
  margin: 0;
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #202124;
  background: #eef1f5;
}}
header {{
  flex: 0 0 auto;
  padding: 14px 18px 10px;
  background: #ffffff;
  border-bottom: 1px solid #d7dbe2;
}}
h1 {{
  margin: 0 0 6px;
  font-size: 20px;
  letter-spacing: 0;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px 18px;
  color: #4c5666;
  font-size: 13px;
}}
main {{
  flex: 1 1 auto;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(680px, 1fr) minmax(300px, 380px);
  gap: 12px;
  padding: 12px;
}}
#graph {{
  width: 100%;
  height: 100%;
  min-height: 0;
  background: #ffffff;
  border: 1px solid #d7dbe2;
  border-radius: 8px;
}}
.clusters {{
  min-height: 0;
  overflow: auto;
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
  align-content: start;
  padding-right: 2px;
}}
.cluster {{
  background: #ffffff;
  border: 1px solid #d7dbe2;
  border-radius: 8px;
  padding: 9px;
}}
.cluster h2 {{
  margin: 0 0 7px;
  font-size: 13px;
  display: flex;
  justify-content: space-between;
  color: #293241;
}}
.chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}}
.emoji-chip {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  min-height: 26px;
  padding: 2px 6px;
  border-radius: 7px;
  border: 1px solid #dde2ea;
  background: #fbfcfe;
  font-size: 19px;
}}
.emoji-chip small {{
  font-size: 10px;
  color: #5d6878;
}}
.emoji-chip.multi-cluster {{
  border: 2px solid #172033;
  box-shadow: 0 0 0 2px rgba(23, 32, 51, 0.08);
}}
@media (max-width: 900px) {{
  body {{ overflow: auto; }}
  main {{ grid-template-columns: 1fr; }}
  #graph {{ height: 72vh; }}
  .clusters {{ overflow: visible; }}
}}
</style>
</head>
<body>
<header>
<h1>Name-free Emoji Clusters</h1>
<div class="meta">
  <span>Selected solution: <strong>{html.escape(solution.method)} {html.escape(solution.parameter)}</strong></span>
  <span>Algorithm: <strong>{html.escape(solution.algorithm_used)}</strong></span>
  <span>Clusters: <strong>{solution.cluster_count}</strong></span>
  <span>Observed emojis: <strong>{len(labels)}</strong></span>
  <span>Boxed candidates: <strong>{len(overlap_candidates)}</strong></span>
</div>
</header>
<main>
<svg id="graph" role="img" aria-label="Emoji kNN cluster graph"></svg>
<div class="clusters">
{''.join(cluster_sections)}
</div>
</main>
<script>
const data = {data_json};
const svg = document.getElementById("graph");
const palette = ["#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6","#06b6d4","#f97316","#64748b","#14b8a6","#ec4899","#84cc16","#6366f1","#a855f7","#0ea5e9","#d946ef","#22c55e"];
function colorFor(cluster) {{
  const n = Number(cluster.replace("cluster_", ""));
  return palette[(Number.isFinite(n) ? n : 0) % palette.length];
}}
function draw() {{
  const rect = svg.getBoundingClientRect();
  const width = Math.max(rect.width, 320);
  const height = Math.max(rect.height, 320);
  svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  svg.innerHTML = "";
  const byEmoji = new Map(data.nodes.map(node => [node.emoji, node]));
  const radii = new Map(data.nodes.map(node => [node.emoji, 9 + Math.min(13, Math.log10(Math.max(1, node.count)) * 4)]));
  const maxRadius = Math.max(...Array.from(radii.values()));
  const pad = Math.max(44, maxRadius + 20);
  const toX = node => pad + node.x * (width - 2 * pad);
  const toY = node => pad + node.y * (height - 2 * pad);
  const byCluster = new Map();
  for (const node of data.nodes) {{
    if (!byCluster.has(node.cluster)) byCluster.set(node.cluster, []);
    byCluster.get(node.cluster).push(node);
  }}
  for (const [cluster, members] of byCluster.entries()) {{
    const xs = members.map(toX);
    const ys = members.map(toY);
    const rectNode = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rectNode.setAttribute("x", Math.max(8, Math.min(...xs) - maxRadius - 18));
    rectNode.setAttribute("y", Math.max(8, Math.min(...ys) - maxRadius - 18));
    rectNode.setAttribute("width", Math.min(width - 16, Math.max(...xs) - Math.min(...xs) + 2 * maxRadius + 36));
    rectNode.setAttribute("height", Math.min(height - 16, Math.max(...ys) - Math.min(...ys) + 2 * maxRadius + 36));
    rectNode.setAttribute("rx", "16");
    rectNode.setAttribute("fill", colorFor(cluster));
    rectNode.setAttribute("fill-opacity", "0.055");
    rectNode.setAttribute("stroke", colorFor(cluster));
    rectNode.setAttribute("stroke-opacity", "0.18");
    rectNode.setAttribute("stroke-width", "1");
    svg.appendChild(rectNode);
  }}
  for (const edge of data.edges) {{
    const a = byEmoji.get(edge.source);
    const b = byEmoji.get(edge.target);
    if (!a || !b) continue;
    if (a.cluster === b.cluster) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", toX(a));
    line.setAttribute("y1", toY(a));
    line.setAttribute("x2", toX(b));
    line.setAttribute("y2", toY(b));
    line.setAttribute("stroke", "#9aa5b5");
    line.setAttribute("stroke-opacity", String(0.06 + 0.16 * edge.weight));
    line.setAttribute("stroke-width", String(0.4 + 1.0 * edge.weight));
    svg.appendChild(line);
  }}
  for (const edge of data.clusterEdges) {{
    const a = byEmoji.get(edge.source);
    const b = byEmoji.get(edge.target);
    if (!a || !b) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", toX(a));
    line.setAttribute("y1", toY(a));
    line.setAttribute("x2", toX(b));
    line.setAttribute("y2", toY(b));
    line.setAttribute("stroke", colorFor(edge.cluster));
    line.setAttribute("stroke-opacity", String(0.34 + 0.34 * edge.weight));
    line.setAttribute("stroke-width", String(1.1 + 2.0 * edge.weight));
    line.setAttribute("stroke-linecap", "round");
    svg.appendChild(line);
  }}
  for (const node of data.nodes) {{
    const x = toX(node);
    const y = toY(node);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const radius = radii.get(node.emoji);
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", radius);
    circle.setAttribute("fill", colorFor(node.cluster));
    circle.setAttribute("fill-opacity", "0.88");
    circle.setAttribute("stroke", "#ffffff");
    circle.setAttribute("stroke-width", "2");
    if (node.multiCluster) {{
      const box = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      const boxSize = 2 * radius + 13;
      box.setAttribute("x", x - boxSize / 2);
      box.setAttribute("y", y - boxSize / 2);
      box.setAttribute("width", boxSize);
      box.setAttribute("height", boxSize);
      box.setAttribute("rx", "7");
      box.setAttribute("fill", "none");
      box.setAttribute("stroke", "#172033");
      box.setAttribute("stroke-width", "2.5");
      box.setAttribute("stroke-dasharray", "5 3");
      group.appendChild(box);
    }}
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", x);
    text.setAttribute("y", y + 5);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "16");
    text.textContent = node.emoji;
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    const candidateText = node.multiCluster
      ? ` · candidate clusters ${{node.candidateClusters.map(item => `${{item.cluster}}(${{item.support.toFixed(2)}})`).join(", ")}}`
      : "";
    title.textContent = `${{node.emoji}} · ${{node.cluster}} · count ${{node.count}}${{candidateText}}`;
    group.appendChild(title);
    group.appendChild(circle);
    group.appendChild(text);
    svg.appendChild(group);
  }}
}}
window.addEventListener("resize", draw);
draw();
</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
