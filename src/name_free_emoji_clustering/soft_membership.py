from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
CLUSTER_ID_RE = re.compile(r"^cluster_\d+$")
HTML_DATA_MARKERS = (
    "const data =",
    "let data =",
    "var data =",
    "window.__CLUSTER_DATA__ =",
    "window.clusterData =",
)
DISCOVERY_PEEK_BYTES = 131_072
VALIDATION_TOLERANCE = 1e-8
DEFAULT_TEMPERATURE = 0.7


@dataclass(frozen=True)
class CandidateCluster:
    cluster_id: str
    support: float


@dataclass(frozen=True)
class EmojiClusterNode:
    emoji: str
    primary_cluster: str
    multi_cluster: bool
    candidate_clusters: tuple[CandidateCluster, ...]
    observed_count: int | None


@dataclass(frozen=True)
class ClusterArtifact:
    path: Path
    kind: str
    nodes: tuple[EmojiClusterNode, ...]
    score: int
    rationale: str


@dataclass(frozen=True)
class MembershipRow:
    emoji: str
    cluster_id: str
    membership_raw: float
    membership_sharp: float
    source_type: str
    observed_count: int | None
    support_score: float | None


@dataclass(frozen=True)
class MembershipValidation:
    observed_emoji_count: int
    onehot_emoji_count: int
    soft_multi_cluster_emoji_count: int
    average_clusters_per_emoji: float
    max_raw_sum_error: float
    max_sharp_sum_error: float
    top_ambiguous_raw: tuple[dict[str, object], ...]
    top_ambiguous_sharp: tuple[dict[str, object], ...]


def iter_files(root: Path, suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def cluster_sort_key(cluster_id: str) -> tuple[int, str]:
    match = re.match(r"^cluster_(\d+)$", cluster_id)
    if match:
        return int(match.group(1)), cluster_id
    return math.inf, cluster_id


def require_anonymous_cluster_id(cluster_id: Any, path: Path) -> str:
    value = str(cluster_id)
    if not CLUSTER_ID_RE.match(value):
        raise ValueError(
            f"Cluster artifact {path} contains non-anonymous cluster id {value!r}; "
            "expected ids like cluster_00."
        )
    return value


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "multi", "soft_candidate"}


def parse_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def get_first(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def parse_candidate_item(item: Any, path: Path) -> CandidateCluster | None:
    if isinstance(item, dict):
        cluster_value = get_first(item, ("cluster", "cluster_id", "clusterId"))
        support_value = get_first(item, ("support", "support_score", "score", "weight", "prob"))
        if cluster_value is None:
            return None
        support = parse_float_or_none(support_value)
        if support is None:
            return None
        return CandidateCluster(
            cluster_id=require_anonymous_cluster_id(cluster_value, path),
            support=support,
        )

    if isinstance(item, (list, tuple)) and len(item) >= 2:
        support = parse_float_or_none(item[1])
        if support is None:
            return None
        return CandidateCluster(
            cluster_id=require_anonymous_cluster_id(item[0], path),
            support=support,
        )

    return None


def parse_candidate_clusters(value: Any, path: Path) -> tuple[CandidateCluster, ...]:
    if value in (None, ""):
        return ()

    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            items: list[CandidateCluster] = []
            for part in re.split(r"[;,]", text):
                if not part.strip():
                    continue
                if ":" not in part:
                    continue
                cluster_text, support_text = part.split(":", 1)
                support = parse_float_or_none(support_text)
                if support is None:
                    continue
                items.append(
                    CandidateCluster(
                        cluster_id=require_anonymous_cluster_id(cluster_text.strip(), path),
                        support=support,
                    )
                )
            return tuple(items)

    if isinstance(parsed, dict):
        if "cluster" in parsed or "cluster_id" in parsed:
            item = parse_candidate_item(parsed, path)
            return (item,) if item is not None else ()
        items = []
        for cluster_id, support_value in parsed.items():
            support = parse_float_or_none(support_value)
            if support is None:
                continue
            items.append(
                CandidateCluster(
                    cluster_id=require_anonymous_cluster_id(cluster_id, path),
                    support=support,
                )
            )
        return tuple(items)

    if isinstance(parsed, list):
        items = [
            candidate
            for candidate in (parse_candidate_item(item, path) for item in parsed)
            if candidate is not None
        ]
        return tuple(items)

    return ()


def row_to_node(row: dict[str, Any], path: Path) -> EmojiClusterNode | None:
    emoji = get_first(row, ("emoji", "label", "id"))
    cluster = get_first(row, ("cluster", "cluster_id", "clusterId", "primary_cluster"))
    if emoji is None or cluster is None:
        return None

    candidates = parse_candidate_clusters(
        get_first(row, ("candidateClusters", "candidate_clusters", "candidateClusterScores")),
        path,
    )
    observed_count = parse_int_or_none(
        get_first(row, ("count", "observed_count", "observedCount", "frequency"))
    )
    multi_cluster = parse_bool(get_first(row, ("multiCluster", "multi_cluster", "boxed_candidate")))
    return EmojiClusterNode(
        emoji=str(emoji),
        primary_cluster=require_anonymous_cluster_id(cluster, path),
        multi_cluster=multi_cluster or bool(candidates),
        candidate_clusters=candidates,
        observed_count=observed_count,
    )


def score_artifact(kind: str, nodes: tuple[EmojiClusterNode, ...]) -> tuple[int, str]:
    candidate_count = sum(1 for node in nodes if node.candidate_clusters)
    observed_count_count = sum(1 for node in nodes if node.observed_count is not None)
    soft_bonus = 1_000 if candidate_count else 0
    kind_base = {"json": 300, "csv": 250, "html": 200}.get(kind, 0)
    score = kind_base + soft_bonus + min(len(nodes), 500) + candidate_count * 4 + observed_count_count
    rationale = (
        f"{kind.upper()} artifact with {len(nodes)} emoji nodes, "
        f"{candidate_count} nodes carrying candidate-cluster supports, "
        f"and {observed_count_count} observed counts."
    )
    return score, rationale


def artifact_from_rows(path: Path, kind: str, rows: Iterable[dict[str, Any]]) -> ClusterArtifact | None:
    nodes = [node for node in (row_to_node(row, path) for row in rows) if node is not None]
    if not nodes:
        return None
    seen: set[str] = set()
    deduped: list[EmojiClusterNode] = []
    for node in nodes:
        if node.emoji in seen:
            continue
        seen.add(node.emoji)
        deduped.append(node)
    node_tuple = tuple(deduped)
    score, rationale = score_artifact(kind, node_tuple)
    return ClusterArtifact(path=path, kind=kind, nodes=node_tuple, score=score, rationale=rationale)


def read_csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    except (OSError, UnicodeDecodeError):
        return []


def parse_csv_artifact(path: Path) -> ClusterArtifact | None:
    header = read_csv_header(path)
    if {"membership_prob", "membership_raw", "membership_sharp"}.intersection(header):
        return None
    if "emoji" not in header or not {"cluster", "cluster_id", "clusterId"}.intersection(header):
        return None

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return artifact_from_rows(path, "csv", rows)


def looks_like_cluster_json(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")[:DISCOVERY_PEEK_BYTES]
    except OSError:
        return False
    lowered = text.lower()
    return "emoji" in lowered and ("cluster" in lowered or "cluster_id" in lowered)


def rows_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        nodes = payload.get("nodes")
        if isinstance(nodes, list):
            return [node for node in nodes if isinstance(node, dict)]
        if "emoji" in payload and {"cluster", "cluster_id", "clusterId"}.intersection(payload):
            return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def parse_json_artifact(path: Path) -> ClusterArtifact | None:
    if not looks_like_cluster_json(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    rows = rows_from_json_payload(payload)
    return artifact_from_rows(path, "json", rows)


def extract_balanced_json(text: str, start: int) -> str | None:
    brace_start = text.find("{", start)
    if brace_start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    quote = ""
    for index in range(brace_start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : index + 1]
    return None


def parse_html_artifact(path: Path) -> ClusterArtifact | None:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    if "emoji" not in text or "cluster" not in text:
        return None

    payload_text: str | None = None
    for marker in HTML_DATA_MARKERS:
        marker_index = text.find(marker)
        if marker_index < 0:
            continue
        candidate = extract_balanced_json(text, marker_index + len(marker))
        if candidate is not None and '"nodes"' in candidate:
            payload_text = candidate
            break

    if payload_text is None:
        return None

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    rows = rows_from_json_payload(payload)
    return artifact_from_rows(path, "html", rows)


def discover_cluster_artifacts(root: Path) -> list[ClusterArtifact]:
    artifacts: list[ClusterArtifact] = []
    for path in iter_files(root, {".json"}):
        artifact = parse_json_artifact(path)
        if artifact is not None:
            artifacts.append(artifact)
    for path in iter_files(root, {".csv"}):
        artifact = parse_csv_artifact(path)
        if artifact is not None:
            artifacts.append(artifact)
    for path in iter_files(root, {".html", ".htm"}):
        artifact = parse_html_artifact(path)
        if artifact is not None:
            artifacts.append(artifact)
    return sorted(
        artifacts,
        key=lambda artifact: (
            artifact.score,
            artifact.path.stat().st_mtime if artifact.path.exists() else 0.0,
            str(artifact.path),
        ),
        reverse=True,
    )


def load_cluster_artifact(path: Path) -> ClusterArtifact:
    suffix = path.suffix.lower()
    parsers = {
        ".csv": parse_csv_artifact,
        ".json": parse_json_artifact,
        ".html": parse_html_artifact,
        ".htm": parse_html_artifact,
    }
    parser = parsers.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported cluster artifact suffix for {path}.")
    artifact = parser(path)
    if artifact is None:
        raise ValueError(
            f"Could not parse {path} as a cluster artifact. Expected node rows with "
            "`emoji` and anonymous `cluster`/`cluster_id` fields."
        )
    return artifact


def discover_current_cluster_artifact(root: Path) -> ClusterArtifact:
    artifacts = discover_cluster_artifacts(root)
    if not artifacts:
        raise FileNotFoundError(
            f"No cluster artifact found under {root}. Expected CSV/JSON rows or an HTML payload "
            "with `emoji` and anonymous `cluster`/`cluster_id` fields."
        )
    return artifacts[0]


def support_map_for_node(node: EmojiClusterNode) -> dict[str, float]:
    support_by_cluster: dict[str, float] = {}
    for candidate in node.candidate_clusters:
        if candidate.support > 0 and math.isfinite(candidate.support):
            support_by_cluster[candidate.cluster_id] = max(
                support_by_cluster.get(candidate.cluster_id, 0.0),
                candidate.support,
            )

    if not support_by_cluster:
        return {}

    if node.primary_cluster not in support_by_cluster:
        support_by_cluster[node.primary_cluster] = max(support_by_cluster.values())
    return support_by_cluster


def sharpen_memberships(raw_by_cluster: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    weights = {
        cluster_id: probability ** (1.0 / temperature)
        for cluster_id, probability in raw_by_cluster.items()
        if probability > 0
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Cannot sharpen an empty membership distribution.")
    return {cluster_id: weight / total for cluster_id, weight in weights.items()}


def build_membership_rows(
    nodes: Iterable[EmojiClusterNode],
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[MembershipRow]:
    rows: list[MembershipRow] = []
    for node in nodes:
        support_by_cluster = support_map_for_node(node)
        if support_by_cluster:
            support_sum = sum(support_by_cluster.values())
            raw_by_cluster = {
                cluster_id: support / support_sum
                for cluster_id, support in support_by_cluster.items()
            }
            sharp_by_cluster = sharpen_memberships(raw_by_cluster, temperature)
            source_type = "soft_candidate" if len(raw_by_cluster) > 1 else "onehot"
            for cluster_id in sorted(raw_by_cluster, key=cluster_sort_key):
                rows.append(
                    MembershipRow(
                        emoji=node.emoji,
                        cluster_id=cluster_id,
                        membership_raw=raw_by_cluster[cluster_id],
                        membership_sharp=sharp_by_cluster[cluster_id],
                        source_type=source_type,
                        observed_count=node.observed_count,
                        support_score=support_by_cluster[cluster_id],
                    )
                )
        else:
            rows.append(
                MembershipRow(
                    emoji=node.emoji,
                    cluster_id=node.primary_cluster,
                    membership_raw=1.0,
                    membership_sharp=1.0,
                    source_type="onehot",
                    observed_count=node.observed_count,
                    support_score=None,
                )
            )
    return rows


def entropy(probabilities: Iterable[float]) -> float:
    value = 0.0
    for probability in probabilities:
        if probability > 0:
            value -= probability * math.log2(probability)
    return value


def ambiguous_rows(
    by_emoji: dict[str, list[MembershipRow]],
    probability_attr: str,
    top_k: int,
) -> tuple[dict[str, object], ...]:
    top_ambiguous: list[dict[str, object]] = []
    for emoji, emoji_rows in by_emoji.items():
        probabilities = [float(getattr(row, probability_attr)) for row in emoji_rows]
        memberships = ", ".join(
            f"{row.cluster_id}:{float(getattr(row, probability_attr)):.4f}"
            for row in sorted(
                emoji_rows,
                key=lambda item: (-float(getattr(item, probability_attr)), item.cluster_id),
            )
        )
        observed_count = next(
            (row.observed_count for row in emoji_rows if row.observed_count is not None),
            None,
        )
        top_ambiguous.append(
            {
                "emoji": emoji,
                "entropy": entropy(probabilities),
                "cluster_count": len(emoji_rows),
                "observed_count": observed_count,
                "memberships": memberships,
            }
        )
    top_ambiguous.sort(
        key=lambda row: (
            float(row["entropy"]),
            int(row["cluster_count"]),
            int(row["observed_count"] or 0),
        ),
        reverse=True,
    )
    return tuple(top_ambiguous[:top_k])


def validate_membership_rows(rows: list[MembershipRow], top_k: int = 20) -> MembershipValidation:
    by_emoji: dict[str, list[MembershipRow]] = {}
    for row in rows:
        by_emoji.setdefault(row.emoji, []).append(row)

    if not by_emoji:
        raise ValueError("Membership matrix is empty.")

    max_raw_error = 0.0
    max_sharp_error = 0.0
    onehot_count = 0
    soft_count = 0
    for emoji, emoji_rows in by_emoji.items():
        raw_sum = sum(row.membership_raw for row in emoji_rows)
        sharp_sum = sum(row.membership_sharp for row in emoji_rows)
        max_raw_error = max(max_raw_error, abs(raw_sum - 1.0))
        max_sharp_error = max(max_sharp_error, abs(sharp_sum - 1.0))
        if abs(raw_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Raw membership probabilities for {emoji} sum to {raw_sum:.12f}, not 1.0."
            )
        if abs(sharp_sum - 1.0) > VALIDATION_TOLERANCE:
            raise ValueError(
                f"Sharpened membership probabilities for {emoji} sum to {sharp_sum:.12f}, not 1.0."
            )

        if len(emoji_rows) == 1 and emoji_rows[0].source_type == "onehot":
            onehot_count += 1
        else:
            soft_count += 1
    return MembershipValidation(
        observed_emoji_count=len(by_emoji),
        onehot_emoji_count=onehot_count,
        soft_multi_cluster_emoji_count=soft_count,
        average_clusters_per_emoji=len(rows) / len(by_emoji),
        max_raw_sum_error=max_raw_error,
        max_sharp_sum_error=max_sharp_error,
        top_ambiguous_raw=ambiguous_rows(by_emoji, "membership_raw", top_k),
        top_ambiguous_sharp=ambiguous_rows(by_emoji, "membership_sharp", top_k),
    )


def write_membership_table(path: Path, rows: list[MembershipRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "emoji",
                "cluster_id",
                "membership_raw",
                "membership_sharp",
                "source_type",
                "observed_count",
                "support_score",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.emoji, cluster_sort_key(item.cluster_id))):
            writer.writerow(
                {
                    "emoji": row.emoji,
                    "cluster_id": row.cluster_id,
                    "membership_raw": f"{row.membership_raw:.15f}",
                    "membership_sharp": f"{row.membership_sharp:.15f}",
                    "source_type": row.source_type,
                    "observed_count": row.observed_count if row.observed_count is not None else "",
                    "support_score": f"{row.support_score:.15f}" if row.support_score is not None else "",
                }
            )


def write_wide_matrix(path: Path, rows: list[MembershipRow]) -> None:
    clusters = sorted({row.cluster_id for row in rows}, key=cluster_sort_key)
    by_emoji: dict[str, dict[str, MembershipRow]] = {}
    observed_counts: dict[str, int | None] = {}
    for row in rows:
        by_emoji.setdefault(row.emoji, {})[row.cluster_id] = row
        if row.emoji not in observed_counts or observed_counts[row.emoji] is None:
            observed_counts[row.emoji] = row.observed_count

    with path.open("w", encoding="utf-8", newline="") as handle:
        raw_fields = [f"raw_{cluster_id}" for cluster_id in clusters]
        sharp_fields = [f"sharp_{cluster_id}" for cluster_id in clusters]
        writer = csv.DictWriter(handle, fieldnames=["emoji", "observed_count", *raw_fields, *sharp_fields])
        writer.writeheader()
        for emoji in sorted(by_emoji):
            writer.writerow(
                {
                    "emoji": emoji,
                    "observed_count": observed_counts[emoji] if observed_counts[emoji] is not None else "",
                    **{
                        f"raw_{cluster_id}": f"{by_emoji[emoji].get(cluster_id).membership_raw:.15f}"
                        if cluster_id in by_emoji[emoji]
                        else "0.000000000000000"
                        for cluster_id in clusters
                    },
                    **{
                        f"sharp_{cluster_id}": f"{by_emoji[emoji].get(cluster_id).membership_sharp:.15f}"
                        if cluster_id in by_emoji[emoji]
                        else "0.000000000000000"
                        for cluster_id in clusters
                    },
                }
            )


def write_ambiguous_summary(
    path: Path,
    rows: tuple[dict[str, object], ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["emoji", "entropy", "cluster_count", "observed_count", "memberships"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "emoji": row["emoji"],
                    "entropy": f"{float(row['entropy']):.12f}",
                    "cluster_count": row["cluster_count"],
                    "observed_count": row["observed_count"] if row["observed_count"] is not None else "",
                    "memberships": row["memberships"],
                }
            )


def write_summary_json(
    path: Path,
    artifact: ClusterArtifact,
    rows: list[MembershipRow],
    validation: MembershipValidation,
    temperature: float,
) -> None:
    payload = {
        "source_artifact": str(artifact.path),
        "source_kind": artifact.kind,
        "source_rationale": artifact.rationale,
        "temperature": temperature,
        "membership_row_count": len(rows),
        "observed_emoji_count": validation.observed_emoji_count,
        "onehot_emoji_count": validation.onehot_emoji_count,
        "soft_multi_cluster_emoji_count": validation.soft_multi_cluster_emoji_count,
        "average_clusters_per_emoji": validation.average_clusters_per_emoji,
        "max_raw_sum_error": validation.max_raw_sum_error,
        "max_sharp_sum_error": validation.max_sharp_sum_error,
        "top_ambiguous_raw": list(validation.top_ambiguous_raw),
        "top_ambiguous_sharp": list(validation.top_ambiguous_sharp),
        "hard_constraints": {
            "emoji_names_used": False,
            "aliases_used": False,
            "unicode_names_used": False,
            "transition_features_used": False,
            "cluster_ids_anonymous": True,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(
    path: Path,
    artifact: ClusterArtifact,
    validation: MembershipValidation,
    output_paths: dict[str, Path],
    temperature: float,
) -> None:
    lines = [
        "# Soft Emoji-to-Cluster Membership",
        "",
        "## Source Artifact",
        "",
        f"- Path: `{artifact.path}`",
        f"- Kind: `{artifact.kind}`",
        f"- Discovery rationale: {artifact.rationale}",
        "- Emoji names, aliases, Unicode names, and transition information were not used.",
        "",
        "## Validation",
        "",
        f"- Observed emojis: `{validation.observed_emoji_count}`",
        f"- One-hot emojis: `{validation.onehot_emoji_count}`",
        f"- Soft multi-cluster emojis: `{validation.soft_multi_cluster_emoji_count}`",
        f"- Average clusters per emoji: `{validation.average_clusters_per_emoji:.6f}`",
        f"- Temperature for sharpened membership: `{temperature}`",
        f"- Max raw membership-sum error: `{validation.max_raw_sum_error:.12g}`",
        f"- Max sharpened membership-sum error: `{validation.max_sharp_sum_error:.12g}`",
        "",
        "## Top Ambiguous Emojis by Raw Membership Entropy",
        "",
    ]
    if validation.top_ambiguous_raw:
        lines.append("| emoji | entropy | clusters | observed_count | memberships |")
        lines.append("|---|---:|---:|---:|---|")
        for row in validation.top_ambiguous_raw[:12]:
            observed = "" if row["observed_count"] is None else str(row["observed_count"])
            lines.append(
                f"| {row['emoji']} | {float(row['entropy']):.6f} | "
                f"{row['cluster_count']} | {observed} | `{row['memberships']}` |"
            )
    else:
        lines.append("- No ambiguous emojis.")

    lines.extend(["", "## Top Ambiguous Emojis by Sharpened Membership Entropy", ""])
    if validation.top_ambiguous_sharp:
        lines.append("| emoji | entropy | clusters | observed_count | memberships |")
        lines.append("|---|---:|---:|---:|---|")
        for row in validation.top_ambiguous_sharp[:12]:
            observed = "" if row["observed_count"] is None else str(row["observed_count"])
            lines.append(
                f"| {row['emoji']} | {float(row['entropy']):.6f} | "
                f"{row['cluster_count']} | {observed} | `{row['memberships']}` |"
            )
    else:
        lines.append("- No ambiguous emojis.")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Canonical membership table: `{output_paths['membership']}`",
            f"- Wide matrix: `{output_paths['wide']}`",
            f"- Raw ambiguous summary: `{output_paths['ambiguous_raw']}`",
            f"- Sharpened ambiguous summary: `{output_paths['ambiguous_sharp']}`",
            f"- JSON summary: `{output_paths['summary_json']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_soft_membership(
    artifact: ClusterArtifact,
    output_dir: Path,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[dict[str, Path], MembershipValidation]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_membership_rows(artifact.nodes, temperature=temperature)
    validation = validate_membership_rows(rows)

    output_paths = {
        "membership": output_dir / "emoji_cluster_membership.csv",
        "wide": output_dir / "emoji_cluster_membership_wide.csv",
        "ambiguous_raw": output_dir / "ambiguous_emojis_raw.csv",
        "ambiguous_sharp": output_dir / "ambiguous_emojis_sharp.csv",
        "summary_json": output_dir / "membership_summary.json",
        "report": output_dir / "emoji_cluster_membership_report.md",
    }
    write_membership_table(output_paths["membership"], rows)
    write_wide_matrix(output_paths["wide"], rows)
    write_ambiguous_summary(output_paths["ambiguous_raw"], validation.top_ambiguous_raw)
    write_ambiguous_summary(output_paths["ambiguous_sharp"], validation.top_ambiguous_sharp)
    write_summary_json(output_paths["summary_json"], artifact, rows, validation, temperature)
    write_markdown_report(output_paths["report"], artifact, validation, output_paths, temperature)
    return output_paths, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a soft emoji-to-cluster membership matrix from the current cluster artifact."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to search for cluster artifacts.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Optional explicit cluster artifact to parse instead of auto-discovery.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <selected artifact directory>/soft_membership.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sharpened membership. Defaults to 0.7.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = (
        load_cluster_artifact(args.artifact.resolve())
        if args.artifact
        else discover_current_cluster_artifact(args.root.resolve())
    )
    output_dir = args.output_dir.resolve() if args.output_dir else artifact.path.parent / "soft_membership"
    output_paths, validation = build_soft_membership(artifact, output_dir, temperature=args.temperature)
    print(f"Selected cluster artifact: {artifact.path}")
    print(f"Artifact rationale: {artifact.rationale}")
    print(f"Wrote membership table: {output_paths['membership']}")
    print(f"Wrote report: {output_paths['report']}")
    print(
        "Summary: "
        f"{validation.observed_emoji_count} emojis, "
        f"{validation.onehot_emoji_count} one-hot, "
        f"{validation.soft_multi_cluster_emoji_count} soft multi-cluster."
    )


if __name__ == "__main__":
    main()
