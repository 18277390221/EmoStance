from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

from .data import (
    discover_annotation_files,
    entropy,
    group_annotation_rows,
    iter_adjacent_pairs,
    load_annotation_records,
    read_records,
    write_json,
    write_jsonl,
)
from .space import cluster_distribution, emoji_distribution, load_stance_space, stance_vector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLUSTERS = PROJECT_ROOT / "src/name_free_emoji_clustering/outputs/soft_membership/emoji_cluster_membership.csv"
DEFAULT_EMOJI_VECTORS = PROJECT_ROOT / "src/name_free_emoji_clustering/outputs/tables/emoji_centroids.csv"


def split_name(value: str) -> str:
    value = value.lower()
    return "dev" if value in {"valid", "validation"} else value


def resolve_annotation_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.annotation_root:
        paths.extend(discover_annotation_files(args.annotation_root, args.annotation_glob))
    if args.annotations:
        for item in args.annotations:
            paths.append(Path(item))
    if not paths:
        raise SystemExit("Provide --annotation-root data or one/more files via --annotations.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing annotation file(s): " + ", ".join(missing[:8]))
    return sorted(dict.fromkeys(paths))


def build_examples(annotation_paths: Sequence[str | Path], cluster_path: str | Path, emoji_vector_path: str | Path | None = None) -> tuple[List[Dict], Dict]:
    records = load_annotation_records(annotation_paths)
    cluster_rows = read_records(cluster_path)
    vector_rows = read_records(emoji_vector_path) if emoji_vector_path else None
    utterances = group_annotation_rows(records)
    space = load_stance_space(cluster_rows, vector_rows)

    q_cluster_by_key = {}
    z_by_key = {}
    missing_label_count = 0
    for utt in utterances:
        key = (split_name(str(utt["split"])), str(utt["dialogue_id"]), int(utt["turn_id"]))
        q_e = emoji_distribution(utt.get("emoji_votes", {}), space.emojis)
        if not utt.get("emoji_votes"):
            missing_label_count += 1
        q_c = cluster_distribution(q_e, space.membership)
        q_cluster_by_key[key] = q_c
        z_by_key[key] = stance_vector(q_e, space.emoji_vectors)

    examples: List[Dict] = []
    for cur, nxt, context in iter_adjacent_pairs(utterances):
        cur_key = (split_name(str(cur["split"])), str(cur["dialogue_id"]), int(cur["turn_id"]))
        nxt_key = (split_name(str(nxt["split"])), str(nxt["dialogue_id"]), int(nxt["turn_id"]))
        transition = f'{cur["role"]}->{nxt["role"]}'
        examples.append(
            {
                "dialogue_id": cur["dialogue_id"],
                "turn_id": cur["turn_id"],
                "split": split_name(str(cur["split"])),
                "role": cur["role"],
                "next_role": nxt["role"],
                "transition": transition,
                "text": cur["text"],
                "context": context,
                "response": nxt["text"],
                "situation": cur.get("situation", ""),
                "source_cluster": q_cluster_by_key[cur_key].tolist(),
                "target_cluster": q_cluster_by_key[nxt_key].tolist(),
                "source_vector": z_by_key[cur_key].tolist(),
                "target_vector": z_by_key[nxt_key].tolist(),
                "source_entropy": entropy(q_cluster_by_key[cur_key]),
                "target_entropy": entropy(q_cluster_by_key[nxt_key]),
            }
        )

    meta = {
        "num_annotation_rows": len(records),
        "num_utterances": len(utterances),
        "num_examples": len(examples),
        "num_missing_label_utterances": missing_label_count,
        "num_emojis": space.num_emojis,
        "num_clusters": space.num_clusters,
        "vector_dim": space.vector_dim,
        "emojis": space.emojis,
        "clusters": space.clusters,
        "membership": space.membership.tolist(),
        "emoji_vectors": space.emoji_vectors.tolist(),
    }
    return examples, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare latent stance control examples.")
    parser.add_argument("--annotations", nargs="*", help="One or more JSON/JSONL/CSV annotation files. Existing dialogue-level JSON is supported.")
    parser.add_argument("--annotation-root", help="Root containing model subfolders, e.g. data.")
    parser.add_argument("--annotation-glob", default="*/*_emoji_annotations.json")
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTERS), help="Cluster membership artifact. Defaults to the repository's released name-free soft membership CSV.")
    parser.add_argument("--emoji-vectors", default=None, help="Optional emoji vector/centroid CSV. Defaults to existing emoji_centroids.csv if present.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cluster_path = Path(args.clusters)
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster artifact not found: {cluster_path}")
    emoji_vector_path = Path(args.emoji_vectors) if args.emoji_vectors else DEFAULT_EMOJI_VECTORS
    if not emoji_vector_path.exists():
        emoji_vector_path = None

    annotation_paths = resolve_annotation_paths(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    examples, meta = build_examples(annotation_paths, cluster_path, emoji_vector_path)
    by_split: Dict[str, List[Dict]] = {}
    for ex in examples:
        by_split.setdefault(ex.get("split", "train"), []).append(ex)
    for split, rows in sorted(by_split.items()):
        write_jsonl(out / f"{split}.jsonl", rows)
    write_json(out / "meta.json", meta)
    write_json(
        out / "prepare_summary.json",
        {
            "annotation_files": [str(path) for path in annotation_paths],
            "clusters": str(cluster_path),
            "emoji_vectors": str(emoji_vector_path) if emoji_vector_path else None,
            "num_examples": len(examples),
            "splits": {split: len(rows) for split, rows in sorted(by_split.items())},
            "num_emojis": meta["num_emojis"],
            "num_clusters": meta["num_clusters"],
            "vector_dim": meta["vector_dim"],
        },
    )


if __name__ == "__main__":
    main()
