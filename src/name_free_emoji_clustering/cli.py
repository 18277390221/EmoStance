from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intrinsic, name-free emoji clustering from repository data."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan for inputs. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <root>/src/name_free_emoji_clustering/outputs.",
    )
    parser.add_argument("--tau", type=float, default=50.0, help="Rare-emoji shrinkage tau.")
    parser.add_argument("--knn-k", type=int, default=8, help="Top-k neighbors per emoji.")
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Dimensionality for deterministic hashed TF-IDF utterance embeddings.",
    )
    parser.add_argument(
        "--lambda-ctx",
        type=float,
        default=0.65,
        help="Context similarity fusion weight.",
    )
    parser.add_argument(
        "--lambda-conf",
        type=float,
        default=0.35,
        help="Soft-confusion similarity fusion weight.",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Rebuild utterance embedding cache even if a matching cache exists.",
    )
    parser.add_argument(
        "--cluster-splits",
        default="train",
        help=(
            "Comma-separated split names used to build the emoji graph and clusters. "
            "Defaults to train to prevent development/test leakage."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.root / "src" / "name_free_emoji_clustering" / "outputs"

    result = run_pipeline(
        PipelineConfig(
            root=args.root.resolve(),
            output_dir=output_dir.resolve(),
            tau=args.tau,
            knn_k=args.knn_k,
            embedding_dim=args.embedding_dim,
            lambda_ctx=args.lambda_ctx,
            lambda_conf=args.lambda_conf,
            force_embeddings=args.force_embeddings,
            cluster_splits=tuple(
                split.strip()
                for split in args.cluster_splits.split(",")
                if split.strip()
            ),
        )
    )
    print(f"Wrote name-free clustering outputs to {result.output_dir}")
    print(f"Analysis report: {result.analysis_report_path}")
    print(f"HTML visualization: {result.html_path}")
