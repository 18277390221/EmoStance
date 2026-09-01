from __future__ import annotations

import argparse
from pathlib import Path

from audit_lib import build_experiment_payload, write_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the LLM-Human emoji distribution audit experiment.")
    parser.add_argument("--project-root", default=".", help="Project root containing pre_data/, data/, and clustering outputs.")
    parser.add_argument("--output-dir", default="human_llm_emoji_audit", help="Experiment output directory.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--n-items", type=int, default=120)
    parser.add_argument("--n-annotators", type=int, default=3)
    parser.add_argument("--sampling-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_experiment_payload(
        project_root=Path(args.project_root),
        split=args.split,
        n_items=args.n_items,
        n_annotators=args.n_annotators,
        sampling_seed=args.sampling_seed,
    )
    written = write_experiment(payload, Path(args.output_dir))
    report = payload["sampling_report"]
    print("LLM-Human emoji audit experiment built.")
    print(f"Output directory: {Path(args.output_dir)}")
    print(f"Test candidate total: {report['candidate_count']}")
    print(f"Final samples: {len(payload['public_items'])}")
    print(f"low/medium/high: {report['stratum_sample_counts']}")
    print(f"Unique dialogues: {len({item['dialogue_id'] for item in payload['private_items']})}")
    print(f"Candidate emoji: {len(payload['emoji_inventory'])}")
    print(f"Questionnaires: {args.n_annotators}")
    print("HTML files:")
    for path in written:
        if path.suffix == ".html":
            print(f"  {path}")


if __name__ == "__main__":
    main()
