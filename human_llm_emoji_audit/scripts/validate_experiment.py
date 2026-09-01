from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audit_lib import validate_experiment_artifacts, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the built LLM-Human emoji audit experiment.")
    parser.add_argument("--experiment-dir", default="human_llm_emoji_audit")
    parser.add_argument("--skip-reproducibility", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    valid, errors, summary = validate_experiment_artifacts(
        experiment_dir,
        check_reproducible=not args.skip_reproducibility,
    )
    reports = experiment_dir / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "experiment_validation.json", summary)
    lines = [
        "# Experiment Validation",
        "",
        f"- Valid: `{str(valid).lower()}`",
        f"- Sample count: {summary['sample_count']}",
        f"- Stratum counts: {summary['stratum_counts']}",
        f"- Unique dialogues: {summary['unique_dialogues']}",
        f"- Emoji count: {summary['emoji_count']}",
        f"- Questionnaires: {summary['questionnaire_count']}",
        f"- Duplicate dialogue count: {summary['duplicate_dialogue_count']}",
        f"- Reproducibility: {summary['reproducibility']}",
    ]
    if errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {err}" for err in errors)
    (reports / "experiment_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if valid:
        print("validate_experiment: PASS")
        print(f"sample_count={summary['sample_count']} unique_dialogues={summary['unique_dialogues']} emoji_count={summary['emoji_count']}")
        return
    print("validate_experiment: FAIL")
    for err in errors:
        print(f"- {err}")
    sys.exit(1)


if __name__ == "__main__":
    main()
