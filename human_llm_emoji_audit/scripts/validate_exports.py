from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audit_lib import validate_exports, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate three human annotation JSON exports.")
    parser.add_argument("--experiment-dir", default="human_llm_emoji_audit")
    parser.add_argument("--exports-dir", default="human_llm_emoji_audit/exports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    valid, result, markdown = validate_exports(experiment_dir, Path(args.exports_dir))
    reports = experiment_dir / "reports"
    reports.mkdir(exist_ok=True)
    write_json(reports / "export_validation.json", result)
    (reports / "export_validation.md").write_text(markdown, encoding="utf-8")
    if valid:
        print("validate_exports: PASS")
        print(f"total_judgments={result['total_judgments']}")
        return
    print("validate_exports: FAIL")
    for err in result["errors"]:
        print(f"- {err}")
    sys.exit(1)


if __name__ == "__main__":
    main()
