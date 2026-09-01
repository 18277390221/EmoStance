#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


MODEL_NAMES = ["Claude Sonnet", "DeepSeek", "Gemini", "GPT", "aggregate_top1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate offline emoji audit outputs.")
    parser.add_argument("--out_dir", default="data_eval")
    parser.add_argument("--allow_smoke_test", action="store_true")
    return parser.parse_args()


def choose_dir(out_dir: Path, allow_smoke_test: bool) -> Path:
    if (out_dir / "html" / "annotator_01.html").exists():
        return out_dir
    smoke = out_dir / "smoke_test"
    if allow_smoke_test and (smoke / "html" / "annotator_01.html").exists():
        return smoke
    raise FileNotFoundError(f"No audit HTML files found under {out_dir}; run build_human_audit.py first.")


def extract_packages(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'<script\s+type="application/json"\s+id="audit-data">(.*?)</script>',
        text,
        flags=re.S,
    )
    if not match:
        raise ValueError(f"Could not find embedded audit-data JSON in {path}")
    return json.loads(match.group(1))


def visible_ui_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'<script\s+type="application/json"\s+id="audit-data">.*?</script>', "", text, flags=re.S)
    return text


def has_remote_resource(text: str) -> bool:
    patterns = [
        r"<script[^>]+\bsrc\s*=",
        r"<link[^>]+\bhref\s*=",
        r"https?://",
        r"@import",
    ]
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def validate_package_content(packages: list[dict]) -> list[str]:
    errors: list[str] = []
    seen_package_ids: set[str] = set()
    seen_dialogue_ids: set[str] = set()
    duplicate_dialogues = 0
    for pkg in packages:
        package_id = pkg.get("package_id")
        if not package_id:
            errors.append("Package missing package_id")
        elif package_id in seen_package_ids:
            errors.append(f"Duplicate package_id: {package_id}")
        seen_package_ids.add(package_id)
        dialogue_id = pkg.get("dialogue_id")
        if dialogue_id in seen_dialogue_ids:
            duplicate_dialogues += 1
        seen_dialogue_ids.add(dialogue_id)
        turns = pkg.get("turns") or []
        if len(turns) < 2:
            errors.append(f"{package_id} has fewer than 2 utterances")
        if pkg.get("situation") and not pkg.get("situation_zh"):
            errors.append(f"{package_id} has situation but missing situation_zh")
        for turn in turns:
            tid = turn.get("turn_index")
            if not turn.get("english_original_text"):
                errors.append(f"{package_id} turn {tid} missing English text")
            if not turn.get("zh_translation"):
                errors.append(f"{package_id} turn {tid} missing zh_translation")
            if not turn.get("displayed_emoji"):
                errors.append(f"{package_id} turn {tid} missing displayed emoji")
    if duplicate_dialogues:
        errors.append(f"Duplicate dialogue_id count: {duplicate_dialogues}")
    return errors


def main() -> None:
    args = parse_args()
    root = choose_dir(Path(args.out_dir), args.allow_smoke_test)
    html_dir = root / "html"
    html_paths = [html_dir / f"annotator_{idx:02d}.html" for idx in range(1, 4)]
    errors: list[str] = []
    for path in html_paths:
        if not path.exists():
            errors.append(f"Missing HTML file: {path}")
    if errors:
        raise SystemExit("\n".join(errors))

    package_lists = [extract_packages(path) for path in html_paths]
    package_id_sets = [{pkg["package_id"] for pkg in packages} for packages in package_lists]
    first_ids = package_id_sets[0]
    for idx, ids in enumerate(package_id_sets[1:], 2):
        if ids != first_ids:
            errors.append(f"annotator_0{idx}.html package_id set differs from annotator_01.html")
    n = len(package_lists[0])
    expected_n = 300 if root.name != "smoke_test" else n
    if n != expected_n:
        errors.append(f"Expected {expected_n} packages, found {n}")
    if root.name != "smoke_test" and n != 300:
        errors.append(f"Final build must contain exactly 300 packages, found {n}")
    orders = [[pkg["package_id"] for pkg in packages] for packages in package_lists]
    if orders[0] == orders[1] or orders[0] == orders[2] or orders[1] == orders[2]:
        errors.append("Annotator package orders are not all different")

    content_errors = validate_package_content(package_lists[0])
    errors.extend(content_errors)

    model_counts = Counter(pkg.get("sampled_model", "") for pkg in package_lists[0])
    split_counts = Counter(pkg.get("split", "") for pkg in package_lists[0])
    if root.name != "smoke_test":
        for model, count in model_counts.items():
            if model != "aggregate_top1" and count != 75:
                errors.append(f"Model balance expected 75 per model, got {model}={count}")
    for path in html_paths:
        text = path.read_text(encoding="utf-8")
        if has_remote_resource(text):
            errors.append(f"Remote/external resource reference detected in {path}")
        ui_text = visible_ui_text(path)
        for model_name in MODEL_NAMES:
            if model_name and model_name in ui_text:
                errors.append(f"Hidden sampled_model visible in annotation UI for {path}: {model_name}")

    manifest_path = root / "audit_sample_manifest.csv"
    sample_path = root / "audit_sample_300_model_dialogue_packages.jsonl"
    if not manifest_path.exists():
        errors.append(f"Missing manifest CSV: {manifest_path}")
    if not sample_path.exists():
        errors.append(f"Missing sample JSONL: {sample_path}")

    print(f"Validated directory: {root}")
    print(f"Packages per HTML: {n}")
    print(f"Model distribution: {dict(model_counts)}")
    print(f"Split distribution: {dict(split_counts)}")
    if errors:
        print("Validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("Validation PASSED.")


if __name__ == "__main__":
    main()
