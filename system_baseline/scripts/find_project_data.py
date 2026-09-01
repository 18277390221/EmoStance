from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from system_baseline.utils.io import PROJECT_ROOT, ensure_dir, read_table_like, relpath, write_json
from system_baseline.utils.text import normalize_for_match


KEYWORDS = [
    "empatheticdialogues",
    "ed",
    "train",
    "valid",
    "dev",
    "test",
    "situation",
    "context",
    "conv_id",
    "dialogue_id",
    "utterance",
    "speaker",
    "response",
    "reference",
    "target",
]

TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml"}
DATA_SUFFIXES = TEXT_SUFFIXES | {".p", ".pkl", ".npy", ".npz", ".parquet", ".pq"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover ED/EmpatheticDialogues data and generation artifacts in the repository.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--max-file-mb", type=float, default=80.0, help="Skip text/schema inspection above this size.")
    parser.add_argument("--output-dir", default="system_baseline/outputs/diagnostics")
    return parser.parse_args()


def iter_candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel_lower = str(path.relative_to(root)).lower()
        if suffix in DATA_SUFFIXES or any(k.lower() in rel_lower for k in KEYWORDS):
            files.append(path)
    return sorted(files)


def detected_keys(rows: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows[:50]:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                keys.append(str(key))
    return keys


def nested_key_hits(rows: list[dict[str, Any]]) -> set[str]:
    hits: set[str] = set()
    stack: list[Any] = rows[:20]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                low = str(key).lower()
                for kw in KEYWORDS:
                    if kw.lower() == low or kw.lower() in low:
                        hits.add(str(key))
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item[:20])
    return hits


def estimate_row_count(path: Path, rows: list[dict[str, Any]]) -> int | None:
    if path.suffix.lower() == ".jsonl":
        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    if path.suffix.lower() == ".csv":
        count = -1
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for count, _ in enumerate(f):
                pass
        return max(count, 0)
    if rows:
        return len(rows)
    return None


def looks_like_ed_test(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> bool:
    name = str(path).lower()
    key_text = " ".join(k.lower() for k in keys)
    sample = normalize_for_match(json.dumps(rows[:3], ensure_ascii=False)[:4000]) if rows else ""
    has_ed = "empathetic" in name or "/ed" in name or "ed_" in name or "situation" in key_text or "situation" in sample
    has_test = "test" in name or any(str(row.get("split", "")).lower() == "test" for row in rows[:100])
    has_response = any(token in key_text for token in ["target", "response", "utterance", "turns"]) or any(
        token in sample for token in ["target", "response", "utterance", "turns"]
    )
    return bool(has_ed and has_test and has_response)


def inspect_file(path: Path, root: Path, max_file_mb: float) -> dict[str, Any]:
    suffix = path.suffix.lower()
    size_mb = path.stat().st_size / (1024 * 1024)
    rows: list[dict[str, Any]] = []
    error = ""
    if suffix in {".json", ".jsonl", ".csv"} and size_mb <= max_file_mb:
        try:
            rows = read_table_like(path, max_rows=200)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    keys = detected_keys(rows)
    nested = sorted(nested_key_hits(rows))
    row_count = None
    try:
        row_count = estimate_row_count(path, rows)
    except Exception:
        row_count = len(rows) if rows else None
    return {
        "path": relpath(path, root),
        "file_type": suffix.lstrip(".") or "unknown",
        "size_mb": round(size_mb, 3),
        "estimated_row_count": row_count,
        "detected_columns_or_keys": keys[:80],
        "nested_keyword_keys": nested[:80],
        "appears_to_contain_ed_test_examples": looks_like_ed_test(path, rows, keys + nested),
        "inspection_error": error,
    }


def write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        "# Data Discovery Report",
        "",
        "This report is generated by `python -m system_baseline.scripts.find_project_data`.",
        "",
        "| path | type | rows | ED test? | detected columns/keys |",
        "|---|---:|---:|---:|---|",
    ]
    for rec in records:
        keys = ", ".join(rec.get("detected_columns_or_keys", [])[:16])
        marker = "yes" if rec.get("appears_to_contain_ed_test_examples") else "no"
        lines.append(
            f"| `{rec['path']}` | {rec['file_type']} | {rec.get('estimated_row_count') or ''} | {marker} | {keys} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    out_dir = ensure_dir(root / args.output_dir)
    records = [inspect_file(path, root, args.max_file_mb) for path in iter_candidate_files(root)]
    records.sort(key=lambda x: (not x["appears_to_contain_ed_test_examples"], x["path"]))
    write_json(out_dir / "data_discovery.json", records)
    write_markdown(out_dir / "data_discovery_report.md", records)
    print(f"Wrote {out_dir / 'data_discovery_report.md'}")
    print(f"Candidate files: {len(records)}")
    print(f"Likely ED test files: {sum(1 for r in records if r['appears_to_contain_ed_test_examples'])}")


if __name__ == "__main__":
    main()

