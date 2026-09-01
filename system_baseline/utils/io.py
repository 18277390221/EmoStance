from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SYSTEM_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SYSTEM_DIR.parent


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(project_root or PROJECT_ROOT).resolve() / p


def relpath(path: str | Path, root: str | Path | None = None) -> str:
    p = Path(path).resolve()
    base = Path(root or PROJECT_ROOT).resolve()
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(sanitize(payload), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {p}:{line_no}: {exc}") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(sanitize(dict(row)), ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: set[str] = set()
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(str(key))
                    keys.append(str(key))
        fieldnames = keys
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fieldnames})


def csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(sanitize(value), ensure_ascii=False, sort_keys=True)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to read system_baseline configs.") from exc
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def git_commit(project_root: str | Path | None = None) -> str | None:
    root = Path(project_root or PROJECT_ROOT)
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def python_executable(project_root: str | Path | None = None) -> str:
    root = Path(project_root or PROJECT_ROOT)
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return os.environ.get("PYTHON", "python3")


def read_table_like(path: str | Path, max_rows: int = 1000) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(p)[:max_rows]
    if suffix == ".json":
        obj = read_json(p)
        if isinstance(obj, list):
            return [x for x in obj[:max_rows] if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("records", "data", "examples", "items", "utterances", "annotations"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key][:max_rows] if isinstance(x, dict)]
            return [obj]
        return []
    if suffix == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = []
            for idx, row in enumerate(csv.DictReader(f)):
                if idx >= max_rows:
                    break
                rows.append(dict(row))
            return rows
    return []

