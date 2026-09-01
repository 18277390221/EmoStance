from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_log(path: str | Path, message: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {message}\n")


def write_event_json(path: str | Path, payload: dict[str, Any]) -> None:
    data = {"time_utc": utc_now(), **payload}
    write_json(path, data)

