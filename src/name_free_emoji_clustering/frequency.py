from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .discovery import Candidate


@dataclass
class FrequencyPoolSummary:
    sources_used: list[str]
    pool_size: int
    observed_size: int
    zero_frequency_size: int
    clustered_zero_frequency_emojis: list[str]
    clustered_not_in_frequency_pool: list[str]


def parse_bool(value: str) -> bool | None:
    text = value.strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def load_frequency_pool(
    candidates: list[Candidate],
    clustered_emojis: list[str],
    output_path: Path,
) -> FrequencyPoolSummary:
    pool: set[str] = set()
    observed: set[str] = set()
    sources_used: list[str] = []

    for candidate in candidates:
        path = candidate.paths[0]
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
                if "emoji" not in header or "count" not in header:
                    continue
                if "role" in header and "observed" not in header and "avg_confidence" not in header:
                    continue
                emoji_idx = header.index("emoji")
                count_idx = header.index("count")
                observed_idx = header.index("observed") if "observed" in header else None
                sources_used.append(str(path))
                for row in reader:
                    if len(row) <= max(emoji_idx, count_idx):
                        continue
                    emoji = row[emoji_idx]
                    pool.add(emoji)
                    try:
                        count = float(row[count_idx])
                    except ValueError:
                        count = 0.0
                    if observed_idx is not None and len(row) > observed_idx:
                        observed_flag = parse_bool(row[observed_idx])
                        if observed_flag is True or count > 0:
                            observed.add(emoji)
                    elif count > 0:
                        observed.add(emoji)
        except OSError:
            continue

    zero_frequency = pool - observed
    clustered = set(clustered_emojis)
    summary = FrequencyPoolSummary(
        sources_used=sources_used,
        pool_size=len(pool),
        observed_size=len(observed),
        zero_frequency_size=len(zero_frequency),
        clustered_zero_frequency_emojis=sorted(clustered & zero_frequency),
        clustered_not_in_frequency_pool=sorted(clustered - pool) if pool else [],
    )
    output_path.write_text(
        json.dumps(
            {
                "sources_used": summary.sources_used,
                "columns_read": ["emoji", "count", "observed"],
                "pool_size": summary.pool_size,
                "observed_size": summary.observed_size,
                "zero_frequency_size": summary.zero_frequency_size,
                "clustered_zero_frequency_emojis": summary.clustered_zero_frequency_emojis,
                "clustered_not_in_frequency_pool": summary.clustered_not_in_frequency_pool,
                "valid_no_zero_frequency_clustered": not summary.clustered_zero_frequency_emojis,
                "emoji_names_used": False,
                "aliases_used": False,
                "unicode_descriptions_used": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
