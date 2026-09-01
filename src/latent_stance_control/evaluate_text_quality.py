from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .data import write_json


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def read_jsonl(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text: Any) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text or ""))]


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / max(len(sa | sb), 1))


def recall_overlap(generated: Iterable[str], target: Iterable[str]) -> float:
    sg = set(generated)
    st = set(target)
    if not st:
        return 0.0
    return float(len(sg & st) / len(st))


def repeated_rate(items: Sequence[Tuple[str, ...] | str]) -> float:
    if not items:
        return 0.0
    return float((len(items) - len(set(items))) / len(items))


def corpus_distinct(all_items: List[Tuple[str, ...] | str]) -> float:
    if not all_items:
        return 0.0
    return float(len(set(all_items)) / len(all_items))


def group_name(row: dict, requested: str) -> str:
    if requested != "auto":
        return str(row.get(requested, "unknown"))
    if "selection_type" in row:
        return str(row.get("selection_type", "unknown"))
    if "control_type" in row:
        return str(row.get("control_type", "unknown"))
    return "all"


def add_source_metadata(rows: List[dict], path: Path) -> None:
    for row in rows:
        row["_source_file"] = str(path)


def load_inputs(paths: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for item in paths:
        path = Path(item)
        loaded = read_jsonl(path)
        add_source_metadata(loaded, path)
        rows.extend(loaded)
    return rows


def metric_block(rows: List[dict]) -> Dict[str, Any]:
    word_counts: List[int] = []
    char_counts: List[int] = []
    empty: List[int] = []
    all_unigrams: List[str] = []
    all_bigrams: List[Tuple[str, ...]] = []
    repeat_uni: List[float] = []
    repeat_bi: List[float] = []
    ref_jaccard: List[float] = []
    ctx_jaccard: List[float] = []
    ref_recall: List[float] = []
    ctx_recall: List[float] = []

    for row in rows:
        generated = str(row.get("generated_response", "") or "")
        gen_tokens = tokenize(generated)
        ref_tokens = tokenize(row.get("reference_response", ""))
        ctx_tokens = tokenize(row.get("context", ""))
        gen_bigrams = ngrams(gen_tokens, 2)

        word_counts.append(len(gen_tokens))
        char_counts.append(len(generated))
        empty.append(1 if len(gen_tokens) == 0 else 0)
        all_unigrams.extend(gen_tokens)
        all_bigrams.extend(gen_bigrams)
        repeat_uni.append(repeated_rate(gen_tokens))
        repeat_bi.append(repeated_rate(gen_bigrams))
        ref_jaccard.append(jaccard(gen_tokens, ref_tokens))
        ctx_jaccard.append(jaccard(gen_tokens, ctx_tokens))
        ref_recall.append(recall_overlap(gen_tokens, ref_tokens))
        ctx_recall.append(recall_overlap(gen_tokens, ctx_tokens))

    return {
        "count": len(rows),
        "mean_words": float(np.mean(word_counts)) if word_counts else 0.0,
        "std_words": float(np.std(word_counts)) if word_counts else 0.0,
        "mean_chars": float(np.mean(char_counts)) if char_counts else 0.0,
        "empty_rate": float(np.mean(empty)) if empty else 0.0,
        "distinct_1": corpus_distinct(all_unigrams),
        "distinct_2": corpus_distinct(all_bigrams),
        "repeat_unigram_rate": float(np.mean(repeat_uni)) if repeat_uni else 0.0,
        "repeat_bigram_rate": float(np.mean(repeat_bi)) if repeat_bi else 0.0,
        "reference_lexical_overlap": float(np.mean(ref_jaccard)) if ref_jaccard else 0.0,
        "context_lexical_overlap": float(np.mean(ctx_jaccard)) if ctx_jaccard else 0.0,
        "reference_token_recall": float(np.mean(ref_recall)) if ref_recall else 0.0,
        "context_token_recall": float(np.mean(ctx_recall)) if ctx_recall else 0.0,
    }


def evaluate(rows: List[dict], group_field: str) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split", "unknown"))
        group = group_name(row, group_field)
        groups[(split, group)].append(row)

    by_split: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for (split, group), group_rows in sorted(groups.items()):
        by_split[split][group] = metric_block(group_rows)
    return {"by_split": dict(by_split), "overall": metric_block(rows)}


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "NA"


def write_report(path: Path, metrics: Dict[str, Any], args) -> None:
    lines = [
        "# Text Quality Evaluation",
        "",
        f"Inputs: `{', '.join(args.inputs)}`",
        f"Group field: `{args.group_field}`",
        "",
        "Metrics are lexical diagnostics, not human preference scores.",
        "",
    ]
    for split, groups in metrics["by_split"].items():
        lines += [
            f"## {split}",
            "",
            "| group | count | words | empty | distinct-1 | distinct-2 | repeat-1 | repeat-2 | ref overlap | ctx overlap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for group, block in groups.items():
            lines.append(
                f"| {group} | {int(block['count'])} | {fmt(block['mean_words'])} | {fmt(block['empty_rate'])} | "
                f"{fmt(block['distinct_1'])} | {fmt(block['distinct_2'])} | {fmt(block['repeat_unigram_rate'])} | "
                f"{fmt(block['repeat_bigram_rate'])} | {fmt(block['reference_lexical_overlap'])} | "
                f"{fmt(block['context_lexical_overlap'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lightweight lexical quality metrics for generated responses.")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSONL generation/selection files.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-field", default="auto", help="Use selection_type/control_type automatically, or provide a field name.")
    args = parser.parse_args()

    rows = load_inputs(args.inputs)
    metrics = evaluate(rows, args.group_field)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metrics.json", {"inputs": args.inputs, "group_field": args.group_field, "metrics": metrics})
    write_report(out / "report.md", metrics, args)


if __name__ == "__main__":
    main()
