from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

from ablation_utils import generation_rows_metrics, read_jsonl, sanitize, write_json


def maybe_bertscore(rows: Sequence[Dict[str, Any]], enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"bert_score_f1": None, "bert_score_note": "not requested; pass --bertscore to compute if bert_score is installed"}
    try:
        from bert_score import score  # type: ignore
    except Exception as exc:
        return {"bert_score_f1": None, "bert_score_note": f"bert_score unavailable: {exc}"}
    candidates = [str(r.get("generated_response", "")) for r in rows]
    references = [str(r.get("reference_response", "")) for r in rows]
    if not candidates:
        return {"bert_score_f1": None, "bert_score_note": "no rows"}
    _, _, f1 = score(candidates, references, lang="en", verbose=False)
    return {"bert_score_f1": float(f1.mean().item()), "bert_score_note": "computed with bert_score.score(lang='en')"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute generation stance/text diagnostics from scored JSONL.")
    parser.add_argument("--input", required=True, help="Scored generation JSONL.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--control-type", default=None, help="Optional control_type filter.")
    parser.add_argument("--selection-type", default=None, help="Optional selection_type filter.")
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore-F1 if the optional bert_score package is installed.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.control_type:
        rows = [r for r in rows if r.get("control_type") == args.control_type]
    if args.selection_type:
        rows = [r for r in rows if r.get("selection_type") == args.selection_type]
    metrics = generation_rows_metrics(rows)
    metrics.update(maybe_bertscore(rows, args.bertscore))
    payload = {
        "input": str(Path(args.input)),
        "control_type": args.control_type,
        "selection_type": args.selection_type,
        "num_rows": len(rows),
        "metrics": metrics,
    }
    write_json(args.out, sanitize(payload))


if __name__ == "__main__":
    main()
