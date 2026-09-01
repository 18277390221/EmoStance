from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


METRICS = ["BERTScore", "ROUGE-L", "BLEU-2", "METEOR", "Distinct-1", "Distinct-2", "Self-BLEU", "Generic"]
HIGHER_IS_BETTER = {
    "BERTScore": True,
    "ROUGE-L": True,
    "BLEU-2": True,
    "METEOR": True,
    "Distinct-1": True,
    "Distinct-2": True,
    "Self-BLEU": False,
    "Generic": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LaTeX tables from system_metrics_full.csv.")
    parser.add_argument("--metrics", default="system_baseline/outputs/metrics/system_metrics_full.csv")
    parser.add_argument("--output", default="system_baseline/outputs/metrics/system_metrics_table.tex")
    parser.add_argument("--allow_diagnostic_table", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latex_escape(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def best_values(rows: list[dict[str, str]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for metric in METRICS:
        vals = [to_float(row.get(metric)) for row in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        best[metric] = max(vals) if HIGHER_IS_BETTER[metric] else min(vals)
    return best


def fmt_metric(value: Any, metric: str, best: dict[str, float]) -> str:
    val = to_float(value)
    if val is None:
        return "--"
    text = f"{val:.4f}"
    if metric in best and abs(val - best[metric]) <= 1e-12:
        return r"\textbf{" + text + "}"
    return text


def validate_same_n(rows: list[dict[str, str]], allow: bool) -> None:
    ns = {row.get("N", "") for row in rows}
    if len(ns) > 1 and not allow:
        raise ValueError(f"Main table N values differ: {sorted(ns)}. Use --allow_diagnostic_table only for diagnostic tables.")


def render_table(rows: list[dict[str, str]]) -> str:
    best = best_values(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Method & N & BERTScore & ROUGE-L & BLEU-2 & METEOR & Distinct-1 & Distinct-2 & Self-BLEU & Generic \\",
        r"\midrule",
    ]
    if rows:
        for row in rows:
            cells = [latex_escape(row.get("Method", "")), str(row.get("N", ""))]
            cells.extend(fmt_metric(row.get(metric), metric, best) for metric in METRICS)
            lines.append(" & ".join(cells) + r" \\")
    else:
        lines.append(r"\multicolumn{10}{c}{No full-coverage systems have been evaluated yet.} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{System-level automatic evaluation on the same ED test inputs. All systems in this table are evaluated on the identical canonical test set using the same references and metric implementation. Reference-based metrics compare each generated response with the ED reference response. Distinct-1/2 and Self-BLEU measure surface diversity, while Generic measures the rate of template-like responses. Automatic metrics are used as diagnostics and are not substitutes for human evaluation.}",
        r"\label{tab:system-ed-automatic}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.metrics))
    validate_same_n(rows, args.allow_diagnostic_table)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_table(rows), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

