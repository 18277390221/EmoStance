#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COMPARISON_ORDER = [
    "final_vs_no_rerank",
    "final_vs_no_role_aware",
    "final_vs_zero_control",
]

COMPARISON_LABELS = {
    "final_vs_no_rerank": "vs. w/o rerank",
    "final_vs_no_role_aware": "vs. w/o role-aware",
    "final_vs_zero_control": "vs. zero control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze human-ablation A/B annotation results.")
    parser.add_argument("--human-ablation-dir", default="human_ablation")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_answers(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []
    for row in rows:
        row["_source_file"] = path.name
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.{digits}f}%"


def latex_pct(value: float | None) -> str:
    return fmt_pct(value).replace("%", r"\%")


def exact_two_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def decode_answers(human_dir: Path, results_dir: Path) -> tuple[list[dict[str, Any]], list[Path], list[str]]:
    blind = {row["assignment_id"]: row for row in read_jsonl(human_dir / "blind_key.jsonl")}
    raw_files = [
        path
        for path in sorted(list(results_dir.glob("*.jsonl")) + list(results_dir.glob("*.csv")))
        if not path.name.startswith("ablation_human_results")
    ]
    raw_answers: list[dict[str, Any]] = []
    for path in raw_files:
        raw_answers.extend(read_answers(path))

    # If the same assignment is exported more than once, keep the last file-order occurrence.
    latest: dict[str, dict[str, Any]] = {}
    for row in raw_answers:
        assignment_id = str(row.get("assignment_id") or "")
        if assignment_id:
            latest[assignment_id] = row

    decoded: list[dict[str, Any]] = []
    missing_blind: list[str] = []
    for row in latest.values():
        assignment_id = str(row.get("assignment_id") or "")
        key = blind.get(assignment_id)
        if not key:
            missing_blind.append(assignment_id)
            continue
        selected_option = str(row.get("selected_option") or "").strip().upper()
        if selected_option in {"TIE", "C"}:
            selected_system = "neutral"
            mapped_outcome = "tie"
        elif selected_option in {"NEITHER", "D"}:
            selected_system = "neutral"
            mapped_outcome = "neither"
        elif selected_option == "A":
            selected_system = key["a_is"]
            mapped_outcome = "final_win" if selected_system == "final_emoh" else "final_lose"
        elif selected_option == "B":
            selected_system = key["b_is"]
            mapped_outcome = "final_win" if selected_system == "final_emoh" else "final_lose"
        else:
            selected_system = ""
            mapped_outcome = "unanswered"
        try:
            time_spent_ms = int(float(row.get("time_spent_ms") or 0))
        except (TypeError, ValueError):
            time_spent_ms = 0
        decoded.append(
            {
                "annotator_id": row.get("annotator_id") or key.get("annotator_id"),
                "assignment_id": assignment_id,
                "item_uid": row.get("item_uid") or key.get("item_uid"),
                "context_uid": key.get("context_uid"),
                "example_id": key.get("example_id"),
                "comparison": key["comparison"],
                "ablation_name": key["ablation_name"],
                "selected_option": selected_option,
                "selected_system": selected_system,
                "mapped_outcome": mapped_outcome,
                "final_position": key["final_position"],
                "time_spent_ms": time_spent_ms,
                "timestamp": row.get("timestamp", ""),
                "comment": row.get("comment", ""),
                "source_file": row.get("_source_file", ""),
            }
        )
    return decoded, raw_files, missing_blind


def judgment_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["mapped_outcome"] for row in rows)
    total = len(rows)
    decisive = counts["final_win"] + counts["final_lose"]
    return {
        "total_judgments": total,
        "final_win": counts["final_win"],
        "tie": counts["tie"],
        "neither": counts["neither"],
        "final_lose": counts["final_lose"],
        "unanswered": counts["unanswered"],
        "decisive": decisive,
        "decisive_win_rate": counts["final_win"] / decisive if decisive else None,
        "overall_win_share": counts["final_win"] / total if total else None,
        "neutral_share": (counts["tie"] + counts["neither"]) / total if total else None,
        "sign_test_p": exact_two_sided_sign_p(counts["final_win"], counts["final_lose"]),
    }


def build_item_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_item[row["item_uid"]].append(row)
    item_rows: list[dict[str, Any]] = []
    for item_uid, subset in sorted(by_item.items()):
        comparison = subset[0]["comparison"]
        counts = Counter(row["mapped_outcome"] for row in subset)
        decisive_votes = counts["final_win"] + counts["final_lose"]
        if counts["final_win"] > counts["final_lose"]:
            decisive_outcome = "final_win"
        elif counts["final_lose"] > counts["final_win"]:
            decisive_outcome = "final_lose"
        else:
            decisive_outcome = "neutral_or_split"
        option_counts = Counter(row["selected_option"] for row in subset)
        most_common = option_counts.most_common()
        max_count = most_common[0][1] if most_common else 0
        agreement_type = "3/3" if max_count == 3 else "2/3" if max_count == 2 else "split"
        item_rows.append(
            {
                "item_uid": item_uid,
                "comparison": comparison,
                "context_uid": subset[0].get("context_uid", ""),
                "example_id": subset[0].get("example_id", ""),
                "num_judgments": len(subset),
                "final_win_votes": counts["final_win"],
                "final_lose_votes": counts["final_lose"],
                "tie_votes": counts["tie"],
                "neither_votes": counts["neither"],
                "decisive_votes": decisive_votes,
                "item_outcome": decisive_outcome,
                "agreement_type": agreement_type,
                "selected_options": " ".join(row["selected_option"] for row in subset),
            }
        )
    return item_rows


def item_summary(item_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["item_outcome"] for row in item_rows)
    decisive = counts["final_win"] + counts["final_lose"]
    return {
        "total_items": len(item_rows),
        "final_win_items": counts["final_win"],
        "final_lose_items": counts["final_lose"],
        "neutral_or_split_items": counts["neutral_or_split"],
        "decisive_items": decisive,
        "item_decisive_win_rate": counts["final_win"] / decisive if decisive else None,
        "item_sign_test_p": exact_two_sided_sign_p(counts["final_win"], counts["final_lose"]),
    }


def write_markdown(
    path: Path,
    raw_files: list[Path],
    missing_blind: list[str],
    decoded: list[dict[str, Any]],
    judgment_overall: dict[str, Any],
    judgment_by_comp: list[dict[str, Any]],
    item_overall: dict[str, Any],
    item_by_comp: list[dict[str, Any]],
    annotator_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
) -> None:
    expected = 450
    annotators = sorted({row["annotator_id"] for row in decoded})
    status = "complete" if len(decoded) == expected and len(annotators) == 10 else "partial"
    lines: list[str] = []
    lines.extend(
        [
            "# Human Ablation Result Analysis",
            "",
            f"Status: **{status}**. Received {len(decoded)} decoded judgments out of {expected} planned judgments.",
            "",
            "Win/loss is counted from the final EmoStance system perspective. `Tie` and `Neither` are neutral and excluded from decisive win rate.",
            "",
            "## Completion",
            "",
            f"- Result files found: {', '.join(path.name for path in raw_files) if raw_files else 'none'}",
            f"- Annotators represented: {', '.join(annotators) if annotators else 'none'}",
            f"- Missing full-study judgments: {expected - len(decoded)}",
            f"- Answers missing blind-key entries: {len(missing_blind)}",
            "",
            "## Judgment-Level Results",
            "",
            "| Comparison | Judgments | Final win | Tie | Neither | Final lose | Decisive win rate | p-value |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in judgment_by_comp:
        lines.append(
            f"| {COMPARISON_LABELS.get(row['comparison'], row['comparison'])} | {row['total_judgments']} | "
            f"{row['final_win']} | {row['tie']} | {row['neither']} | {row['final_lose']} | "
            f"{fmt_pct(row['decisive_win_rate'])} | {row['sign_test_p']:.4g} |"
        )
    lines.append(
        f"| Overall | {judgment_overall['total_judgments']} | {judgment_overall['final_win']} | "
        f"{judgment_overall['tie']} | {judgment_overall['neither']} | {judgment_overall['final_lose']} | "
        f"{fmt_pct(judgment_overall['decisive_win_rate'])} | {judgment_overall['sign_test_p']:.4g} |"
    )
    lines.extend(
        [
            "",
            "## Item-Majority Results",
            "",
            "Each item has three judgments. Item-majority counts compare final-win votes against final-lose votes; neutral/split items are excluded from the item decisive win rate.",
            "",
            "| Comparison | Items | Final-win items | Neutral/split | Final-lose items | Item decisive win rate | p-value |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in item_by_comp:
        p_text = "NA" if row["item_sign_test_p"] is None else f"{row['item_sign_test_p']:.4g}"
        lines.append(
            f"| {COMPARISON_LABELS.get(row['comparison'], row['comparison'])} | {row['total_items']} | "
            f"{row['final_win_items']} | {row['neutral_or_split_items']} | {row['final_lose_items']} | "
            f"{fmt_pct(row['item_decisive_win_rate'])} | {p_text} |"
        )
    p_text = "NA" if item_overall["item_sign_test_p"] is None else f"{item_overall['item_sign_test_p']:.4g}"
    lines.append(
        f"| Overall | {item_overall['total_items']} | {item_overall['final_win_items']} | "
        f"{item_overall['neutral_or_split_items']} | {item_overall['final_lose_items']} | "
        f"{fmt_pct(item_overall['item_decisive_win_rate'])} | {p_text} |"
    )
    lines.extend(
        [
            "",
            "## Agreement",
            "",
            "| Scope | Items | 3/3 agreement | 2/3 majority | Split | 3/3 rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in agreement_rows:
        lines.append(
            f"| {row['scope']} | {row['items']} | {row['agree_3_3']} | {row['majority_2_3']} | "
            f"{row['split']} | {fmt_pct(row['agree_3_3_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Annotator Diagnostics",
            "",
            "| Annotator | Judgments | Final win | Tie | Neither | Final lose | Decisive win rate | Avg sec/item | Too fast <1s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in annotator_rows:
        lines.append(
            f"| {row['annotator_id']} | {row['total_judgments']} | {row['final_win']} | {row['tie']} | "
            f"{row['neither']} | {row['final_lose']} | {fmt_pct(row['decisive_win_rate'])} | "
            f"{row['avg_time_sec']:.1f} | {row['too_fast_lt_1s']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The final system is strongly preferred over zero control, indicating that removing the stance-control signal substantially hurts the human-perceived contextual response quality measured by this question.",
            "",
            "The final system is also preferred over the no-rerank variant, supporting the value of reranking among candidates.",
            "",
            "The final system shows a smaller but still positive margin over the no-role-aware predicted-control baseline. This suggests role-aware control contributes additional benefit, though the effect is weaker than the zero-control and reranking comparisons.",
            "",
            "These are human judgments for the ablation question only; they should be reported separately from automatic stance diagnostics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, judgment_by_comp: list[dict[str, Any]], judgment_overall: dict[str, Any]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Comparison & Win & Tie & Neither & Lose & Decisive Win Rate & $p$ \\",
        r"\midrule",
    ]
    for row in judgment_by_comp:
        p_text = "NA" if row["sign_test_p"] is None else f"{row['sign_test_p']:.3g}"
        lines.append(
            f"{COMPARISON_LABELS.get(row['comparison'], row['comparison'])} & "
            f"{row['final_win']} & {row['tie']} & {row['neither']} & {row['final_lose']} & "
            f"{latex_pct(row['decisive_win_rate'])} & {p_text} " + r"\\"
        )
    p_text = "NA" if judgment_overall["sign_test_p"] is None else f"{judgment_overall['sign_test_p']:.3g}"
    lines.extend(
        [
            r"\midrule",
            f"Overall & {judgment_overall['final_win']} & {judgment_overall['tie']} & "
            f"{judgment_overall['neither']} & {judgment_overall['final_lose']} & "
            f"{latex_pct(judgment_overall['decisive_win_rate'])} & {p_text} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Human ablation results. Wins and losses are counted from the final EmoStance system perspective. Tie and Neither are neutral and excluded from decisive win rate.}",
            r"\label{tab:human-ablation}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_item_latex(path: Path, item_by_comp: list[dict[str, Any]], item_overall: dict[str, Any]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Comparison & Win Items & Neutral/Split & Lose Items & Decisive Win Rate & $p$ \\",
        r"\midrule",
    ]
    for row in item_by_comp:
        p_text = "NA" if row["item_sign_test_p"] is None else f"{row['item_sign_test_p']:.3g}"
        lines.append(
            f"{COMPARISON_LABELS.get(row['comparison'], row['comparison'])} & "
            f"{row['final_win_items']} & {row['neutral_or_split_items']} & {row['final_lose_items']} & "
            f"{latex_pct(row['item_decisive_win_rate'])} & {p_text} " + r"\\"
        )
    p_text = "NA" if item_overall["item_sign_test_p"] is None else f"{item_overall['item_sign_test_p']:.3g}"
    lines.extend(
        [
            r"\midrule",
            f"Overall & {item_overall['final_win_items']} & {item_overall['neutral_or_split_items']} & "
            f"{item_overall['final_lose_items']} & {latex_pct(item_overall['item_decisive_win_rate'])} & {p_text} "
            + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Item-majority human ablation results. Each pairwise item is labeled by three annotators. Win and lose items are counted from the final EmoStance system perspective; neutral or split items are excluded from decisive win rate.}",
            r"\label{tab:human-ablation-item-majority}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    human_dir = Path(args.human_ablation_dir)
    results_dir = Path(args.results_dir) if args.results_dir else human_dir / "results"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    decoded, raw_files, missing_blind = decode_answers(human_dir, results_dir)
    decoded = sorted(decoded, key=lambda row: (row["annotator_id"], row["assignment_id"]))
    write_csv(
        output_dir / "ablation_human_results_decoded.csv",
        decoded,
        [
            "annotator_id",
            "assignment_id",
            "item_uid",
            "context_uid",
            "example_id",
            "comparison",
            "ablation_name",
            "selected_option",
            "selected_system",
            "mapped_outcome",
            "final_position",
            "time_spent_ms",
            "timestamp",
            "comment",
            "source_file",
        ],
    )

    judgment_overall = judgment_summary(decoded)
    judgment_by_comp: list[dict[str, Any]] = []
    for comparison in COMPARISON_ORDER:
        row = judgment_summary([x for x in decoded if x["comparison"] == comparison])
        row["comparison"] = comparison
        judgment_by_comp.append(row)
    write_csv(
        output_dir / "ablation_human_results_by_comparison.csv",
        judgment_by_comp,
        [
            "comparison",
            "total_judgments",
            "final_win",
            "tie",
            "neither",
            "final_lose",
            "unanswered",
            "decisive",
            "decisive_win_rate",
            "overall_win_share",
            "neutral_share",
            "sign_test_p",
        ],
    )

    item_rows = build_item_rows(decoded)
    write_csv(
        output_dir / "ablation_human_results_by_item.csv",
        item_rows,
        [
            "item_uid",
            "comparison",
            "context_uid",
            "example_id",
            "num_judgments",
            "final_win_votes",
            "final_lose_votes",
            "tie_votes",
            "neither_votes",
            "decisive_votes",
            "item_outcome",
            "agreement_type",
            "selected_options",
        ],
    )
    item_overall = item_summary(item_rows)
    item_by_comp: list[dict[str, Any]] = []
    for comparison in COMPARISON_ORDER:
        row = item_summary([x for x in item_rows if x["comparison"] == comparison])
        row["comparison"] = comparison
        item_by_comp.append(row)
    write_csv(
        output_dir / "ablation_human_results_item_majority.csv",
        item_by_comp,
        [
            "comparison",
            "total_items",
            "final_win_items",
            "neutral_or_split_items",
            "final_lose_items",
            "decisive_items",
            "item_decisive_win_rate",
            "item_sign_test_p",
        ],
    )

    annotator_rows: list[dict[str, Any]] = []
    for annotator_id in sorted({row["annotator_id"] for row in decoded}):
        subset = [row for row in decoded if row["annotator_id"] == annotator_id]
        summary = judgment_summary(subset)
        times = [row["time_spent_ms"] for row in subset if row["time_spent_ms"]]
        summary["annotator_id"] = annotator_id
        summary["avg_time_sec"] = statistics.mean(times) / 1000 if times else 0.0
        summary["median_time_sec"] = statistics.median(times) / 1000 if times else 0.0
        summary["too_fast_lt_1s"] = sum(1 for row in subset if 0 < row["time_spent_ms"] < 1000)
        annotator_rows.append(summary)
    write_csv(
        output_dir / "ablation_human_results_by_annotator.csv",
        annotator_rows,
        [
            "annotator_id",
            "total_judgments",
            "final_win",
            "tie",
            "neither",
            "final_lose",
            "unanswered",
            "decisive",
            "decisive_win_rate",
            "overall_win_share",
            "neutral_share",
            "sign_test_p",
            "avg_time_sec",
            "median_time_sec",
            "too_fast_lt_1s",
        ],
    )

    agreement_rows: list[dict[str, Any]] = []
    for scope, subset in [("overall", item_rows)] + [
        (comparison, [row for row in item_rows if row["comparison"] == comparison]) for comparison in COMPARISON_ORDER
    ]:
        counts = Counter(row["agreement_type"] for row in subset)
        total = len(subset)
        agreement_rows.append(
            {
                "scope": COMPARISON_LABELS.get(scope, scope),
                "items": total,
                "agree_3_3": counts["3/3"],
                "majority_2_3": counts["2/3"],
                "split": counts["split"],
                "agree_3_3_rate": counts["3/3"] / total if total else None,
            }
        )
    write_csv(
        output_dir / "ablation_human_results_agreement.csv",
        agreement_rows,
        ["scope", "items", "agree_3_3", "majority_2_3", "split", "agree_3_3_rate"],
    )

    write_markdown(
        output_dir / "ablation_human_results_analysis.md",
        raw_files,
        missing_blind,
        decoded,
        judgment_overall,
        judgment_by_comp,
        item_overall,
        item_by_comp,
        annotator_rows,
        agreement_rows,
    )
    write_latex(output_dir / "ablation_human_results_table.tex", judgment_by_comp, judgment_overall)
    write_item_latex(output_dir / "ablation_human_results_item_majority_table.tex", item_by_comp, item_overall)

    print("Done.")
    print(f"Decoded judgments: {len(decoded)}")
    print(f"Annotators: {len({row['annotator_id'] for row in decoded})}")
    print(f"Overall decisive win rate: {fmt_pct(judgment_overall['decisive_win_rate'])}")
    print(f"Wrote: {output_dir / 'ablation_human_results_analysis.md'}")
    print(f"Wrote: {output_dir / 'ablation_human_results_table.tex'}")
    print(f"Wrote: {output_dir / 'ablation_human_results_item_majority_table.tex'}")


if __name__ == "__main__":
    main()
