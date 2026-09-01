from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .data import write_json


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_choice_payload(payload: Any) -> List[dict]:
    if isinstance(payload, dict):
        if "choices" in payload:
            payload = payload["choices"]
        elif "data" in payload:
            payload = payload["data"]
        else:
            raise ValueError("choices JSON must be a list, or a dict containing 'choices'/'data'.")
    if not isinstance(payload, list):
        raise ValueError("choices JSON must contain a list of example choice objects.")
    return payload


def build_option_map(blind_key: dict) -> Tuple[Dict[Tuple[int, str], dict], Dict[str, dict]]:
    option_map: Dict[Tuple[int, str], dict] = {}
    for row in blind_key.get("options", []):
        option_map[(int(row["example_index"]), str(row["option"]))] = row
    tasks = {str(task["id"]): task for task in blind_key.get("tasks", [])}
    return option_map, tasks


def summarize(choices: List[dict], blind_key: dict) -> dict:
    option_map, tasks = build_option_map(blind_key)
    methods = sorted({row["method"] for row in blind_key.get("options", [])})
    task_counts: Dict[str, Counter] = defaultdict(Counter)
    task_totals: Counter = Counter()
    missing: List[dict] = []

    for item in choices:
        example_index = int(item.get("example_index", item.get("example", 1) - 1))
        for task in item.get("tasks", []):
            task_id = str(task.get("task_id"))
            selected = task.get("selected_option")
            if selected is None:
                missing.append({"example_index": example_index, "task_id": task_id, "reason": "not_selected"})
                continue
            option = option_map.get((example_index, str(selected)))
            if option is None:
                missing.append({"example_index": example_index, "task_id": task_id, "selected_option": selected, "reason": "option_not_found"})
                continue
            method = option["method"]
            task_counts[task_id][method] += 1
            task_totals[task_id] += 1

    task_summary = {}
    positive_total = Counter()
    negative_total = Counter()
    for task_id, counter in task_counts.items():
        total = max(task_totals[task_id], 1)
        task_info = tasks.get(task_id, {"id": task_id, "question": task_id, "polarity": "unknown"})
        rows = []
        for method in methods:
            count = int(counter.get(method, 0))
            rows.append({"method": method, "count": count, "rate": count / total})
            if task_info.get("polarity") == "negative":
                negative_total[method] += count
            else:
                positive_total[method] += count
        rows.sort(key=lambda row: (-row["count"], row["method"]))
        task_summary[task_id] = {
            "question": task_info.get("question", task_id),
            "polarity": task_info.get("polarity", "unknown"),
            "n_selected": int(task_totals[task_id]),
            "methods": rows,
        }

    positive_n = sum(positive_total.values()) or 1
    negative_n = sum(negative_total.values()) or 1
    aggregate = {
        "positive_tasks": [
            {"method": method, "count": int(positive_total.get(method, 0)), "rate": positive_total.get(method, 0) / positive_n}
            for method in methods
        ],
        "negative_tasks": [
            {"method": method, "count": int(negative_total.get(method, 0)), "rate": negative_total.get(method, 0) / negative_n}
            for method in methods
        ],
    }
    aggregate["positive_tasks"].sort(key=lambda row: (-row["count"], row["method"]))
    aggregate["negative_tasks"].sort(key=lambda row: (-row["count"], row["method"]))

    return {
        "num_examples": len(choices),
        "num_tasks": len(tasks),
        "tasks": task_summary,
        "aggregate": aggregate,
        "missing": missing,
    }


def format_rate(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_markdown(path: Path, summary: dict) -> None:
    lines: List[str] = []
    lines.append("# Human Choice Evaluation Summary")
    lines.append("")
    lines.append(f"Examples: {summary['num_examples']}")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("Positive tasks: higher is better.")
    lines.append("")
    lines.append("| method | count | rate |")
    lines.append("|---|---:|---:|")
    for row in summary["aggregate"]["positive_tasks"]:
        lines.append(f"| {row['method']} | {row['count']} | {format_rate(row['rate'])} |")
    lines.append("")
    lines.append("Negative tasks: lower is better, because this means fewer people judged the reply passive or AI-like.")
    lines.append("")
    lines.append("| method | count | rate |")
    lines.append("|---|---:|---:|")
    for row in summary["aggregate"]["negative_tasks"]:
        lines.append(f"| {row['method']} | {row['count']} | {format_rate(row['rate'])} |")
    lines.append("")
    lines.append("## By Task")
    for task_id, task in summary["tasks"].items():
        lines.append("")
        lines.append(f"### {task_id}")
        lines.append(task["question"])
        lines.append("")
        lines.append(f"Polarity: {task['polarity']}; selected: {task['n_selected']}")
        lines.append("")
        lines.append("| method | count | rate |")
        lines.append("|---|---:|---:|")
        for row in task["methods"]:
            lines.append(f"| {row['method']} | {row['count']} | {format_rate(row['rate'])} |")
    if summary.get("missing"):
        lines.append("")
        lines.append("## Missing / Invalid")
        lines.append("")
        lines.append(f"Missing or invalid selections: {len(summary['missing'])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exported blind four-choice human evaluation results.")
    parser.add_argument("--choices", required=True, help="Path to human_eval_choices.json exported from the HTML page.")
    parser.add_argument("--blind-key", required=True, help="Path to blind_key.json generated next to the HTML page.")
    parser.add_argument("--out", default="", help="Output directory. Defaults to the choices file directory.")
    args = parser.parse_args()

    choices_path = Path(args.choices)
    out = Path(args.out) if args.out else choices_path.parent
    out.mkdir(parents=True, exist_ok=True)
    choices = normalize_choice_payload(read_json(choices_path))
    blind_key = read_json(Path(args.blind_key))
    summary = summarize(choices, blind_key)
    write_json(out / "human_choice_summary.json", summary)
    write_markdown(out / "human_choice_summary.md", summary)
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"Wrote {out / 'human_choice_summary.md'}")


if __name__ == "__main__":
    main()
