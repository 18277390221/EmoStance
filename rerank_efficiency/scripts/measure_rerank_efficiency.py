from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TimingResult:
    configuration: str
    candidates: int
    num_examples: int
    total_latency_s: float
    generation_s: float
    scoring_s: float
    selection_s: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    samples_per_s: float
    candidates_per_s: float
    relative_cost: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure latency/throughput for EmoStance no-rerank B=1 and "
            "stance-consistency rerank B=4 using existing project runners."
        )
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-dir", default="rerank_efficiency")
    parser.add_argument("--prepared", default="runs/main/prepared")
    parser.add_argument(
        "--predicted-prepared",
        default="runs/main/prepared_predicted_control_c7_gated_t075_mix050",
    )
    parser.add_argument("--generator-dir", default="runs/main/generator_gold")
    parser.add_argument("--stance-dir", default="runs/main/stance_fp32")
    parser.add_argument("--model", default=None, help="Optional base model override.")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--num-examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--warmup-examples", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--score-max-length", type=int, default=384)
    parser.add_argument("--min-words", type=int, default=0)
    parser.add_argument("--length-penalty", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution for debugging. Do not use CPU timings in the paper.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "results": out_dir / "results",
        "reports": out_dir / "reports",
        "logs": out_dir / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def append_command(log_path: Path, argv: List[str]) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] ")
        f.write(" ".join(argv) + "\n")


def run_cmd(cmd: List[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def import_project_modules(project_root: Path):
    sys.path.insert(0, str(project_root))
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import transformers  # type: ignore
        from latent_stance_control.data import load_prepared_split
        from latent_stance_control.generate_and_rerank import attach_candidate_scores, select_reranked
        from latent_stance_control.generate_and_score_controls import (
            aligned_predicted_rows,
            generate_one,
            load_generator,
            load_stance_scorer,
            prompt_text,
            score_records,
            select_rows,
            unload_generator,
        )
        transformers.utils.logging.set_verbosity_error()
    except Exception as exc:
        raise RuntimeError(
            "Failed to import required project dependencies. This experiment "
            "requires torch, transformers, numpy, and the existing "
            "latent_stance_control package. Install them in the paper GPU "
            "environment before running the full benchmark."
        ) from exc
    return {
        "np": np,
        "torch": torch,
        "load_prepared_split": load_prepared_split,
        "attach_candidate_scores": attach_candidate_scores,
        "select_reranked": select_reranked,
        "aligned_predicted_rows": aligned_predicted_rows,
        "generate_one": generate_one,
        "load_generator": load_generator,
        "load_stance_scorer": load_stance_scorer,
        "prompt_text": prompt_text,
        "score_records": score_records,
        "select_rows": select_rows,
        "unload_generator": unload_generator,
    }


def collect_environment(project_root: Path, torch_mod: Any = None) -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "project_root": str(project_root.resolve()),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"], project_root),
        "git_status_short": run_cmd(["git", "status", "--short"], project_root),
    }
    try:
        import transformers  # type: ignore

        env["transformers_version"] = transformers.__version__
    except Exception as exc:
        env["transformers_import_error"] = repr(exc)
    if torch_mod is None:
        try:
            import torch as torch_mod  # type: ignore
        except Exception as exc:
            env["torch_import_error"] = repr(exc)
            return env
    env["torch_version"] = getattr(torch_mod, "__version__", None)
    env["cuda_available"] = bool(torch_mod.cuda.is_available())
    if torch_mod.cuda.is_available():
        index = torch_mod.cuda.current_device()
        env["cuda_device_index"] = int(index)
        env["cuda_device_name"] = torch_mod.cuda.get_device_name(index)
        props = torch_mod.cuda.get_device_properties(index)
        env["cuda_total_memory_gb"] = round(float(props.total_memory) / (1024**3), 3)
    return env


def set_seed(seed: int, np_mod: Any, torch_mod: Any) -> None:
    random.seed(seed)
    np_mod.random.seed(seed)
    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)


def sync_cuda(torch_mod: Any, device: str) -> None:
    if "cuda" in device and torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def make_runtime_args(args: argparse.Namespace, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        device=device,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        score_batch_size=args.score_batch_size,
        score_max_length=args.score_max_length,
        min_words=args.min_words,
        length_penalty=args.length_penalty,
        progress_every=args.progress_every,
    )


def make_candidate_record(
    split: str,
    example_index: int,
    candidate_index: int,
    gold_row: Dict[str, Any],
    pred_row: Dict[str, Any],
    generated: str,
    control_cluster: List[float],
    gold_cluster: List[float],
    prompt_text_fn: Any,
    top1_fn: Any,
) -> Dict[str, Any]:
    return {
        "split": split,
        "example_index": example_index,
        "candidate_index": candidate_index,
        "dialogue_id": gold_row.get("dialogue_id"),
        "turn_id": gold_row.get("turn_id"),
        "role": gold_row.get("role"),
        "next_role": gold_row.get("next_role"),
        "transition": gold_row.get("transition"),
        "situation": gold_row.get("situation", ""),
        "context": gold_row.get("context", ""),
        "prompt": prompt_text_fn(gold_row),
        "reference_response": gold_row.get("response", ""),
        "generated_response": generated,
        "gold_target_cluster": gold_cluster,
        "gold_target_top1": top1_fn(gold_cluster),
        "control_target_cluster": control_cluster,
        "control_target_top1": top1_fn(control_cluster),
        "control_source": "predicted",
        "control_source_dialogue_id": pred_row.get("dialogue_id"),
        "control_source_turn_id": pred_row.get("turn_id"),
    }


def prepare_rows(args: argparse.Namespace, modules: Dict[str, Any]) -> Tuple[List[dict], List[dict]]:
    project_root = Path(args.project_root).resolve()
    prepared = resolve_path(project_root, args.prepared)
    predicted_prepared = resolve_path(project_root, args.predicted_prepared)
    load_prepared_split = modules["load_prepared_split"]
    select_rows = modules["select_rows"]
    aligned_predicted_rows = modules["aligned_predicted_rows"]

    gold_all = load_prepared_split(prepared, args.split)
    pred_all = load_prepared_split(predicted_prepared, args.split)
    gold_rows, indices = select_rows(gold_all, args.num_examples, args.seed, args.sample_strategy)
    pred_rows = aligned_predicted_rows(gold_rows, pred_all, indices)
    if len(gold_rows) != args.num_examples:
        raise ValueError(f"Expected {args.num_examples} examples, found {len(gold_rows)}")
    return gold_rows, pred_rows


def warmup_generation(
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    modules: Dict[str, Any],
    lm: Any,
    tokenizer: Any,
    projector: Any,
    prefix_tokens: int,
    gold_rows: List[dict],
    pred_rows: List[dict],
) -> None:
    generate_one = modules["generate_one"]
    np_mod = modules["np"]
    torch_mod = modules["torch"]
    n = min(max(args.warmup_examples, 0), len(gold_rows))
    if n <= 0:
        return
    print(f"Warmup: generating {n} example(s).", flush=True)
    for i in range(n):
        stance_vector = np_mod.asarray(pred_rows[i]["target_vector"], dtype=np_mod.float32)
        _ = generate_one(lm, tokenizer, projector, gold_rows[i], stance_vector, runtime_args, prefix_tokens)
    sync_cuda(torch_mod, runtime_args.device)


def generate_candidates_timed(
    *,
    config_name: str,
    num_candidates: int,
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    modules: Dict[str, Any],
    lm: Any,
    tokenizer: Any,
    projector: Any,
    prefix_tokens: int,
    gold_rows: List[dict],
    pred_rows: List[dict],
) -> Tuple[List[dict], List[float], float]:
    np_mod = modules["np"]
    torch_mod = modules["torch"]
    generate_one = modules["generate_one"]
    prompt_text_fn = modules["prompt_text"]

    def top1(value: Any) -> int:
        return int(np_mod.argmax(np_mod.asarray(value, dtype=np_mod.float64)))

    records: List[dict] = []
    per_example_generation_s: List[float] = []
    total = len(gold_rows) * num_candidates
    done = 0
    sync_cuda(torch_mod, runtime_args.device)
    generation_start = time.perf_counter()
    for example_index, (gold_row, pred_row) in enumerate(zip(gold_rows, pred_rows)):
        if "target_vector" not in pred_row or "target_cluster" not in pred_row:
            raise KeyError(f"{args.split} example {example_index} missing predicted target_vector/target_cluster")
        stance_vector = np_mod.asarray(pred_row["target_vector"], dtype=np_mod.float32)
        control_cluster = np_mod.asarray(pred_row["target_cluster"], dtype=np_mod.float32).astype(float).tolist()
        gold_cluster = np_mod.asarray(gold_row["target_cluster"], dtype=np_mod.float32).astype(float).tolist()

        sync_cuda(torch_mod, runtime_args.device)
        example_start = time.perf_counter()
        for candidate_index in range(num_candidates):
            generated = generate_one(lm, tokenizer, projector, gold_row, stance_vector, runtime_args, prefix_tokens)
            records.append(
                make_candidate_record(
                    args.split,
                    example_index,
                    candidate_index,
                    gold_row,
                    pred_row,
                    generated,
                    control_cluster,
                    gold_cluster,
                    prompt_text_fn,
                    top1,
                )
            )
            done += 1
        sync_cuda(torch_mod, runtime_args.device)
        per_example_generation_s.append(time.perf_counter() - example_start)
        if args.progress_every and (example_index + 1) % args.progress_every == 0:
            print(
                f"{config_name}: generated {done}/{total} candidates "
                f"for {example_index + 1}/{len(gold_rows)} examples.",
                flush=True,
            )
    sync_cuda(torch_mod, runtime_args.device)
    generation_s = time.perf_counter() - generation_start
    return records, per_example_generation_s, generation_s


def score_and_select_timed(
    *,
    args: argparse.Namespace,
    runtime_args: SimpleNamespace,
    modules: Dict[str, Any],
    records: List[dict],
    stance_model: Any,
    stance_tokenizer: Any,
    selection_type: str,
) -> Tuple[List[dict], float, float]:
    torch_mod = modules["torch"]
    score_records = modules["score_records"]
    attach_candidate_scores = modules["attach_candidate_scores"]
    select_reranked = modules["select_reranked"]

    sync_cuda(torch_mod, runtime_args.device)
    score_start = time.perf_counter()
    pred_cluster = score_records(records, stance_model, stance_tokenizer, runtime_args)
    sync_cuda(torch_mod, runtime_args.device)
    scoring_s = time.perf_counter() - score_start

    selection_start = time.perf_counter()
    attach_candidate_scores(records, pred_cluster, runtime_args)
    selected_all = select_reranked(records)
    selected = [row for row in selected_all if row.get("selection_type") == selection_type]
    selection_s = time.perf_counter() - selection_start
    return selected, scoring_s, selection_s


def summarize_timing(
    configuration: str,
    candidates: int,
    n: int,
    generation_s: float,
    scoring_s: float,
    selection_s: float,
    per_example_generation_s: List[float],
    baseline_total_s: Optional[float] = None,
) -> TimingResult:
    total_s = generation_s + scoring_s + selection_s
    overhead_per_example = (scoring_s + selection_s) / n if n else 0.0
    per_example_total_s = [value + overhead_per_example for value in per_example_generation_s]
    mean_latency_ms = (total_s / n * 1000.0) if n else 0.0
    p50_latency_ms = percentile(per_example_total_s, 0.50) * 1000.0
    p95_latency_ms = percentile(per_example_total_s, 0.95) * 1000.0
    samples_per_s = n / total_s if total_s > 0 else 0.0
    candidates_per_s = (n * candidates) / total_s if total_s > 0 else 0.0
    relative_cost = total_s / baseline_total_s if baseline_total_s else 1.0
    return TimingResult(
        configuration=configuration,
        candidates=candidates,
        num_examples=n,
        total_latency_s=total_s,
        generation_s=generation_s,
        scoring_s=scoring_s,
        selection_s=selection_s,
        mean_latency_ms=mean_latency_ms,
        p50_latency_ms=p50_latency_ms,
        p95_latency_ms=p95_latency_ms,
        samples_per_s=samples_per_s,
        candidates_per_s=candidates_per_s,
        relative_cost=relative_cost,
    )


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_tradeoff_human_result(project_root: Path) -> Dict[str, Any]:
    path = project_root / "human_ablation_combined20/results/ablation_human_results_by_comparison.csv"
    if not path.exists():
        return {
            "available": False,
            "note": "Combined human ablation CSV not found.",
        }
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("comparison") == "final_vs_no_rerank":
                return {
                    "available": True,
                    "comparison": row.get("comparison"),
                    "decisive_win_rate": float(row.get("decisive_win_rate", "nan")),
                    "p_value": float(row.get("p_value") or row.get("sign_test_p") or "nan"),
                    "final_win": int(float(row.get("final_win", 0))),
                    "final_lose": int(float(row.get("final_lose", 0))),
                    "tie": int(float(row.get("tie", 0))),
                    "neither": int(float(row.get("neither", 0))),
                }
    return {"available": False, "note": "final_vs_no_rerank row not found."}


def write_reports(
    out_dir: Path,
    args: argparse.Namespace,
    results: List[TimingResult],
    environment: Dict[str, Any],
    human_tradeoff: Dict[str, Any],
) -> None:
    reports = out_dir / "reports"
    rows = [asdict(item) for item in results]

    md_lines = [
        "# Reranking Efficiency Experiment",
        "",
        "## Setup",
        "",
        f"- Split: `{args.split}`",
        f"- Examples: `{args.num_examples}`",
        f"- Seed: `{args.seed}`",
        f"- Sampling strategy: `{args.sample_strategy}`",
        f"- Max new tokens: `{args.max_new_tokens}`",
        f"- Decoding: do_sample={args.do_sample}, temperature={args.temperature}, top_p={args.top_p}",
        f"- Score batch size: `{args.score_batch_size}`",
        f"- Device: `{environment.get('cuda_device_name', args.device)}`",
        "",
        "Timings include tokenization inside generation. The reranking row also includes stance scoring and selection.",
        "",
        "## Results",
        "",
        "| Configuration | Candidates | Mean latency (ms) | P50/P95 (ms) | Samples/s | Relative cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {configuration} | {candidates} | {mean_latency_ms:.1f} | {p50_latency_ms:.1f}/{p95_latency_ms:.1f} | "
            "{samples_per_s:.3f} | {relative_cost:.2f}x |".format(**row)
        )
    md_lines += [
        "",
        "## Component Timing",
        "",
        "| Configuration | Generation (s) | Stance scoring (s) | Selection (s) | Total (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {configuration} | {generation_s:.3f} | {scoring_s:.3f} | {selection_s:.3f} | {total_latency_s:.3f} |".format(
                **row
            )
        )
    md_lines += ["", "## Quality-Cost Context", ""]
    if human_tradeoff.get("available"):
        rate = human_tradeoff["decisive_win_rate"] * 100.0
        p_value = human_tradeoff["p_value"]
        p_text = "< .001" if p_value < 0.001 else f"= {p_value:.3f}"
        md_lines.append(
            f"The combined human ablation shows final EmoStance vs. no-rerank at {rate:.1f}% decisive win rate "
            f"(two-sided sign test p {p_text}). This supports reporting no-rerank as the low-latency option "
            "and B=4 reranking as the quality-prioritizing option."
        )
    else:
        md_lines.append(str(human_tradeoff.get("note", "Human ablation trade-off row unavailable.")))
    md_lines += [
        "",
        "## Reporting Note",
        "",
        "These numbers are system efficiency diagnostics. They should be interpreted together with the human ablation, "
        "not as an automatic proxy for human preference.",
        "",
    ]
    (reports / "efficiency_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    latex_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Configuration & Cand. & Mean latency & P50/P95 & Samples/s & Rel. cost \\\\",
        "\\midrule",
    ]
    for row in rows:
        latex_lines.append(
            "{configuration} & {candidates} & {mean_latency_ms:.1f} ms & "
            "{p50_latency_ms:.1f}/{p95_latency_ms:.1f} ms & {samples_per_s:.3f} & {relative_cost:.2f}$\\times$ \\\\".format(
                **row
            )
        )
    latex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Inference efficiency of stance-consistency reranking on 512 ED test examples. "
        "The no-rerank setting generates one response, while reranking samples four candidates and includes "
        "stance scoring and selection. Latency includes tokenization and generation; reranking additionally "
        "includes scorer and selection overhead.}",
        "\\label{tab:rerank-efficiency}",
        "\\end{table}",
        "",
    ]
    (reports / "efficiency_table.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = resolve_path(project_root, args.out_dir)
    dirs = ensure_dirs(out_dir)
    append_command(dirs["logs"] / "commands.log", sys.argv)

    modules = import_project_modules(project_root)
    np_mod = modules["np"]
    torch_mod = modules["torch"]

    if args.device == "auto":
        device = "cuda" if torch_mod.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is required for paper latency numbers. This environment has no CUDA device. "
            "Run on the RTX 4090 machine, or pass --allow-cpu only for a non-reportable debug run."
        )

    args.project_root = str(project_root)
    runtime_args = make_runtime_args(args, device)
    set_seed(args.seed, np_mod, torch_mod)
    environment = collect_environment(project_root, torch_mod)
    environment["resolved_device"] = device
    environment["warning"] = None if device != "cpu" else "CPU debug run; do not report as paper latency."
    write_json(dirs["logs"] / "environment.json", environment)

    gold_rows, pred_rows = prepare_rows(args, modules)

    load_generator = modules["load_generator"]
    unload_generator = modules["unload_generator"]
    load_stance_scorer = modules["load_stance_scorer"]

    generator_dir = resolve_path(project_root, args.generator_dir)
    stance_dir = resolve_path(project_root, args.stance_dir)
    prepared = resolve_path(project_root, args.prepared)

    print("Loading generator...", flush=True)
    lm, tokenizer, projector, prefix_tokens, vector_dim, base_model = load_generator(
        generator_dir, args.model, args.bf16, device
    )
    warmup_generation(args, runtime_args, modules, lm, tokenizer, projector, prefix_tokens, gold_rows, pred_rows)

    set_seed(args.seed, np_mod, torch_mod)
    print("Measuring no_rerank_b1...", flush=True)
    no_records, no_gen_latencies, no_generation_s = generate_candidates_timed(
        config_name="no_rerank_b1",
        num_candidates=1,
        args=args,
        runtime_args=runtime_args,
        modules=modules,
        lm=lm,
        tokenizer=tokenizer,
        projector=projector,
        prefix_tokens=prefix_tokens,
        gold_rows=gold_rows,
        pred_rows=pred_rows,
    )
    no_selected = []
    for record in no_records:
        row = dict(record)
        row["selection_type"] = "raw_first"
        row["selected_candidate_index"] = int(record["candidate_index"])
        row["candidate_count"] = 1
        no_selected.append(row)
    no_summary = summarize_timing(
        "No reranking",
        1,
        len(gold_rows),
        no_generation_s,
        0.0,
        0.0,
        no_gen_latencies,
    )

    set_seed(args.seed, np_mod, torch_mod)
    print("Measuring rerank_b4 generation...", flush=True)
    rerank_records, rerank_gen_latencies, rerank_generation_s = generate_candidates_timed(
        config_name="rerank_b4",
        num_candidates=4,
        args=args,
        runtime_args=runtime_args,
        modules=modules,
        lm=lm,
        tokenizer=tokenizer,
        projector=projector,
        prefix_tokens=prefix_tokens,
        gold_rows=gold_rows,
        pred_rows=pred_rows,
    )

    unload_generator(lm, projector)
    print("Loading stance scorer...", flush=True)
    stance_model, stance_tokenizer = load_stance_scorer(stance_dir, prepared, device)
    print("Measuring rerank_b4 stance scoring and selection...", flush=True)
    rerank_selected, rerank_scoring_s, rerank_selection_s = score_and_select_timed(
        args=args,
        runtime_args=runtime_args,
        modules=modules,
        records=rerank_records,
        stance_model=stance_model,
        stance_tokenizer=stance_tokenizer,
        selection_type="rerank_control",
    )
    rerank_summary = summarize_timing(
        "Reranking",
        4,
        len(gold_rows),
        rerank_generation_s,
        rerank_scoring_s,
        rerank_selection_s,
        rerank_gen_latencies,
        baseline_total_s=no_summary.total_latency_s,
    )

    results = [no_summary, rerank_summary]
    summary_rows = [asdict(item) for item in results]
    summary_fields = list(summary_rows[0].keys())
    write_csv(dirs["results"] / "latency_summary.csv", summary_rows, summary_fields)
    write_json(dirs["results"] / "latency_summary.json", {"results": summary_rows, "environment": environment})

    component_rows = [
        {
            "configuration": item.configuration,
            "generation_s": item.generation_s,
            "scoring_s": item.scoring_s,
            "selection_s": item.selection_s,
            "total_latency_s": item.total_latency_s,
        }
        for item in results
    ]
    write_csv(
        dirs["results"] / "component_times.csv",
        component_rows,
        ["configuration", "generation_s", "scoring_s", "selection_s", "total_latency_s"],
    )
    write_jsonl(dirs["results"] / "no_rerank_b1_selected.jsonl", no_selected)
    write_jsonl(dirs["results"] / "rerank_b4_candidates.scored.jsonl", rerank_records)
    write_jsonl(dirs["results"] / "rerank_b4_selected.jsonl", rerank_selected)

    run_metadata = {
        "base_model": base_model,
        "vector_dim": vector_dim,
        "prefix_tokens": prefix_tokens,
        "args": vars(args),
        "runtime_device": device,
        "num_examples": len(gold_rows),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(dirs["results"] / "run_metadata.json", run_metadata)

    human_tradeoff = read_tradeoff_human_result(project_root)
    write_reports(out_dir, args, results, environment, human_tradeoff)

    print("\nDone.", flush=True)
    print("Reranking efficiency experiment completed.", flush=True)
    print("Outputs:", flush=True)
    print(f"- {dirs['results'] / 'latency_summary.csv'}", flush=True)
    print(f"- {dirs['results'] / 'component_times.csv'}", flush=True)
    print(f"- {dirs['reports'] / 'efficiency_report.md'}", flush=True)
    print(f"- {dirs['reports'] / 'efficiency_table.tex'}", flush=True)


if __name__ == "__main__":
    main()
