from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Callable

from system_baseline.utils.io import read_jsonl, write_csv
from system_baseline.utils.metrics import meteor_score, rouge_l_f1, sentence_bleu

BOOTSTRAP_METRICS = ["bertscore_f1", "rouge_l", "bleu_2", "meteor"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired bootstrap tests over canonical example_id units.")
    parser.add_argument("--metric", default="bertscore_f1", choices=BOOTSTRAP_METRICS + ["all"])
    parser.add_argument("--generation_dir", default="system_baseline/outputs/generations")
    parser.add_argument("--per-example", default="system_baseline/outputs/metrics/per_example_metrics_full.jsonl")
    parser.add_argument("--output", default="system_baseline/outputs/metrics/pairwise_bootstrap.csv")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--emostance-id", default="emostance")
    return parser.parse_args()


def metric_fn(name: str) -> Callable[[str, str], float]:
    if name == "rouge_l":
        return rouge_l_f1
    if name == "bleu_2":
        return lambda pred, ref: sentence_bleu(pred, [ref], max_n=2)
    if name == "meteor":
        return meteor_score
    raise ValueError(name)


def load_per_example(path: Path, metric: str) -> dict[str, dict[str, float]]:
    rows = read_jsonl(path)
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        system_id = str(row.get("system_id") or "")
        example_id = str(row.get("example_id") or "")
        value = row.get(metric)
        if system_id and example_id and isinstance(value, (int, float)):
            out.setdefault(system_id, {})[example_id] = float(value)
    return out


def load_from_generations(generation_dir: Path, metric: str) -> dict[str, dict[str, float]]:
    fn = metric_fn(metric)
    out: dict[str, dict[str, float]] = {}
    for path in generation_dir.glob("*.jsonl"):
        system_id = path.stem
        for row in read_jsonl(path):
            example_id = str(row.get("example_id") or "")
            pred = str(row.get("prediction") or row.get("generated_response") or "")
            ref = str(row.get("reference") or "")
            if example_id and pred and ref:
                out.setdefault(system_id, {})[example_id] = fn(pred, ref)
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_delta(
    a: dict[str, float],
    b: dict[str, float],
    samples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    ids = sorted(set(a) & set(b))
    if not ids:
        return 0.0, 0.0, 0.0, 1.0
    observed = mean([a[i] - b[i] for i in ids])
    rng = random.Random(seed)
    deltas: list[float] = []
    n = len(ids)
    for _ in range(samples):
        resampled = [ids[rng.randrange(n)] for _ in range(n)]
        deltas.append(mean([a[i] - b[i] for i in resampled]))
    deltas.sort()
    lo = deltas[int(0.025 * (samples - 1))]
    hi = deltas[int(0.975 * (samples - 1))]
    if observed >= 0:
        tail = sum(1 for d in deltas if d <= 0) / samples
    else:
        tail = sum(1 for d in deltas if d >= 0) / samples
    p_value = min(1.0, 2.0 * tail)
    return observed, lo, hi, p_value


def main() -> None:
    args = parse_args()
    per_example_path = Path(args.per_example)
    rows: list[dict[str, Any]] = []
    metrics = BOOTSTRAP_METRICS if args.metric == "all" else [args.metric]
    for metric in metrics:
        if per_example_path.exists():
            data = load_per_example(per_example_path, metric)
        elif metric == "bertscore_f1":
            data = {}
        else:
            data = load_from_generations(Path(args.generation_dir), metric)
        emostance_scores = data.get(args.emostance_id) or data.get("ours") or {}
        if not emostance_scores:
            continue
        for system_id, scores in sorted(data.items()):
            if system_id in {args.emostance_id, "ours"}:
                continue
            common = sorted(set(emostance_scores) & set(scores))
            if not common:
                continue
            delta, lo, hi, p = bootstrap_delta(emostance_scores, scores, args.samples, args.seed)
            rows.append(
                {
                    "baseline": system_id,
                    "comparison": f"{args.emostance_id}_minus_{system_id}",
                    "metric": metric,
                    "n": len(common),
                    "delta": delta,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value_two_sided": p,
                    "bootstrap_samples": args.samples,
                    "seed": args.seed,
                }
            )
    write_csv(args.output, rows, fieldnames=["baseline", "comparison", "metric", "n", "delta", "ci_low", "ci_high", "p_value_two_sided", "bootstrap_samples", "seed"])
    print(f"Wrote {args.output} ({len(rows)} comparisons)")


if __name__ == "__main__":
    main()
