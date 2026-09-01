from __future__ import annotations

import argparse
import importlib
import json
import traceback
from pathlib import Path
from typing import Any

from system_baseline.adapters.base import BaseSystemAdapter
from system_baseline.utils.io import PROJECT_ROOT, load_yaml, read_jsonl, relpath, write_json, write_jsonl
from system_baseline.utils.logging import append_log, utc_now


ADAPTERS: dict[str, tuple[str, str]] = {
    "emostance": ("system_baseline.adapters.emostance", "EmoStanceAdapter"),
    "ours": ("system_baseline.adapters.emostance", "EmoStanceAdapter"),
    "llm_only": ("system_baseline.adapters.llm_only", "LLMOnlyAdapter"),
    "llm_prompt": ("system_baseline.adapters.llm_prompt", "LLMPromptAdapter"),
    "llm_sft": ("system_baseline.adapters.llm_sft", "LLMSFTAdapter"),
    "empo_dpo": ("system_baseline.adapters.empo_dpo", "EmPODPOAdapter"),
    "case": ("system_baseline.adapters.case", "CASEAdapter"),
    "aptness": ("system_baseline.adapters.aptness", "APTNESSAdapter"),
    "sibyl": ("system_baseline.adapters.sibyl", "SibylAdapter"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate system outputs on the canonical ED test set.")
    parser.add_argument("--config", default="system_baseline/configs/system_eval.yaml")
    parser.add_argument("--systems", default=None, help="Comma-separated system ids. Defaults to config order.")
    parser.add_argument("--input", default="system_baseline/data/ed_test_canonical.jsonl")
    parser.add_argument("--output_dir", default="system_baseline/outputs/generations")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--execute-full", action="store_true", help="Actually run full model inference even if config generation.execute_full is false.")
    parser.add_argument("--smoke-test", action="store_true", help="Run only the first --smoke-n examples and mark output partial.")
    parser.add_argument("--smoke-n", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_adapter(system_id: str, project_root: Path, system_cfg: dict[str, Any], generation_cfg: dict[str, Any]) -> BaseSystemAdapter:
    module_name, class_name = ADAPTERS[system_id]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    cfg = dict(system_cfg)
    cfg["generation"] = generation_cfg
    return cls(project_root, cfg)


def configured_systems(cfg: dict[str, Any], explicit: str | None) -> list[str]:
    if explicit:
        return [x.strip() for x in explicit.split(",") if x.strip()]
    order = cfg.get("system_order") or ["emostance", "llm_only", "llm_prompt", "llm_sft", "empo_dpo", "case", "aptness", "sibyl"]
    return [str(x) for x in order]


def output_path(output_dir: Path, system_id: str) -> Path:
    return output_dir / f"{system_id}.jsonl"


def existing_full_output(path: Path, expected_ids: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_jsonl(path)
    except Exception:
        return False
    ids = [str(row.get("example_id", "")) for row in rows]
    return len(ids) == len(expected_ids) and set(ids) == expected_ids and len(ids) == len(set(ids))


def generate_streaming(
    adapter: BaseSystemAdapter,
    examples: list[dict[str, Any]],
    final_path: Path,
    log_path: Path,
    progress_every: int,
) -> list[dict[str, Any]]:
    partial_path = final_path.with_suffix(".partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    batch_size = max(
        1,
        int(
            getattr(adapter, "generation_batch_size", 0)
            or adapter.config.get("generation_batch_size", 0)
            or adapter.decode_config.get("generation_batch_size", 1)
        ),
    )
    with partial_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            if batch_size > 1:
                chunk_rows = adapter.generate_batch(chunk)
            else:
                prediction = adapter.generate_one(chunk[0])
                chunk_rows = [adapter.build_output_record(chunk[0], prediction)]
            for row in chunk_rows:
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            index = len(rows)
            if progress_every and (index % progress_every == 0 or index == len(examples)):
                append_log(log_path, f"generated {index}/{len(examples)} examples")
                print(f"{adapter.name}: generated {index}/{len(examples)}", flush=True)
    partial_path.replace(final_path)
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).resolve()
    cfg = load_yaml(root / args.config)
    examples = read_jsonl(root / args.input)
    if not examples:
        raise ValueError(f"No canonical examples found at {args.input}. Run prepare_ed_test.py first.")
    expected_ids = {str(row["example_id"]) for row in examples}
    if len(expected_ids) != len(examples):
        raise ValueError("Canonical file has duplicate example_id values.")

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = root / "system_baseline/outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    generation_cfg = dict(cfg.get("generation", {}))
    execute_full = bool(generation_cfg.get("execute_full", False)) or args.execute_full
    smoke_test = args.smoke_test or bool(generation_cfg.get("smoke_test", False))
    smoke_n = int(args.smoke_n or generation_cfg.get("smoke_n", 5))
    systems_cfg = cfg.get("systems", {})
    statuses: list[dict[str, Any]] = []

    for system_id in configured_systems(cfg, args.systems):
        if system_id not in ADAPTERS:
            statuses.append(
                {
                    "system_id": system_id,
                    "display_name": system_id,
                    "status": "not_runnable_from_current_repository",
                    "reason": "No adapter registered.",
                    "expected_n": len(examples),
                    "generated_n": 0,
                    "matched_n": 0,
                    "missing_example_id_count": len(examples),
                }
            )
            continue
        system_cfg = dict(systems_cfg.get(system_id, {}))
        display_name = str(system_cfg.get("display_name") or system_id)
        path = output_path(output_dir, system_id)
        log_path = log_dir / f"{system_id}.log"
        append_log(log_path, f"generation requested; execute_full={execute_full}; smoke_test={smoke_test}")
        adapter = load_adapter(system_id, root, system_cfg, generation_cfg)
        runnable = adapter.is_runnable()
        status: dict[str, Any] = {
            "system_id": system_id,
            "display_name": getattr(adapter, "display_name", display_name),
            "runnable": runnable,
            "expected_n": len(examples),
            "output_path": relpath(path, root),
            "started_utc": utc_now(),
            "decode_config": generation_cfg,
            "model_path": system_cfg.get("model_path") or system_cfg.get("generator_dir") or system_cfg.get("checkpoint"),
            "tokenizer_path": system_cfg.get("tokenizer_path") or system_cfg.get("model_path"),
            "checkpoint_path": system_cfg.get("adapter_path") or system_cfg.get("checkpoint") or system_cfg.get("generator_dir"),
        }
        try:
            if not runnable:
                status.update(
                    {
                        "status": "not_runnable_from_current_repository",
                        "reason": adapter.not_runnable_reason or "adapter reported not runnable",
                        "generated_n": 0,
                        "matched_n": 0,
                        "missing_example_id_count": len(examples),
                    }
                )
                append_log(log_path, status["reason"])
            elif existing_full_output(path, expected_ids) and not args.overwrite:
                rows = read_jsonl(path)
                status.update(
                    {
                        "status": "full_run_completed",
                        "reason": "existing full coverage generation file reused",
                        "generated_n": len(rows),
                        "matched_n": len(expected_ids),
                        "missing_example_id_count": 0,
                    }
                )
                append_log(log_path, "existing full output reused")
            elif not execute_full and not smoke_test:
                status.update(
                    {
                        "status": "runnable_but_not_run",
                        "reason": "Full inference disabled. Pass --execute-full or set generation.execute_full=true.",
                        "generated_n": 0,
                        "matched_n": 0,
                        "missing_example_id_count": len(examples),
                    }
                )
                append_log(log_path, status["reason"])
            else:
                selected = examples[:smoke_n] if smoke_test and not execute_full else examples
                adapter.load()
                rows = generate_streaming(
                    adapter,
                    selected,
                    path,
                    log_path,
                    int(generation_cfg.get("progress_every", 20)),
                )
                ids = [str(row.get("example_id", "")) for row in rows]
                matched = set(ids) & expected_ids
                full = len(ids) == len(examples) and set(ids) == expected_ids and len(ids) == len(set(ids))
                status.update(
                    {
                        "status": "full_run_completed" if full else "partial_output_only",
                        "reason": "" if full else "Generated output does not cover the complete canonical set.",
                        "generated_n": len(rows),
                        "matched_n": len(matched),
                        "missing_example_id_count": len(expected_ids - set(ids)),
                    }
                )
                append_log(log_path, f"wrote {len(rows)} generations to {relpath(path, root)}")
        except Exception as exc:
            partial_path = path.with_suffix(".partial.jsonl")
            status.update(
                {
                    "status": "partial_output_only" if partial_path.exists() else "runnable_but_not_run",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "generated_n": 0,
                    "matched_n": 0,
                    "missing_example_id_count": len(examples),
                    "traceback": traceback.format_exc(),
                }
            )
            append_log(log_path, "FAILED\n" + traceback.format_exc())
        finally:
            adapter.close()
            status["finished_utc"] = utc_now()
            statuses.append(status)

    status_payload = {
        "input": relpath(root / args.input, root),
        "expected_n": len(examples),
        "execute_full": execute_full,
        "smoke_test": smoke_test,
        "systems": statuses,
    }
    write_json(log_dir / "generation_status.json", status_payload)
    print(f"Wrote {relpath(log_dir / 'generation_status.json', root)}")
    for item in statuses:
        print(f"{item['system_id']}: {item['status']} ({item.get('generated_n', 0)}/{len(examples)})")


if __name__ == "__main__":
    main()
