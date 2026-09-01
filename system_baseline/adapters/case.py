from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from system_baseline.adapters.base import BaseSystemAdapter
from system_baseline.utils.io import read_jsonl, resolve_project_path
from system_baseline.utils.text import clean_prediction, context_key, history_from_context_list


class CASEAdapter(BaseSystemAdapter):
    name = "case"
    display_name = "CASE"

    def is_runnable(self) -> bool:
        artifact = self._artifact_path()
        checkpoint = self._checkpoint_path()
        if not artifact.exists():
            self.not_runnable_reason = f"CASE full generation artifact not found: {artifact}"
            return False
        if not checkpoint.exists():
            self.not_runnable_reason = f"CASE checkpoint not found: {checkpoint}"
            return False
        return True

    def _artifact_path(self) -> Path:
        return resolve_project_path(
            self.config.get(
                "generation_artifact",
                "baseline/CASE/outputs/case_baseline/generations/case_test_generations.jsonl",
            ),
            self.project_root,
        )

    def _checkpoint_path(self) -> Path:
        return resolve_project_path(
            self.config.get(
                "checkpoint",
                "baseline/CASE/outputs/case_baseline/checkpoints/CASE_13999_41.4115",
            ),
            self.project_root,
        )

    @staticmethod
    def _history_from_case_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
        return history_from_context_list(row.get("context") if isinstance(row.get("context"), list) else [])

    def load(self, config: dict[str, Any] | None = None) -> None:
        super().load(config)
        artifact = self._artifact_path()
        rows = read_jsonl(artifact)
        self._by_key: dict[tuple[str, tuple[str, ...], str], str] = {}
        for row in rows:
            history = self._history_from_case_row(row)
            key = context_key(row.get("situation", ""), history, row.get("gold_response") or row.get("reference") or "")
            prediction = clean_prediction(row.get("case_response") or row.get("prediction") or row.get("response") or "")
            if key not in self._by_key and prediction:
                self._by_key[key] = prediction
        self.metadata.update(
            {
                "model_path": str(self._checkpoint_path()),
                "checkpoint_path": str(self._checkpoint_path()),
                "source_generation_path": str(artifact),
                "baseline_command": (
                    "bash baseline/CASE/scripts/run_case_baseline.sh --mode test --gpu 0 "
                    "--seed 13 --ckpt baseline/CASE/outputs/case_baseline/checkpoints/CASE_13999_41.4115 "
                    "--emb_file baseline/CASE/vectors/glove.6B.300d.txt"
                ),
                "note": "CASE is aligned from the repository-local full CASE ED test generation exported by the CASE wrapper.",
            }
        )

    def generate_one(self, example: dict[str, Any]) -> str:
        if not hasattr(self, "_by_key"):
            self.load()
        key = context_key(example.get("situation", ""), example.get("history", []), example.get("reference", ""))
        prediction = self._by_key.get(key)
        if prediction is None:
            raise KeyError(f"CASE output not found for canonical example_id={example.get('example_id')}")
        return prediction
