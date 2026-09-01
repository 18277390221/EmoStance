from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any, Mapping

from system_baseline.adapters.base import BaseSystemAdapter
from system_baseline.utils.io import resolve_project_path
from system_baseline.utils.text import clean_prediction, clean_text, context_key, history_from_context_list


SYSTEM_INSTRUCTION = (
    "You are an empathetic dialogue assistant. Generate the next response based on "
    "the dialogue history and the future-aware commonsense knowledge."
)


class SibylAdapter(BaseSystemAdapter):
    name = "sibyl"
    display_name = "Sibyl"

    def __init__(self, project_root: str | Path | None = None, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(project_root, config)
        self.model = None
        self.tokenizer = None
        self._knowledge_by_key: dict[tuple[str, tuple[str, ...], str], dict[str, Any]] = {}

    @property
    def model_path(self) -> Path:
        return resolve_project_path(self.config.get("model_path", "Mistral-7B-Instruct-v0.3"), self.project_root)

    @property
    def adapter_path(self) -> Path:
        return resolve_project_path(
            self.config.get("adapter_path", "baseline/Sibyl/local_mistral_baseline/outputs/response_generator_lora"),
            self.project_root,
        )

    @property
    def sibyl_test_path(self) -> Path:
        return resolve_project_path(self.config.get("sibyl_test_path", "baseline/Sibyl/ED_data/Sibyl_test.json"), self.project_root)

    def is_runnable(self) -> bool:
        try:
            import peft  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            self.not_runnable_reason = f"missing generation dependency: {exc}"
            return False
        if not (self.model_path / "config.json").exists():
            self.not_runnable_reason = f"model path not found or incomplete: {self.model_path}"
            return False
        if not (self.adapter_path / "adapter_config.json").exists():
            self.not_runnable_reason = f"Sibyl LoRA adapter not found: {self.adapter_path}"
            return False
        if not self.sibyl_test_path.exists():
            self.not_runnable_reason = f"Sibyl full ED test knowledge file not found: {self.sibyl_test_path}"
            return False
        return True

    @staticmethod
    def _sample_from_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
        history = history_from_context_list(raw.get("context") if isinstance(raw.get("context"), list) else [])
        return {
            "situation": clean_text(raw.get("situation", "")),
            "history": history,
            "gold_response": clean_text(raw.get("target") or raw.get("gold_response") or raw.get("response") or ""),
            "sibyl_knowledge": {
                "cause": clean_text(raw.get("ChatGPT_cause") or raw.get("cause") or ""),
                "subsequent_event": clean_text(raw.get("ChatGPT_subs") or raw.get("subsequent_event") or ""),
                "emotion_state": clean_text(raw.get("ChatGPT_emo") or raw.get("emotion_state") or ""),
                "listener_intent": clean_text(raw.get("ChatGPT_intent") or raw.get("listener_intent") or ""),
            },
        }

    def _load_knowledge(self) -> None:
        if self._knowledge_by_key:
            return
        with self.sibyl_test_path.open("r", encoding="utf-8") as f:
            raw_rows = json.load(f)
        if not isinstance(raw_rows, list):
            raise ValueError(f"Sibyl test file must be a list: {self.sibyl_test_path}")
        for raw in raw_rows:
            if not isinstance(raw, dict):
                continue
            sample = self._sample_from_raw(raw)
            key = context_key(sample["situation"], sample["history"], sample["gold_response"])
            self._knowledge_by_key.setdefault(key, sample["sibyl_knowledge"])

    def load(self, config: dict[str, Any] | None = None) -> None:
        super().load(config)
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._load_knowledge()
        seed = int(self.decode_config.get("seed", 20260512))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dtype_name = str(self.config.get("torch_dtype", "auto")).lower()
        torch_dtype: Any = "auto"
        if dtype_name == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype_name == "fp16":
            torch_dtype = torch.float16
        elif dtype_name == "fp32":
            torch_dtype = torch.float32
        kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": torch_dtype}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path), local_files_only=True)
        self.model.eval()
        self.metadata.update(
            {
                "model_path": str(self.model_path),
                "tokenizer_path": str(self.model_path),
                "adapter_path": str(self.adapter_path),
                "sibyl_test_path": str(self.sibyl_test_path),
                "method_note": (
                    "Repository-local Mistral-Sibyl LoRA response generator using preconstructed "
                    "Sibyl ED commonsense knowledge. Gold response is used only for example_id alignment and metrics."
                ),
            }
        )

    @staticmethod
    def _format_history(history: Any) -> str:
        if not isinstance(history, list) or not history:
            return "N/A"
        lines: list[str] = []
        for index, turn in enumerate(history):
            speaker = str(turn.get("speaker") or turn.get("role") or ("A" if index % 2 == 0 else "B"))
            text = clean_text(turn.get("text") or "")
            if text:
                lines.append(f"{speaker}: {text}")
        return "\n".join(lines) if lines else "N/A"

    @staticmethod
    def _knowledge_text(knowledge: Mapping[str, Any], key: str) -> str:
        return clean_text(knowledge.get(key, "")) or "N/A"

    def _prompt(self, example: Mapping[str, Any]) -> str:
        key = context_key(example.get("situation", ""), example.get("history", []), example.get("reference", ""))
        knowledge = self._knowledge_by_key.get(key)
        if knowledge is None:
            raise KeyError(f"Sibyl knowledge not found for canonical example_id={example.get('example_id')}")
        return (
            "[INST] "
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"Situation:\n{clean_text(example.get('situation', '')) or 'N/A'}\n\n"
            f"Dialogue History:\n{self._format_history(example.get('history'))}\n\n"
            "Future-aware Commonsense Knowledge:\n"
            f"Cause: {self._knowledge_text(knowledge, 'cause')}\n"
            f"Subsequent Event: {self._knowledge_text(knowledge, 'subsequent_event')}\n"
            f"Emotion State: {self._knowledge_text(knowledge, 'emotion_state')}\n"
            f"Listener Intent: {self._knowledge_text(knowledge, 'listener_intent')}\n\n"
            "Please generate a natural, coherent, and empathetic next response. "
            "Only output the response itself, in 1 to 3 sentences. [/INST]"
        )

    def generate_one(self, example: dict[str, Any]) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        return self.generate_batch([example])[0]["prediction"]

    def generate_batch(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None or self.tokenizer is None:
            self.load()
        import torch

        assert self.model is not None and self.tokenizer is not None
        prompts = [self._prompt(example) for example in examples]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.decode_config.get("max_prompt_length", 2048)),
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        temperature = float(self.decode_config.get("temperature", 0.7))
        do_sample = bool(self.decode_config.get("do_sample", temperature > 0))
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.decode_config.get("max_new_tokens", 80)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = float(self.decode_config.get("top_p", 0.9))
            if int(self.decode_config.get("top_k", 0)) > 0:
                kwargs["top_k"] = int(self.decode_config.get("top_k", 50))
        with torch.no_grad():
            output = self.model.generate(**encoded, **kwargs)
        rows: list[dict[str, Any]] = []
        start = encoded["input_ids"].shape[-1]
        for example, output_ids in zip(examples, output):
            text = self.tokenizer.decode(output_ids[start:], skip_special_tokens=True)
            for marker in ("</s>", "<s>", "[/INST]", "[INST]"):
                text = text.replace(marker, "")
            rows.append(self.build_output_record(example, clean_prediction(text)))
        return rows

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
