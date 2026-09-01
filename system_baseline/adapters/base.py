from __future__ import annotations

import gc
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from system_baseline.utils.io import PROJECT_ROOT, resolve_project_path
from system_baseline.utils.text import build_input_text, clean_prediction, serialize_history


class BaseSystemAdapter:
    name = "base"
    display_name = "Base"

    def __init__(self, project_root: str | Path | None = None, config: Mapping[str, Any] | None = None) -> None:
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.config = dict(config or {})
        self.decode_config = dict(self.config.get("generation", {}))
        self.metadata: dict[str, Any] = {}
        self.not_runnable_reason = ""

    def is_runnable(self) -> bool:
        return False

    def load(self, config: dict[str, Any] | None = None) -> None:
        self.config.update(config or {})

    def generate_one(self, example: dict[str, Any]) -> str:
        raise NotImplementedError

    def generate_batch(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for example in examples:
            prediction = self.generate_one(example)
            rows.append(self.build_output_record(example, prediction))
        return rows

    def build_output_record(self, example: Mapping[str, Any], prediction: str) -> dict[str, Any]:
        return {
            "example_id": str(example["example_id"]),
            "system": self.display_name,
            "prediction": clean_prediction(prediction),
            "reference": str(example.get("reference", "")),
            "input_text": str(example.get("input_text") or build_input_text(example)),
            "metadata": {
                **self.metadata,
                "decode_config": self.decode_config,
                "adapter": self.__class__.__name__,
            },
        }

    def close(self) -> None:
        gc.collect()


class NotRunnableAdapter(BaseSystemAdapter):
    reason = "not runnable from current repository"

    def is_runnable(self) -> bool:
        self.not_runnable_reason = self.reason
        return False


class HFChatAdapter(BaseSystemAdapter):
    """Local Hugging Face causal LM adapter for canonical ED inputs."""

    system_instruction = ""
    prompt_style = "plain"

    def __init__(self, project_root: str | Path | None = None, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(project_root, config)
        self.model = None
        self.tokenizer = None

    @property
    def model_path(self) -> Path:
        return resolve_project_path(self.config.get("model_path", "Mistral-7B-Instruct-v0.3"), self.project_root)

    def deps_available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            return False, f"missing generation dependency: {exc}"
        return True, ""

    def is_runnable(self) -> bool:
        ok, reason = self.deps_available()
        if not ok:
            self.not_runnable_reason = reason
            return False
        if not (self.model_path / "config.json").exists():
            self.not_runnable_reason = f"model path not found or incomplete: {self.model_path}"
            return False
        return True

    def load(self, config: dict[str, Any] | None = None) -> None:
        super().load(config)
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed = int(self.decode_config.get("seed", 20260512))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dtype_name = str(self.config.get("torch_dtype", "auto"))
        torch_dtype = "auto"
        if dtype_name == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype_name == "fp16":
            torch_dtype = torch.float16
        kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": torch_dtype}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        if not getattr(self.model, "hf_device_map", None) and torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()
        self.metadata.update({"model_path": str(self.model_path), "tokenizer_path": str(self.model_path)})

    def messages_for_example(self, example: Mapping[str, Any]) -> list[dict[str, str]]:
        prompt = self.prompt_for_example(example)
        if self.system_instruction:
            return [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": prompt},
            ]
        return [{"role": "user", "content": prompt}]

    def prompt_for_example(self, example: Mapping[str, Any]) -> str:
        target = str(example.get("target_speaker") or "B")
        history = serialize_history(example.get("history", []) if isinstance(example.get("history"), list) else [])
        return (
            f"Situation:\n{example.get('situation', '')}\n\n"
            f"Dialogue history:\n{history}\n\n"
            f"Continue the dialogue as speaker {target}. Only write the next response."
        )

    def format_prompt(self, example: Mapping[str, Any]) -> str:
        assert self.tokenizer is not None
        messages = self.messages_for_example(example)
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        chunks = []
        for msg in messages:
            role = msg["role"].capitalize()
            chunks.append(f"{role}: {msg['content']}")
        chunks.append("Assistant:")
        return "\n\n".join(chunks)

    def generate_one(self, example: dict[str, Any]) -> str:
        if self.model is None or self.tokenizer is None:
            self.load()
        import torch

        assert self.model is not None and self.tokenizer is not None
        prompt = self.format_prompt(example)
        max_prompt_length = int(self.decode_config.get("max_prompt_length", 2048))
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_length)
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        temperature = float(self.decode_config.get("temperature", 0.7))
        do_sample = bool(self.decode_config.get("do_sample", temperature > 0))
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.decode_config.get("max_new_tokens", 80)),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = float(self.decode_config.get("top_p", 0.9))
            if int(self.decode_config.get("top_k", 0)) > 0:
                gen_kwargs["top_k"] = int(self.decode_config.get("top_k", 50))
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **gen_kwargs)
        new_tokens = output_ids[0, encoded["input_ids"].shape[-1] :]
        return clean_prediction(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        super().close()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class PeftChatAdapter(HFChatAdapter):
    adapter_path_key = "adapter_path"

    @property
    def adapter_path(self) -> Path:
        return resolve_project_path(self.config.get(self.adapter_path_key, ""), self.project_root)

    def deps_available(self) -> tuple[bool, str]:
        ok, reason = super().deps_available()
        if not ok:
            return ok, reason
        try:
            import peft  # noqa: F401
        except Exception as exc:
            return False, f"missing PEFT dependency: {exc}"
        return True, ""

    def is_runnable(self) -> bool:
        if not super().is_runnable():
            return False
        if not (self.adapter_path / "adapter_config.json").exists():
            self.not_runnable_reason = f"adapter checkpoint not found or incomplete: {self.adapter_path}"
            return False
        return True

    def load(self, config: dict[str, Any] | None = None) -> None:
        BaseSystemAdapter.load(self, config)
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        seed = int(self.decode_config.get("seed", 20260512))
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dtype_name = str(self.config.get("torch_dtype", "auto"))
        torch_dtype = "auto"
        if dtype_name == "bf16":
            torch_dtype = torch.bfloat16
        elif dtype_name == "fp16":
            torch_dtype = torch.float16
        kwargs: dict[str, Any] = {"local_files_only": True, "torch_dtype": torch_dtype}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path), local_files_only=True)
        if not getattr(self.model, "hf_device_map", None) and torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()
        self.metadata.update(
            {
                "model_path": str(self.model_path),
                "tokenizer_path": str(self.model_path),
                "adapter_path": str(self.adapter_path),
            }
        )


def namespace_from_decode_config(decode_config: Mapping[str, Any], **extra: Any) -> SimpleNamespace:
    payload = {
        "max_prompt_length": int(decode_config.get("max_prompt_length", 384)),
        "max_new_tokens": int(decode_config.get("max_new_tokens", 80)),
        "do_sample": bool(decode_config.get("do_sample", float(decode_config.get("temperature", 0.7)) > 0)),
        "temperature": float(decode_config.get("temperature", 0.7)),
        "top_p": float(decode_config.get("top_p", 0.9)),
        "progress_every": int(decode_config.get("progress_every", 0)),
        **extra,
    }
    return SimpleNamespace(**payload)

