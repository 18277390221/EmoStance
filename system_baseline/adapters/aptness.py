from __future__ import annotations

import gc
import json
import pickle
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from system_baseline.adapters.base import BaseSystemAdapter
from system_baseline.utils.io import resolve_project_path
from system_baseline.utils.text import clean_prediction, clean_text


GEN_SYSTEM = "You are an empathetic listener. Generate a concise, natural, and emotionally appropriate response to the speaker."
GEN_USER = """[Dialogue Context]
{context}
[End of Dialogue Context]

Please generate the next Listener response. The response should directly address the speaker's latest emotion or situation, sound natural, and avoid being overly long."""

APT_RAG_SYSTEM = "You are an empathetic listener. Your task is to generate a single natural response that is coherent with the dialogue context and emotionally supportive."
APT_RAG_USER = """[Dialogue Context]
{context}
[End of Dialogue Context]

[Initial Response]
{g1}
[End of Initial Response]

[Retrieved Empathetic Responses]
{retrieved_responses}
[End of Retrieved Empathetic Responses]

Based on the dialogue context, the initial response, and the retrieved empathetic response examples, generate one final Listener response.

Requirements:
- Be directly relevant to the Speaker's latest message.
- Show understanding of the Speaker's feelings and situation.
- Be supportive and natural.
- Do not copy the retrieved responses verbatim.
- Avoid long lists unless the context clearly asks for advice.
- Output only the final response text."""


class APTNESSAdapter(BaseSystemAdapter):
    name = "aptness"
    display_name = "APTNESS"

    def __init__(self, project_root: str | Path | None = None, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(project_root, config)
        self.model = None
        self.tokenizer = None
        self._vectorizer = None
        self._embeddings = None
        self._retrieval_metadata: list[dict[str, Any]] = []
        self._last_extra: dict[str, Any] = {}

    @property
    def model_path(self) -> Path:
        return resolve_project_path(self.config.get("model_path", "Mistral-7B-Instruct-v0.3"), self.project_root)

    @property
    def index_dir(self) -> Path:
        return resolve_project_path(
            self.config.get("index_dir", "baseline/APTNESS/aptrag_mistral_baseline/runs/apt_rag_mistral/apt_response_index"),
            self.project_root,
        )

    def is_runnable(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            self.not_runnable_reason = f"missing generation dependency: {exc}"
            return False
        if not (self.model_path / "config.json").exists():
            self.not_runnable_reason = f"model path not found or incomplete: {self.model_path}"
            return False
        required = ["index_config.json", "metadata.jsonl", "embeddings.npy", "tfidf_vectorizer.pkl"]
        missing = [name for name in required if not (self.index_dir / name).exists()]
        if missing:
            self.not_runnable_reason = f"APT-RAG retrieval index missing files at {self.index_dir}: {', '.join(missing)}"
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

        with (self.index_dir / "tfidf_vectorizer.pkl").open("rb") as f:
            self._vectorizer = pickle.load(f)
        self._embeddings = np.load(self.index_dir / "embeddings.npy").astype(np.float32)
        with (self.index_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
            self._retrieval_metadata = [json.loads(line) for line in f if line.strip()]

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
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs)
        self.model.eval()
        self.metadata.update(
            {
                "model_path": str(self.model_path),
                "tokenizer_path": str(self.model_path),
                "index_dir": str(self.index_dir),
                "retrieval_top_k": int(self.config.get("retrieval_top_k", 2)),
                "method_note": (
                    "Repository-local APT-RAG/Mistral reproduction under baseline/APTNESS, "
                    "run on canonical ED inputs. This is not a full original APTNESS strategy-checkpoint rerun."
                ),
            }
        )

    @staticmethod
    def _context(example: Mapping[str, Any]) -> str:
        lines: list[str] = []
        situation = clean_text(example.get("situation", ""))
        if situation:
            lines.append(f"Situation: {situation}")
        for turn in example.get("history", []) if isinstance(example.get("history"), list) else []:
            speaker = str(turn.get("speaker") or turn.get("role") or "A")
            role = "Listener" if speaker == "B" else "Speaker"
            text = clean_text(turn.get("text") or "")
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines).strip()

    @staticmethod
    def _messages(system: str, user_template: str, **kwargs: Any) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user_template.format(**kwargs)}]

    def _input_device(self):
        assert self.model is not None
        if hasattr(self.model, "hf_device_map"):
            for device in self.model.hf_device_map.values():
                if isinstance(device, str) and device not in {"cpu", "disk"}:
                    import torch

                    return torch.device(device)
        return next(self.model.parameters()).device

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        assert self.tokenizer is not None
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages) + "\n\nAssistant:"

    def _generate_messages_batch(self, batch_messages: list[list[dict[str, str]]]) -> list[str]:
        import torch

        assert self.model is not None and self.tokenizer is not None
        prompts = [self._format_messages(messages) for messages in batch_messages]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.decode_config.get("max_prompt_length", 2048)),
        )
        device = self._input_device()
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
        start = encoded["input_ids"].shape[-1]
        return [clean_prediction(self.tokenizer.decode(output_ids[start:], skip_special_tokens=True)) for output_ids in output]

    def _generate_messages(self, messages: list[dict[str, str]]) -> str:
        return self._generate_messages_batch([messages])[0]

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return x / denom

    def _retrieve(self, query: str) -> list[dict[str, Any]]:
        assert self._vectorizer is not None and self._embeddings is not None
        arr = self._vectorizer.transform([query]).astype(np.float32).toarray()
        query_vec = self._l2_normalize(arr)[0]
        sims = self._embeddings @ query_vec
        top_k = min(int(self.config.get("retrieval_top_k", 2)), len(self._retrieval_metadata))
        order = np.argsort(-sims)[:top_k]
        out: list[dict[str, Any]] = []
        for idx in order:
            item = dict(self._retrieval_metadata[int(idx)])
            item["score"] = float(sims[int(idx)])
            out.append(item)
        return out

    @staticmethod
    def _format_retrieved(items: list[dict[str, Any]]) -> str:
        lines = []
        for index, item in enumerate(items, 1):
            response = clean_text(item.get("response", ""))
            if response:
                lines.append(f"{index}. {response}")
        return "\n".join(lines)

    def generate_one(self, example: dict[str, Any]) -> str:
        return self.generate_batch([example])[0]["prediction"]

    def generate_batch(self, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None:
            self.load()
        contexts = [self._context(example) for example in examples]
        initials = self._generate_messages_batch([self._messages(GEN_SYSTEM, GEN_USER, context=context) for context in contexts])
        retrieved_items = [self._retrieve(initial) for initial in initials]
        finals = self._generate_messages_batch(
            [
                self._messages(
                    APT_RAG_SYSTEM,
                    APT_RAG_USER,
                    context=context,
                    g1=initial,
                    retrieved_responses=self._format_retrieved(retrieved),
                )
                for context, initial, retrieved in zip(contexts, initials, retrieved_items)
            ]
        )
        rows: list[dict[str, Any]] = []
        for example, initial, retrieved, final in zip(examples, initials, retrieved_items, finals):
            row = super().build_output_record(example, final)
            row["metadata"].update(
                {
                    "initial_response": initial,
                    "retrieved_responses": [
                        {key: item.get(key, "") for key in ("apt_id", "response", "score", "dialogue_history", "emotion", "factor", "situation")}
                        for item in retrieved
                    ],
                }
            )
            rows.append(row)
        return rows

    def generate_one_legacy(self, example: dict[str, Any]) -> str:
        if self.model is None:
            self.load()
        context = self._context(example)
        initial = self._generate_messages(self._messages(GEN_SYSTEM, GEN_USER, context=context))
        retrieved = self._retrieve(initial)
        final = self._generate_messages(
            self._messages(
                APT_RAG_SYSTEM,
                APT_RAG_USER,
                context=context,
                g1=initial,
                retrieved_responses=self._format_retrieved(retrieved),
            )
        )
        self._last_extra = {
            "initial_response": initial,
            "retrieved_responses": [
                {key: item.get(key, "") for key in ("apt_id", "response", "score", "dialogue_history", "emotion", "factor", "situation")}
                for item in retrieved
            ],
        }
        return final

    def build_output_record(self, example: Mapping[str, Any], prediction: str) -> dict[str, Any]:
        row = super().build_output_record(example, prediction)
        row["metadata"].update(self._last_extra)
        return row

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._vectorizer = None
        self._embeddings = None
        self._retrieval_metadata = []
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
