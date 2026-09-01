from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

from system_baseline.adapters.base import BaseSystemAdapter, namespace_from_decode_config
from system_baseline.utils.io import resolve_project_path
from system_baseline.utils.text import clean_prediction, serialize_history


class EmoStanceAdapter(BaseSystemAdapter):
    name = "emostance"
    display_name = "EmoStance / Ours"

    def __init__(self, project_root: str | Path | None = None, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(project_root, config)
        self.lm = None
        self.generator_tokenizer = None
        self.projector = None
        self.prefix_tokens = 0
        self.stance_model = None
        self.stance_tokenizer = None
        self.cluster_prototypes = None
        self.args = None

    @property
    def generator_dir(self) -> Path:
        return resolve_project_path(self.config.get("generator_dir", "runs/main/generator"), self.project_root)

    @property
    def stance_dir(self) -> Path:
        return resolve_project_path(self.config.get("stance_dir", "runs/main/stance_baseline"), self.project_root)

    @property
    def prepared_dir(self) -> Path:
        return resolve_project_path(self.config.get("prepared_dir", "runs/main/prepared"), self.project_root)

    @property
    def cluster_prototypes_path(self) -> Path | None:
        raw = self.config.get("cluster_prototypes")
        return resolve_project_path(raw, self.project_root) if raw else None

    @property
    def base_model_path(self) -> Path:
        raw = self.config.get("base_model_override") or self.config.get("model_path") or "Mistral-7B-Instruct-v0.3"
        return resolve_project_path(raw, self.project_root)

    def is_runnable(self) -> bool:
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            self.not_runnable_reason = f"missing EmoStance generation dependency: {exc}"
            return False
        needed = [
            self.generator_dir / "config.json",
            self.generator_dir / "stance_prefix_projector.pt",
            self.base_model_path / "config.json",
            self.stance_dir / "stance_predictor.pt",
            self.stance_dir / "tokenizer",
            self.prepared_dir / "meta.json",
        ]
        missing = [str(path) for path in needed if not path.exists()]
        if missing:
            self.not_runnable_reason = "missing EmoStance checkpoint/config file(s): " + "; ".join(missing)
            return False
        proto = self.cluster_prototypes_path
        if proto is not None and not proto.exists():
            self.not_runnable_reason = f"cluster prototype file configured but not found: {proto}"
            return False
        return True

    def load(self, config: dict[str, Any] | None = None) -> None:
        super().load(config)
        if self.lm is not None:
            return
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from latent_stance_control.data import build_model_text, read_json
        from latent_stance_control.models import DebertaStancePredictor, StancePrefixProjector

        seed = int(self.decode_config.get("seed", 20260512))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = str(self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.args = namespace_from_decode_config(self.decode_config, device=device)
        generator_config = read_json(self.generator_dir / "config.json")
        self.prefix_tokens = int(generator_config.get("prefix_tokens", 8))
        vector_dim = int(generator_config["vector_dim"])
        torch_dtype = torch.bfloat16 if bool(self.config.get("bf16", torch.cuda.is_available())) else torch.float32
        self.generator_tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path), local_files_only=True, use_fast=True)
        if self.generator_tokenizer.pad_token_id is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_path),
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(device)
        self.lm.eval()
        for param in self.lm.parameters():
            param.requires_grad_(False)
        self.projector = StancePrefixProjector(vector_dim, self.lm.config.hidden_size, self.prefix_tokens).to(device)
        self.projector.load_state_dict(torch.load(self.generator_dir / "stance_prefix_projector.pt", map_location=device))
        self.projector.eval()
        meta = read_json(self.prepared_dir / "meta.json")
        self.stance_model = DebertaStancePredictor(str(self.stance_dir / "encoder"), meta["num_clusters"], meta["vector_dim"]).float().to(device)
        state = torch.load(self.stance_dir / "stance_predictor.pt", map_location=device)
        self.stance_model.load_state_dict(state)
        self.stance_model.eval()
        self.stance_tokenizer = AutoTokenizer.from_pretrained(str(self.stance_dir / "tokenizer"), local_files_only=True, use_fast=True)
        proto = self.cluster_prototypes_path
        if proto is not None:
            self.cluster_prototypes = np.asarray(read_json(proto), dtype=np.float32)
        self._build_model_text = build_model_text
        self.metadata.update(
            {
                "model_path": str(self.base_model_path),
                "generator_dir": str(self.generator_dir),
                "stance_dir": str(self.stance_dir),
                "stance_vector_source": "predicted_cluster_prototype" if self.cluster_prototypes is not None else "predicted_target_vector",
                "input_policy": "reference, emotion labels, emoji annotations, gold stance clusters, and oracle vectors are not used.",
            }
        )

    def canonical_to_latent_row(self, example: Mapping[str, Any]) -> dict[str, Any]:
        history = example.get("history", []) if isinstance(example.get("history"), list) else []
        context = serialize_history(history)
        target_role = str(example.get("target_speaker") or "B")
        return {
            "dialogue_id": example.get("dialogue_id", ""),
            "turn_id": example.get("turn_id", 0),
            "role": history[-1].get("speaker", "A") if history else "A",
            "next_role": target_role,
            "transition": f"{history[-1].get('speaker', 'A') if history else 'A'}->{target_role}",
            "situation": example.get("situation", ""),
            "context": context,
            "response": "",
        }

    def predict_stance_vector(self, latent_row: Mapping[str, Any]):
        import numpy as np
        import torch

        assert self.stance_model is not None and self.stance_tokenizer is not None and self.args is not None
        text = self._build_model_text(dict(latent_row))
        encoded = self.stance_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=int(self.config.get("stance_max_length", 384)),
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self.stance_model(
                encoded["input_ids"].to(self.args.device),
                encoded["attention_mask"].to(self.args.device),
            )
            target_prob = torch.softmax(output.target_logits, dim=-1).detach().cpu().numpy()[0]
            if self.cluster_prototypes is not None:
                vector = target_prob @ self.cluster_prototypes
            else:
                vector = output.target_vector.detach().cpu().numpy()[0]
        return np.asarray(vector, dtype=np.float32), target_prob.astype(float).tolist()

    def generate_one(self, example: dict[str, Any]) -> str:
        if self.lm is None:
            self.load()
        from latent_stance_control.generate_and_score_controls import generate_one

        assert self.lm is not None and self.generator_tokenizer is not None and self.projector is not None and self.args is not None
        latent_row = self.canonical_to_latent_row(example)
        stance_vector, target_prob = self.predict_stance_vector(latent_row)
        text = generate_one(self.lm, self.generator_tokenizer, self.projector, latent_row, stance_vector, self.args, self.prefix_tokens)
        self.metadata["last_predicted_target_cluster"] = target_prob
        return clean_prediction(text)

    def build_output_record(self, example: Mapping[str, Any], prediction: str) -> dict[str, Any]:
        row = super().build_output_record(example, prediction)
        row["metadata"] = dict(row["metadata"])
        row["metadata"].pop("last_predicted_target_cluster", None)
        return row

    def close(self) -> None:
        self.lm = None
        self.generator_tokenizer = None
        self.projector = None
        self.stance_model = None
        self.stance_tokenizer = None
        super().close()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
