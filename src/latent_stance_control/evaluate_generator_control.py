from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data import load_prepared_split, read_json, write_json
from .models import StancePrefixProjector
from .train_generator import GenerationDataset, collate


def parse_splits(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_ppl(loss: float) -> float:
    return float(math.exp(loss)) if loss < 50 else float("inf")


@torch.no_grad()
def evaluate_split(lm, projector, embeddings, tokenizer, rows: List[dict], args) -> Dict[str, float]:
    if args.max_examples and args.max_examples > 0:
        rows = rows[: args.max_examples]
    dataset = GenerationDataset(rows, tokenizer, args.max_prompt_length, args.max_response_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer, args.prefix_tokens),
    )
    lm.eval()
    projector.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(args.device)
        token_embeds = embeddings(input_ids)
        prefix = projector(batch["stance"].to(args.device)).to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_labels = torch.full(
            (input_ids.shape[0], args.prefix_tokens),
            -100,
            dtype=torch.long,
            device=args.device,
        )
        labels = torch.cat([prefix_labels, batch["labels"].to(args.device)], dim=1)
        token_count = int((labels != -100).sum().item())
        if token_count == 0:
            continue
        output = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=batch["attention_mask"].to(args.device),
            labels=labels,
        )
        if not torch.isfinite(output.loss):
            raise RuntimeError("Non-finite generator evaluation loss")
        total_nll += float(output.loss.detach().cpu()) * token_count
        total_tokens += token_count
    loss = total_nll / max(total_tokens, 1)
    return {
        "examples": len(rows),
        "tokens": total_tokens,
        "loss": loss,
        "perplexity": safe_ppl(loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained stance-prefix generator under a prepared control dataset.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--generator-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="Override base LM path. Defaults to generator config model.")
    parser.add_argument("--splits", default="dev,test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means full split.")
    parser.add_argument("--prefix-tokens", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to evaluate the generator.") from exc

    generator_dir = Path(args.generator_dir)
    config = read_json(generator_dir / "config.json")
    model_name = args.model or config["model"]
    args.prefix_tokens = args.prefix_tokens or int(config.get("prefix_tokens", 8))
    vector_dim = int(config["vector_dim"])

    tokenizer_path = generator_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.bfloat16} if args.bf16 else {}
    lm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(args.device)
    for param in lm.parameters():
        param.requires_grad_(False)

    projector = StancePrefixProjector(vector_dim, lm.config.hidden_size, args.prefix_tokens).to(args.device)
    projector.load_state_dict(torch.load(generator_dir / "stance_prefix_projector.pt", map_location=args.device))
    embeddings = lm.get_input_embeddings()

    metrics: Dict[str, Dict[str, float]] = {}
    prepared = Path(args.prepared)
    for split in parse_splits(args.splits):
        rows = load_prepared_split(prepared, split)
        if not rows:
            continue
        metrics[split] = evaluate_split(lm, projector, embeddings, tokenizer, rows, args)

    write_json(
        args.out,
        {
            "prepared": str(prepared),
            "generator_dir": str(generator_dir),
            "model": model_name,
            "bf16": args.bf16,
            "max_examples": args.max_examples,
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
