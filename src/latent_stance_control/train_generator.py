from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import build_model_text, load_prepared_split, read_json, write_json
from .models import StancePrefixProjector


class GenerationDataset(Dataset):
    def __init__(self, rows: List[dict], tokenizer, max_prompt_length: int, max_response_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        prompt = build_model_text(row) + f'\n{row.get("next_role", "B")}:'
        response = str(row["response"])
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True)["input_ids"]
        response_ids = self.tokenizer(response, truncation=True, max_length=self.max_response_length, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + response_ids + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "labels": labels, "stance": np.asarray(row["target_vector"], dtype=np.float32)}


def collate(batch: List[Dict], tokenizer, prefix_tokens: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention.append([1] * (len(item["input_ids"]) + prefix_tokens) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "stance": torch.tensor(np.stack([x["stance"] for x in batch]), dtype=torch.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mistral generator with internal stance soft-prefix control.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--freeze-lm", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to train the generator.") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = read_json(Path(args.prepared) / "meta.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.bfloat16} if args.bf16 else {}
    lm = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    if args.freeze_lm:
        for param in lm.parameters():
            param.requires_grad_(False)
    projector = StancePrefixProjector(meta["vector_dim"], lm.config.hidden_size, args.prefix_tokens).to(args.device)
    rows = load_prepared_split(args.prepared, "train")
    loader = DataLoader(GenerationDataset(rows, tokenizer, args.max_prompt_length, args.max_response_length), batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer, args.prefix_tokens))
    params = list(projector.parameters()) + [p for p in lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    embeddings = lm.get_input_embeddings()
    loss_history = []
    for epoch in range(args.epochs):
        lm.train()
        projector.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(args.device)
            token_embeds = embeddings(input_ids)
            prefix = projector(batch["stance"].to(args.device)).to(token_embeds.dtype)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
            prefix_labels = torch.full((input_ids.shape[0], args.prefix_tokens), -100, dtype=torch.long, device=args.device)
            labels = torch.cat([prefix_labels, batch["labels"].to(args.device)], dim=1)
            output = lm(inputs_embeds=inputs_embeds, attention_mask=batch["attention_mask"].to(args.device), labels=labels)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        loss_history.append({"epoch": epoch + 1, "train_lm_loss": total_loss / max(len(loader), 1)})
        write_json(out / "loss_history.json", loss_history)
    torch.save(projector.state_dict(), out / "stance_prefix_projector.pt")
    tokenizer.save_pretrained(out / "tokenizer")
    write_json(out / "config.json", {"model": args.model, "prefix_tokens": args.prefix_tokens, "vector_dim": meta["vector_dim"], "freeze_lm": args.freeze_lm})


if __name__ == "__main__":
    main()
