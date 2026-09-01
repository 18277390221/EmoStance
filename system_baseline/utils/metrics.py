from __future__ import annotations

import math
import random
from collections import Counter
from statistics import mean
from typing import Any, Sequence

from .generic import is_generic_response
from .text import ngrams, tokenize


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def modified_precision(candidate: Sequence[str], references: Sequence[Sequence[str]], n: int) -> float:
    cand_counts = Counter(ngrams(candidate, n))
    if not cand_counts:
        return 0.0
    max_ref: Counter[tuple[str, ...]] = Counter()
    for ref in references:
        ref_counts = Counter(ngrams(ref, n))
        for gram, count in ref_counts.items():
            if count > max_ref[gram]:
                max_ref[gram] = count
    clipped = sum(min(count, max_ref[gram]) for gram, count in cand_counts.items())
    return safe_div(clipped, sum(cand_counts.values()))


def sentence_bleu(candidate_text: str, reference_texts: Sequence[str], max_n: int = 2) -> float:
    cand = tokenize(candidate_text)
    refs = [tokenize(r) for r in reference_texts if r is not None]
    if not cand or not refs:
        return 0.0
    precisions = [max(modified_precision(cand, refs, n), 1e-9) for n in range(1, max_n + 1)]
    cand_len = len(cand)
    ref_len = min((len(r) for r in refs), key=lambda x: (abs(x - cand_len), x))
    bp = 1.0 if cand_len > ref_len else math.exp(1.0 - safe_div(ref_len, cand_len))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for idx, token_b in enumerate(b, 1):
            cur.append(prev[idx - 1] + 1 if token_a == token_b else max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(candidate_text: str, reference_text: str) -> float:
    cand = tokenize(candidate_text)
    ref = tokenize(reference_text)
    if not cand or not ref:
        return 0.0
    lcs = lcs_len(cand, ref)
    precision = safe_div(lcs, len(cand))
    recall = safe_div(lcs, len(ref))
    return safe_div(2 * precision * recall, precision + recall)


def meteor_fallback(candidate_text: str, reference_text: str) -> float:
    cand = tokenize(candidate_text)
    ref = tokenize(reference_text)
    if not cand or not ref:
        return 0.0
    overlap = sum((Counter(cand) & Counter(ref)).values())
    precision = safe_div(overlap, len(cand))
    recall = safe_div(overlap, len(ref))
    # NLTK's METEOR uses synonym/stem matching and chunk penalties. This fallback
    # intentionally keeps only the standard unigram precision/recall Fmean term.
    return safe_div(10 * precision * recall, recall + 9 * precision)


def meteor_score(candidate_text: str, reference_text: str) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score as nltk_meteor_score  # type: ignore

        return float(nltk_meteor_score([tokenize(reference_text)], tokenize(candidate_text)))
    except Exception:
        return meteor_fallback(candidate_text, reference_text)


def distinct_n(responses: Sequence[str], n: int) -> float:
    grams: list[tuple[str, ...]] = []
    for response in responses:
        grams.extend(ngrams(tokenize(response), n))
    return safe_div(len(set(grams)), len(grams))


def self_bleu(responses: Sequence[str], max_items: int = 200, seed: int = 20260512) -> float:
    if len(responses) < 2:
        return 0.0
    indexed = list(enumerate(responses))
    if len(indexed) > max_items:
        rng = random.Random(seed)
        indexed = sorted(rng.sample(indexed, max_items), key=lambda x: x[0])
    subset = [text for _, text in indexed]
    values: list[float] = []
    for idx, response in enumerate(subset):
        refs = [r for j, r in enumerate(subset) if j != idx]
        values.append(sentence_bleu(response, refs, max_n=2))
    return float(mean(values)) if values else 0.0


def bertscore_f1_values(
    predictions: Sequence[str],
    references: Sequence[str],
    model_type: str = "distilbert-base-uncased",
    num_layers: int | None = 6,
    batch_size: int = 64,
) -> tuple[list[float | None], str]:
    if not predictions:
        return [], "no rows"
    try:
        from bert_score import score  # type: ignore
    except Exception as exc:
        return [None] * len(predictions), f"bert_score unavailable: {exc}"
    kwargs: dict[str, Any] = {
        "model_type": model_type,
        "batch_size": batch_size,
        "verbose": False,
        "lang": "en",
    }
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    try:
        _, _, f1 = score(list(predictions), list(references), **kwargs)
        return [float(x) for x in f1.tolist()], (
            f"bert_score.score(model_type='{model_type}', num_layers={num_layers}, lang='en')"
        )
    except Exception as exc:
        return [None] * len(predictions), f"bert_score failed: {exc}"


def per_example_reference_metrics(predictions: Sequence[str], references: Sequence[str]) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for pred, ref in zip(predictions, references):
        out.append(
            {
                "rouge_l": rouge_l_f1(pred, ref),
                "bleu_2": sentence_bleu(pred, [ref], max_n=2),
                "meteor": meteor_score(pred, ref),
            }
        )
    return out


def aggregate_system_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    compute_bertscore: bool = True,
    bertscore_model_type: str = "distilbert-base-uncased",
    bertscore_num_layers: int | None = 6,
    bertscore_batch_size: int = 64,
    self_bleu_max_items: int = 200,
    self_bleu_seed: int = 20260512,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    per_rows = per_example_reference_metrics(predictions, references)
    bert_note = "BERTScore disabled"
    bert_values: list[float | None] = [None] * len(predictions)
    if compute_bertscore:
        bert_values, bert_note = bertscore_f1_values(
            predictions,
            references,
            model_type=bertscore_model_type,
            num_layers=bertscore_num_layers,
            batch_size=bertscore_batch_size,
        )
    for row, bert in zip(per_rows, bert_values):
        row["bertscore_f1"] = bert
    def avg(key: str) -> float | None:
        vals = [row.get(key) for row in per_rows if row.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    metrics = {
        "bertscore_f1": avg("bertscore_f1"),
        "rouge_l": avg("rouge_l") or 0.0,
        "bleu_2": avg("bleu_2") or 0.0,
        "meteor": avg("meteor") or 0.0,
        "distinct_1": distinct_n(predictions, 1),
        "distinct_2": distinct_n(predictions, 2),
        "self_bleu": self_bleu(predictions, max_items=self_bleu_max_items, seed=self_bleu_seed),
        "generic_response_rate": safe_div(sum(1 for text in predictions if is_generic_response(text)), len(predictions)),
    }
    return metrics, per_rows, bert_note

