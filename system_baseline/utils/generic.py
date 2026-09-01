from __future__ import annotations

import re

from .text import clean_text, tokenize


GENERIC_EXACT = {
    "i understand",
    "i understand how you feel",
    "i'm sorry to hear that",
    "i am sorry to hear that",
    "that sounds really hard",
    "that sounds tough",
    "it's okay",
    "i'm here for you",
}

GENERIC_PATTERNS = [
    r"\bi understand\b",
    r"\bi'?m sorry\b",
    r"\bi am sorry\b",
    r"\bsorry to hear\b",
    r"\bthat sounds\b",
    r"\bi hope\b",
    r"\bit'?s okay\b",
    r"\bi can imagine\b",
    r"\bthat must be\b",
    r"\bi'?m here for you\b",
    r"\bthank you for sharing\b",
    r"\bthanks for sharing\b",
    r"\beverything will be okay\b",
]


def is_generic_response(text: str) -> bool:
    norm = clean_text(text).lower()
    norm = re.sub(r"\s+", " ", norm).strip()
    stripped = re.sub(r"[.!?]+$", "", norm)
    words = tokenize(norm)
    if not words:
        return False
    if stripped in GENERIC_EXACT:
        return True
    if len(words) <= 4:
        return True
    if len(words) > 24:
        return False
    return any(re.search(pattern, norm) for pattern in GENERIC_PATTERNS)

