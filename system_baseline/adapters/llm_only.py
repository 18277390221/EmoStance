from __future__ import annotations

from system_baseline.adapters.base import HFChatAdapter


class LLMOnlyAdapter(HFChatAdapter):
    name = "llm_only"
    display_name = "LLM-only"
    system_instruction = ""

