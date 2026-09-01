from __future__ import annotations

from typing import Any, Mapping

from system_baseline.adapters.base import HFChatAdapter
from system_baseline.utils.text import serialize_history


class LLMPromptAdapter(HFChatAdapter):
    name = "llm_prompt"
    display_name = "LLM-prompt"
    system_instruction = (
        "You are an empathetic dialogue assistant. Respond naturally to the speaker using only "
        "the situation and dialogue history. Do not infer from labels or hidden annotations."
    )

    def prompt_for_example(self, example: Mapping[str, Any]) -> str:
        target = str(example.get("target_speaker") or "B")
        history = serialize_history(example.get("history", []) if isinstance(example.get("history"), list) else [])
        return (
            f"Situation:\n{example.get('situation', '')}\n\n"
            f"Dialogue history:\n{history}\n\n"
            f"Write the next response as speaker {target}. Be emotionally appropriate, specific, and conversational. "
            "Only output the response."
        )

