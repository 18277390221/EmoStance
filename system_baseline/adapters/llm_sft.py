from __future__ import annotations

from system_baseline.adapters.base import PeftChatAdapter


class LLMSFTAdapter(PeftChatAdapter):
    name = "llm_sft"
    display_name = "LLM-SFT"
    adapter_path_key = "adapter_path"

