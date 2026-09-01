from __future__ import annotations

from system_baseline.adapters.base import PeftChatAdapter


class EmPODPOAdapter(PeftChatAdapter):
    name = "empo_dpo"
    display_name = "EmPO-DPO"
    adapter_path_key = "adapter_path"

