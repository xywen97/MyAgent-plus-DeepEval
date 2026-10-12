"""Judge model for LLM-based deepeval metrics (any OpenAI-compatible endpoint)."""
from __future__ import annotations

import os
from functools import lru_cache

from deepeval.models import LocalModel

from my_agent.config import Config


@lru_cache(maxsize=1)
def get_judge() -> LocalModel:
    """MYAGENT_JUDGE_* 未设置时回退到被测 Agent 的模型配置。

    建议用比被测模型更强、且不同的模型做裁判，以减少自评偏差。
    """
    agent_cfg = Config.from_env()
    model = os.getenv("MYAGENT_JUDGE_MODEL") or agent_cfg.model
    api_key = os.getenv("MYAGENT_JUDGE_API_KEY") or agent_cfg.api_key
    base_url = os.getenv("MYAGENT_JUDGE_BASE_URL") or agent_cfg.base_url or "https://api.openai.com/v1"
    return LocalModel(model=model, api_key=api_key, base_url=base_url, temperature=0)
