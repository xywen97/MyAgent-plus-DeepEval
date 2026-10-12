from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", str(ROOT / "evals" / "results"))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from my_agent.config import Config  # noqa: E402


@pytest.fixture(scope="session")
def agent_config() -> Config:
    cfg = Config.from_env()
    if not cfg.api_key:
        pytest.skip("未配置 MYAGENT_API_KEY，跳过评测")
    return cfg
