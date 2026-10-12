from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value else default
    except ValueError:
        return default


@dataclass
class Config:
    api_key: str
    base_url: str | None
    model: str
    workspace: Path
    max_steps: int = 50
    command_timeout: int = 120
    auto_approve: bool = False
    max_tool_output: int = 30000
    language: str = "简体中文 (Simplified Chinese)"

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(Path.cwd() / ".env")

        cfg = cls(
            api_key=os.getenv("MYAGENT_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            base_url=os.getenv("MYAGENT_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("MYAGENT_MODEL") or "gpt-4o",
            workspace=Path(os.getenv("MYAGENT_WORKSPACE") or Path.cwd()),
            max_steps=_env_int("MYAGENT_MAX_STEPS", 50),
            command_timeout=_env_int("MYAGENT_COMMAND_TIMEOUT", 120),
            auto_approve=_env_bool("MYAGENT_AUTO_APPROVE", False),
            language=os.getenv("MYAGENT_LANGUAGE") or "简体中文 (Simplified Chinese)",
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        cfg.workspace = Path(cfg.workspace).expanduser().resolve()
        return cfg
