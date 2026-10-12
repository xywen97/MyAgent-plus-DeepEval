"""MyAgent 评测入口：deepeval test run evals/test_agent.py"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import deepeval
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall

import my_agent
from my_agent.config import Config
from my_agent.prompts import SYSTEM_PROMPT_TEMPLATE
from my_agent.tools import registry

from .harness import CaseResult, EvalCase, load_cases, run_case
from .metrics import build_metrics

CASES = load_cases(os.getenv("MYAGENT_EVAL_DATASET", "*.json"))
MAX_TOOL_OUTPUT_IN_REPORT = 2000


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


@deepeval.log_hyperparameters
def hyperparameters():
    cfg = Config.from_env()
    return {
        "agent_version": my_agent.__version__,
        "git_commit": _git_commit(),
        "model": cfg.model,
        "judge_model": os.getenv("MYAGENT_JUDGE_MODEL") or cfg.model,
        "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT_TEMPLATE.encode()).hexdigest()[:12],
        "tools": ",".join(sorted(registry.names())),
        "run_label": os.getenv("MYAGENT_EVAL_LABEL", ""),
    }


def to_test_case(result: CaseResult) -> LLMTestCase:
    rec, case = result.record, result.case
    return LLMTestCase(
        name=case.id,
        input=case.input,
        actual_output=rec.final_output or f"(无最终回复，stop_reason={rec.stop_reason})",
        expected_output=case.expected_output,
        tools_called=[
            ToolCall(name=c.name, input_parameters=c.args, output=c.output[:MAX_TOOL_OUTPUT_IN_REPORT])
            for c in rec.tool_calls
        ],
        expected_tools=[ToolCall(name=n) for n in case.expected_tools] or None,
        context=[f"工作区最终文件 diff:\n{result.diff}"],
        completion_time=rec.duration_s,
        tags=case.tags or None,
        metadata={
            "stop_reason": rec.stop_reason,
            "steps": rec.steps,
            "tool_call_count": len(rec.tool_calls),
            "tool_errors": sum(c.is_error for c in rec.tool_calls),
            "prompt_tokens": rec.prompt_tokens,
            "completion_tokens": rec.completion_tokens,
            "denied_commands": result.ui.denied_commands,
            "questions_asked": result.ui.questions_asked,
            "error": rec.error,
        },
    )


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_agent(case: EvalCase, agent_config: Config) -> None:
    result = run_case(case, agent_config)
    try:
        test_case = to_test_case(result)
        assert_test(test_case, build_metrics(result), run_async=False)
    finally:
        result.cleanup()
