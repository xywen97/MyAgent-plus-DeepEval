"""MyAgent 评测入口：deepeval test run evals/test_agent.py"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import deepeval
import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import ToolCall
from deepeval.tracing import update_current_trace

import my_agent
from my_agent.config import Config
from my_agent.prompts import SYSTEM_PROMPT_TEMPLATE
from my_agent.tools import registry

from .harness import CaseResult, EvalCase, load_cases, run_case
from .metrics import build_metrics

CASES = load_cases(os.getenv("MYAGENT_EVAL_DATASET", "*.json"))
EVAL_PROTOCOL = "native-trace-v2"


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
        "eval_protocol": EVAL_PROTOCOL,
        "checks_only": os.getenv("MYAGENT_EVAL_CHECKS_ONLY", "0"),
        "step_efficiency": os.getenv("MYAGENT_EVAL_STEP_EFFICIENCY", "0"),
        "git_commit": _git_commit(),
        "model": cfg.model,
        "judge_model": os.getenv("MYAGENT_JUDGE_MODEL") or cfg.model,
        "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT_TEMPLATE.encode()).hexdigest()[:12],
        "tools": ",".join(sorted([*registry.names(), "delegate_task"])),
        "stream": cfg.stream,
        "context_budget": cfg.context_budget,
        "subagent_max_steps": cfg.subagent_max_steps,
        "run_label": os.getenv("MYAGENT_EVAL_LABEL", ""),
    }


def to_golden(result: CaseResult) -> Golden:
    rec, case = result.record, result.case
    return Golden(
        name=case.id,
        input=case.input,
        expected_output=case.expected_output,
        expected_tools=[ToolCall(name=n) for n in case.expected_tools] or None,
        additional_metadata={
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
    if os.getenv("MYAGENT_EVAL_PROGRESS", "1") != "0":
        print(f"\n开始用例：{case.id}", flush=True)
    result = run_case(case, agent_config)
    metrics = []
    try:
        golden = to_golden(result)
        update_current_trace(expected_tools=golden.expected_tools, metadata=golden.additional_metadata)
        metrics = build_metrics(result)
        result.ui.progress(f"Agent 运行结束：{result.record.steps} 步，{result.record.duration_s:.1f}s；开始评分...")
        assert_test(golden=golden, metrics=metrics)
    finally:
        for metric in metrics:
            result.ui.progress(f"评分 {metric.__name__}：{metric.score}（阈值 {metric.threshold}）")
        result.ui.progress(f"用例结束：{case.id}")
        result.cleanup()
