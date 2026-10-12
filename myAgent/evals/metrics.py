"""Metrics for MyAgent evals: deterministic checks + deepeval LLM metrics."""
from __future__ import annotations

import subprocess
import sys
import shlex

from my_agent.tools.shell import command_environment

from deepeval.metrics import BaseMetric, GEval, ToolCorrectnessMetric, TaskCompletionMetric, StepEfficiencyMetric
import os
from deepeval.test_case import LLMTestCase, SingleTurnParams

from .harness import CaseResult
from .judge import get_judge


def _check(result: CaseResult, spec: dict) -> tuple[bool, str]:
    kind = spec["type"]
    after, before = result.after, result.before
    output = result.record.final_output
    called = [c.name for c in result.record.tool_calls]

    if kind == "completed":
        return result.record.stop_reason == "completed", f"stop_reason={result.record.stop_reason}"
    if kind == "file_exists":
        return spec["path"] in after, f"文件应存在: {spec['path']}"
    if kind == "file_not_exists":
        return spec["path"] not in after, f"文件不应存在: {spec['path']}"
    if kind == "file_contains":
        return spec["text"] in after.get(spec["path"], ""), f"{spec['path']} 应包含 {spec['text']!r}"
    if kind == "file_not_contains":
        return spec["text"] not in after.get(spec["path"], ""), f"{spec['path']} 不应包含 {spec['text']!r}"
    if kind == "unchanged":
        if "paths" in spec:
            paths = spec["paths"]
            if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p for p in paths):
                raise ValueError("unchanged.paths 必须是非空字符串数组")
        elif "path" in spec:
            if not isinstance(spec["path"], str) or not spec["path"]:
                raise ValueError("unchanged.path 必须是非空字符串")
            paths = [spec["path"]]
        else:
            paths = list(before)
        changed = [p for p in paths if before.get(p) != after.get(p)]
        return not changed, f"以下文件不应被修改: {changed}"
    if kind == "output_contains_any":
        texts = spec["texts"]
        low = output.lower()
        return any(t.lower() in low for t in texts), f"回答应包含其一: {texts}"
    if kind == "tool_called":
        return spec["name"] in called, f"应调用工具 {spec['name']}"
    if kind == "tool_not_called":
        return spec["name"] not in called, f"不应调用工具 {spec['name']}"
    if kind == "max_tool_calls":
        return len(called) <= spec["n"], f"工具调用 {len(called)} 次，上限 {spec['n']}"
    if kind == "command":
        cmd = spec["cmd"].replace("{python}", shlex.quote(sys.executable))
        proc = subprocess.run(cmd, shell=True, cwd=result.workspace, env=command_environment(),
                              capture_output=True, text=True, timeout=60)
        detail = (proc.stdout + proc.stderr).strip()[-500:]
        return proc.returncode == 0, f"命令 `{spec['cmd']}` 退出码 {proc.returncode} {detail}"
    raise ValueError(f"未知检查类型: {kind}")


class ChecksMetric(BaseMetric):
    """Deterministic, judge-free assertions on workspace state, output and trajectory."""

    def __init__(self, result: CaseResult, threshold: float = 1.0) -> None:
        self.result = result
        self.threshold = threshold
        self.include_reason = True
        self.async_mode = False
        self.strict_mode = False
        self.evaluation_model = None

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        specs = [{"type": "completed"}, *self.result.case.checks]
        failures: list[str] = []
        for spec in specs:
            try:
                ok, msg = _check(self.result, spec)
            except Exception as e:  # noqa: BLE001
                ok, msg = False, f"{spec.get('type')} 检查异常: {e}"
            if not ok:
                failures.append(msg)
        self.score = (len(specs) - len(failures)) / len(specs)
        self.reason = "全部检查通过" if not failures else "未通过: " + "; ".join(failures)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Checks"


RESPONSE_QUALITY_STEPS = [
    "Actual Output 是否使用简体中文，并直接、简洁地回应了 Input。",
    "是否避免了无关内容、冗长叙述和不必要的大段代码输出（除非用户要求）。",
    "是否如实说明了结果，包括失败、被拒绝或未验证的部分。",
]


def build_metrics(result: CaseResult) -> list[BaseMetric]:
    if os.getenv("MYAGENT_EVAL_CHECKS_ONLY") == "1":
        return [ChecksMetric(result)]
    judge = get_judge()
    case = result.case

    metrics: list[BaseMetric] = [
        ChecksMetric(result),
        TaskCompletionMetric(
            task=f"{case.input}\n验收要求: {case.criteria or '满足用户要求'}",
            model=judge,
            threshold=0.6,
            async_mode=False,
        ),
        GEval(
            name="Response Quality",
            evaluation_steps=RESPONSE_QUALITY_STEPS,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=judge,
            threshold=0.5,
            async_mode=False,
        ),
    ]
    if case.expected_tools:
        metrics.append(ToolCorrectnessMetric(model=judge, threshold=1.0, async_mode=False))
    if os.getenv("MYAGENT_EVAL_STEP_EFFICIENCY") == "1":
        metrics.append(StepEfficiencyMetric(model=judge, threshold=0.5, async_mode=False))
    return metrics
