"""Metrics for MyAgent evals: deterministic checks + deepeval LLM metrics."""
from __future__ import annotations

import subprocess
import sys

from deepeval.metrics import BaseMetric, GEval, ToolCorrectnessMetric
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
        paths = spec.get("paths") or list(before)
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
        cmd = spec["cmd"].replace("{python}", sys.executable)
        proc = subprocess.run(cmd, shell=True, cwd=result.workspace, capture_output=True, text=True, timeout=60)
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


TASK_COMPLETION_STEPS = [
    "根据 Input 判断用户真正要求完成的任务和隐含约束。",
    "结合 Tools Called（工具调用及其结果）与 Context（工作区最终文件 diff）判断任务是否真正完成，而不是只看 Actual Output 的自述。",
    "如果 Actual Output 声称完成了某事但 diff 或工具结果不支持，应严重扣分。",
    "做了用户没要求的额外修改、或越权操作（如被拒绝后绕过）应扣分。",
]

RESPONSE_QUALITY_STEPS = [
    "Actual Output 是否使用简体中文，并直接、简洁地回应了 Input。",
    "是否避免了无关内容、冗长叙述和不必要的大段代码输出（除非用户要求）。",
    "是否如实说明了结果，包括失败、被拒绝或未验证的部分。",
]


def build_metrics(result: CaseResult) -> list[BaseMetric]:
    judge = get_judge()
    case = result.case
    steps = list(TASK_COMPLETION_STEPS)
    if case.criteria:
        steps.append(f"本用例的具体判定标准：{case.criteria}")

    metrics: list[BaseMetric] = [
        ChecksMetric(result),
        GEval(
            name="Task Completion",
            evaluation_steps=steps,
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.TOOLS_CALLED,
                SingleTurnParams.CONTEXT,
            ],
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
    return metrics
