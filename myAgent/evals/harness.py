"""Run MyAgent headlessly on an eval case inside an isolated temp workspace."""
from __future__ import annotations

import difflib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
import time
from dataclasses import dataclass, field
from pathlib import Path

from my_agent.agent import RunRecord
from deepeval.tracing import update_current_trace

from .tracing import TracedAgent, TracedCompletions
from my_agent.config import Config
from my_agent.tools.base import is_binary, iter_files

DATASETS_DIR = Path(__file__).parent / "datasets"
MAX_DIFF_CHARS = 8000


@dataclass
class EvalCase:
    id: str
    input: str
    files: dict[str, str] = field(default_factory=dict)
    expected_tools: list[str] = field(default_factory=list)
    expected_output: str | None = None
    criteria: str | None = None
    checks: list[dict] = field(default_factory=list)
    approve_commands: bool = False
    followup_answer: str | None = None
    max_steps: int = 20
    tags: list[str] = field(default_factory=list)


def load_cases(pattern: str = "*.json") -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in sorted(DATASETS_DIR.glob(pattern)):
        for raw in json.loads(path.read_text(encoding="utf-8")):
            cases.append(EvalCase(**raw))
    ids = [c.id for c in cases]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"评测用例 id 重复: {dup}")
    return cases


class HeadlessUI:
    """Non-interactive UI: records events, answers approvals/questions from the case."""

    def __init__(self, approve_commands: bool, followup_answer: str | None) -> None:
        self.approve_commands = approve_commands
        self.followup_answer = followup_answer
        self.events: list[tuple] = []
        self.denied_commands: list[str] = []
        self.questions_asked = 0
        self.model_calls = 0

    def progress(self, text: str) -> None:
        if os.getenv("MYAGENT_EVAL_PROGRESS", "1") != "0":
            print(f"  {text}", flush=True)

    @contextmanager
    def thinking(self):
        self.model_calls += 1
        start = time.perf_counter()
        self.progress(f"模型调用 #{self.model_calls}：等待响应...")
        try:
            yield
        finally:
            self.progress(f"模型调用 #{self.model_calls} 结束，耗时 {time.perf_counter() - start:.1f}s")

    def assistant(self, text: str) -> None:
        self.events.append(("assistant", text))

    def info(self, text: str) -> None:
        self.events.append(("info", text))

    def error(self, text: str) -> None:
        self.events.append(("error", text))
        self.progress(text)

    def tool_call(self, name: str, args: dict) -> None:
        self.progress(f"执行工具 {name}")

    def tool_result(self, name: str, result: str) -> None:
        outcome = "错误" if result.startswith("Error") else "返回结果"
        self.progress(f"工具 {name}：{outcome}")

    def confirm_command(self, command: str, explanation: str | None) -> bool:
        if not self.approve_commands:
            self.denied_commands.append(command)
        return self.approve_commands

    def show_todos(self, todos: list[dict]) -> None:
        pass

    def ask_questions(self, questions: list[dict], title: str | None) -> list[dict]:
        self.questions_asked += 1
        answer = self.followup_answer or "用户暂时无法回答，请基于合理假设继续完成任务。"
        return [{"id": q["id"], "question": q["question"], "answer": [answer]} for q in questions]


@dataclass
class CaseResult:
    case: EvalCase
    record: RunRecord
    workspace: Path
    before: dict[str, str]
    after: dict[str, str]
    ui: HeadlessUI

    @property
    def diff(self) -> str:
        chunks: list[str] = []
        for path in sorted(set(self.before) | set(self.after)):
            a, b = self.before.get(path), self.after.get(path)
            if a == b:
                continue
            chunks.extend(
                difflib.unified_diff(
                    (a or "").splitlines(keepends=True),
                    (b or "").splitlines(keepends=True),
                    fromfile=f"a/{path}" if a is not None else "/dev/null",
                    tofile=f"b/{path}" if b is not None else "/dev/null",
                )
            )
        text = "".join(chunks)
        if len(text) > MAX_DIFF_CHARS:
            text = text[:MAX_DIFF_CHARS] + "\n... [diff 已截断]"
        return text or "(工作区无文件变更)"

    def cleanup(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)


def snapshot(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in iter_files(root):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not is_binary(data):
            files[str(path.relative_to(root))] = data.decode("utf-8", errors="replace")
    return files


def run_case(case: EvalCase, base_cfg: Config) -> CaseResult:
    workspace = Path(tempfile.mkdtemp(prefix=f"myagent_eval_{case.id}_")).resolve()
    for rel, content in case.files.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    before = snapshot(workspace)

    cfg = Config(
        api_key=base_cfg.api_key,
        base_url=base_cfg.base_url,
        model=base_cfg.model,
        workspace=workspace,
        max_steps=case.max_steps,
        command_timeout=base_cfg.command_timeout,
        auto_approve=False,
        max_tool_output=base_cfg.max_tool_output,
        language=base_cfg.language,
    )
    ui = HeadlessUI(case.approve_commands, case.followup_answer)
    agent = TracedAgent(cfg, ui)
    agent.client.chat.completions = TracedCompletions(agent.client.chat.completions)
    try:
        agent.run(case.input.replace("{workspace}", str(workspace)))
        assert agent.last_run is not None
        result = CaseResult(case, agent.last_run, workspace, before, snapshot(workspace), ui)
        update_current_trace(
            name=case.id,
            input=case.input,
            output=agent.eval_span.output,
            tools_called=agent.eval_span.tools_called,
            expected_output=case.expected_output,
            context=[f"任务验收标准: {case.criteria or case.input}", f"工作区最终文件 diff:\n{result.diff}"],
            tags=case.tags,
        )
        return result
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
