from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI, OpenAIError

from .config import Config
from .prompts import build_system_prompt
from .tools import ToolContext, registry


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    output: str
    is_error: bool


@dataclass
class RunRecord:
    """Trajectory of a single `Agent.run`, used for evaluation and debugging."""

    input: str
    final_output: str = ""
    stop_reason: str = "completed"  # completed | max_steps | error | interrupted
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_s: float = 0.0
    error: str | None = None


class Agent:
    """A minimal tool-calling agent loop on the OpenAI Chat Completions protocol."""

    def __init__(self, cfg: Config, ui: Any) -> None:
        self.cfg = cfg
        self.ui = ui
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.ctx = ToolContext(
            workspace=cfg.workspace,
            ui=ui,
            command_timeout=cfg.command_timeout,
            auto_approve=cfg.auto_approve,
        )
        self.tools = registry.schemas()
        self.last_run: RunRecord | None = None
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict] = [{"role": "system", "content": build_system_prompt(self.cfg)}]
        self.ctx.todos = []

    def run(self, user_input: str) -> str:
        self.last_run = RunRecord(input=user_input)
        self.messages.append({"role": "user", "content": f"<user_query>\n{user_input}\n</user_query>"})
        start = time.perf_counter()
        try:
            output = self._loop()
        except KeyboardInterrupt:
            self._close_dangling_tool_calls("用户中断了执行。")
            self.ui.info("已中断。")
            self.last_run.stop_reason = "interrupted"
            output = ""
        self.last_run.final_output = output
        self.last_run.duration_s = time.perf_counter() - start
        return output

    def _loop(self) -> str:
        record = self.last_run
        for _ in range(self.cfg.max_steps):
            record.steps += 1
            try:
                with self.ui.thinking():
                    resp = self.client.chat.completions.create(
                        model=self.cfg.model,
                        messages=self.messages,
                        tools=self.tools,
                        tool_choice="auto",
                    )
            except OpenAIError as e:
                self.ui.error(f"模型调用失败: {e}")
                record.stop_reason, record.error = "error", str(e)
                return ""

            usage = getattr(resp, "usage", None)
            if usage is not None:
                record.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                record.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            msg = resp.choices[0].message
            entry: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                    }
                    for tc in msg.tool_calls
                ]
            self.messages.append(entry)

            if msg.content:
                self.ui.assistant(msg.content)
            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                result = self._run_tool(tc.function.name, tc.function.arguments)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        self.ui.error(f"已达到最大步数 {self.cfg.max_steps}，停止执行。")
        record.stop_reason = "max_steps"
        return ""

    def _run_tool(self, name: str, raw_args: str | None) -> str:
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments 必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as e:
            args = {}
            result = f"Error: 工具参数不是合法 JSON: {e}"
            self.ui.tool_call(name, args)
        else:
            self.ui.tool_call(name, args)
            result = registry.execute(name, args, self.ctx, self.cfg.max_tool_output)

        self.ui.tool_result(name, result)
        if self.last_run is not None:
            self.last_run.tool_calls.append(
                ToolCallRecord(name=name, args=args, output=result, is_error=result.startswith("Error"))
            )
        return result

    def _close_dangling_tool_calls(self, note: str) -> None:
        """Ensure every assistant tool_call has a tool response so the history stays valid."""
        for idx in range(len(self.messages) - 1, -1, -1):
            m = self.messages[idx]
            if m["role"] == "assistant":
                answered = {x["tool_call_id"] for x in self.messages[idx + 1 :] if x["role"] == "tool"}
                for tc in m.get("tool_calls", []):
                    if tc["id"] not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": note})
                return
