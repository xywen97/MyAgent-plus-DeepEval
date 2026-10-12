from __future__ import annotations

import json
from typing import Any

from openai import OpenAI, OpenAIError

from .config import Config
from .prompts import build_system_prompt
from .tools import ToolContext, registry


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
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict] = [{"role": "system", "content": build_system_prompt(self.cfg)}]
        self.ctx.todos = []

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": f"<user_query>\n{user_input}\n</user_query>"})
        try:
            return self._loop()
        except KeyboardInterrupt:
            self._close_dangling_tool_calls("用户中断了执行。")
            self.ui.info("已中断。")
            return ""

    def _loop(self) -> str:
        for _ in range(self.cfg.max_steps):
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
                return ""

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
        return ""

    def _run_tool(self, name: str, raw_args: str | None) -> str:
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments 必须是 JSON 对象")
        except (json.JSONDecodeError, ValueError) as e:
            result = f"Error: 工具参数不是合法 JSON: {e}"
            self.ui.tool_call(name, {})
            self.ui.tool_result(name, result)
            return result

        self.ui.tool_call(name, args)
        result = registry.execute(name, args, self.ctx, self.cfg.max_tool_output)
        self.ui.tool_result(name, result)
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
