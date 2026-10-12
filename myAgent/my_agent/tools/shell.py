from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .base import ToolContext, clip, registry

OUTPUT_LIMIT = 12000


def command_environment() -> dict[str, str]:
    env = {**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if sys.prefix != sys.base_prefix:
        env["VIRTUAL_ENV"] = sys.prefix
        env.pop("PYTHONHOME", None)
    return env


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


@registry.tool(
    name="execute_command",
    description=(
        "Run a command in the user's shell, subject to approval.\n"
        "- Prefer dedicated file/search/edit tools when they fit. Use commands for builds, tests, Git and tasks those tools cannot handle.\n"
        "- Each call starts a NEW shell at the workspace root; include needed setup (e.g. `cd dir && ...`).\n"
        "- Use non-interactive flags; disable pagers (e.g. git --no-pager).\n"
        "- Destructive, risky or outside-workspace commands MUST set requires_approval=true. Do not work around a denial.\n"
        "- Never change git config. Commit only when explicitly requested."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The CLI command to execute."},
            "explanation": {"type": "string", "description": "Brief purpose of this action."},
            "requires_approval": {
                "type": "boolean",
                "default": True,
                "description": "Set true for destructive, risky or outside-workspace operations; false for safe operations.",
            },
        },
        "required": ["command"],
    },
)
def execute_command(
    ctx: ToolContext,
    command: str,
    explanation: str | None = None,
    requires_approval: bool = True,
) -> str:
    if requires_approval and not ctx.auto_approve:
        if not ctx.ui.confirm_command(command, explanation):
            return "用户拒绝执行该命令。不要重试或换方式绕过，请询问用户下一步怎么做。"

    shell = os.environ.get("SHELL") or "/bin/sh"
    env = command_environment()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable=shell,
            cwd=ctx.workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=ctx.command_timeout,
        )
    except subprocess.TimeoutExpired as e:
        return json.dumps(
            {
                "timedOut": True,
                "message": f"命令超过 {ctx.command_timeout}s 被终止",
                "stdout": clip(_to_text(e.stdout), OUTPUT_LIMIT),
                "stderr": clip(_to_text(e.stderr), OUTPUT_LIMIT),
            },
            ensure_ascii=False,
            indent=2,
        )

    return json.dumps(
        {
            "exitCode": proc.returncode,
            "stdout": clip(proc.stdout, OUTPUT_LIMIT),
            "stderr": clip(proc.stderr, OUTPUT_LIMIT),
        },
        ensure_ascii=False,
        indent=2,
    )
