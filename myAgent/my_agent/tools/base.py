from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Iterator

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
}


class ToolError(Exception):
    """Expected tool failure; message is returned to the model."""


@dataclass
class ToolContext:
    workspace: Path
    ui: Any
    command_timeout: int = 120
    auto_approve: bool = False
    todos: list[dict] = field(default_factory=list)

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.workspace / p
        return p.resolve()

    def ensure_in_workspace(self, path: Path) -> None:
        try:
            path.relative_to(self.workspace)
        except ValueError:
            raise ToolError(f"路径不在工作区内，拒绝修改: {path}")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, name: str, description: str, parameters: dict):
        def decorator(func: Callable[..., Any]):
            self._tools[name] = Tool(name, description, parameters, func)
            return func

        return decorator

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def execute(self, name: str, args: dict, ctx: ToolContext, max_output: int = 30000) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: 未知工具 `{name}`。可用工具: {', '.join(self._tools)}"

        props = tool.parameters.get("properties", {})
        missing = [r for r in tool.parameters.get("required", []) if r not in args]
        if missing:
            return f"Error: 缺少必填参数: {', '.join(missing)}"
        kwargs = {k: v for k, v in args.items() if k in props}

        try:
            result = tool.func(ctx, **kwargs)
        except ToolError as e:
            return f"Error: {e}"
        except TypeError as e:
            return f"Error: 参数不合法: {e}"
        except Exception as e:  # noqa: BLE001
            return f"Error: {type(e).__name__}: {e}"

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, indent=2)
        if len(result) > max_output:
            result = result[:max_output] + f"\n\n[输出过长，已截断，原始长度 {len(result)} 字符。请缩小范围后重试。]"
        return result


registry = ToolRegistry()


def matches_any(rel_path: str, globs: list[str] | None) -> bool:
    if not globs:
        return False
    name = os.path.basename(rel_path)
    for g in globs:
        full = g if g.startswith("**/") else f"**/{g}"
        if fnmatch(rel_path, g) or fnmatch(rel_path, full) or fnmatch(name, g):
            return True
    return False


def iter_files(root: Path, ignore_globs: list[str] | None = None, recursive: bool = True) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in DEFAULT_IGNORED_DIRS
            and not matches_any(str((base / d).relative_to(root)), ignore_globs)
        )
        for fname in sorted(filenames):
            path = base / fname
            if not matches_any(str(path.relative_to(root)), ignore_globs):
                yield path
        if not recursive:
            break


def is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head
    return f"{text[:head]}\n... [省略 {len(text) - limit} 字符] ...\n{text[-tail:]}"
