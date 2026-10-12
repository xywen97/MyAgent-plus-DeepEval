from __future__ import annotations

from fnmatch import fnmatchcase
from pathlib import Path

from .base import ToolContext, ToolError, is_binary, iter_files, matches_any, registry

READ_DEFAULT_LIMIT = 2000
READ_MAX_LINE_CHARS = 2000
SEARCH_FILE_LIMIT = 300


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


@registry.tool(
    name="list_dir",
    description=(
        "Lists files and directories in a given path.\n"
        "- 'target_directory' can be relative to the workspace root or absolute.\n"
        "- The result does not display dot-files and dot-directories."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target_directory": {"type": "string", "description": "Path to directory to list contents of."},
            "ignore_globs": {
                "type": "array",
                "items": {"type": "string"},
                "description": 'Glob exclusions matched anywhere under the directory, e.g. "*.js".',
            },
        },
        "required": ["target_directory"],
    },
)
def list_dir(ctx: ToolContext, target_directory: str, ignore_globs: list[str] | None = None) -> str:
    path = ctx.resolve(target_directory)
    if not path.exists():
        raise ToolError(f"目录不存在: {path}")
    if not path.is_dir():
        raise ToolError(f"不是目录: {path}")

    lines: list[str] = []
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for entry in entries:
        if entry.name.startswith(".") or matches_any(entry.name, ignore_globs):
            continue
        if entry.is_dir():
            try:
                count: int | str = sum(1 for _ in entry.iterdir())
            except PermissionError:
                count = "?"
            lines.append(f"  {entry.name}/ ({count} items)")
        else:
            try:
                size = _human_size(entry.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"  {entry.name} ({size})")
    return f"{path}/\n" + ("\n".join(lines) if lines else "  (空目录)")


@registry.tool(
    name="search_file",
    description=(
        "File search with wildcard pattern matching and recursive directory search; "
        "returns relative file paths and excludes common directories like node_modules and .git."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": 'File pattern (e.g. "*.py", "test_*.py"). Supports wildcards.'},
            "recursive": {"type": "boolean", "description": "Set to true to search in subdirectories."},
            "target_directory": {"type": "string", "description": "Absolute directory to search. Defaults to workspace root."},
            "caseSensitive": {"type": "boolean", "description": "Case-sensitive search. Default false."},
            "ignore_globs": {"type": "array", "items": {"type": "string"}, "description": "Glob exclusions."},
        },
        "required": ["pattern", "recursive"],
    },
)
def search_file(
    ctx: ToolContext,
    pattern: str,
    recursive: bool,
    target_directory: str | None = None,
    caseSensitive: bool = False,
    ignore_globs: list[str] | None = None,
) -> str:
    root = ctx.resolve(target_directory) if target_directory else ctx.workspace
    if not root.is_dir():
        raise ToolError(f"目录不存在: {root}")

    pat = pattern if caseSensitive else pattern.lower()
    results: list[str] = []
    for path in iter_files(root, ignore_globs, recursive=bool(recursive)):
        rel = str(path.relative_to(root))
        name = path.name if caseSensitive else path.name.lower()
        rel_cmp = rel if caseSensitive else rel.lower()
        if fnmatchcase(name, pat) or ("/" in pat and fnmatchcase(rel_cmp, pat)):
            results.append(rel)
            if len(results) >= SEARCH_FILE_LIMIT:
                break

    if not results:
        return f"在 {root} 下未找到匹配 `{pattern}` 的文件。"
    suffix = f"\n\n[结果过多，仅显示前 {SEARCH_FILE_LIMIT} 个]" if len(results) >= SEARCH_FILE_LIMIT else ""
    return f"在 {root} 下找到 {len(results)} 个文件:\n" + "\n".join(results) + suffix


@registry.tool(
    name="read_file",
    description=(
        "Read a local file. Provide absolute filePath.\n"
        "- For a large file, use offset (1-based line number) and limit.\n"
        "- Output lines are prefixed as LINE_NUMBER:CONTENT; the prefix is metadata, not file content.\n"
        "- Empty files return 'File is empty.'"
    ),
    parameters={
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "The path of the file to read."},
            "offset": {"type": "number", "description": "The line number to start reading from (1-based)."},
            "limit": {"type": "number", "description": "The number of lines to read."},
        },
        "required": ["filePath"],
    },
)
def read_file(ctx: ToolContext, filePath: str, offset: int | None = None, limit: int | None = None) -> str:
    path = ctx.resolve(filePath)
    if not path.exists():
        raise ToolError(f"文件不存在: {path}")
    if path.is_dir():
        raise ToolError(f"这是目录，请使用 list_dir: {path}")

    data = path.read_bytes()
    if is_binary(data):
        raise ToolError(f"二进制文件，无法以文本读取: {path}")
    text = data.decode("utf-8", errors="replace")
    if not text:
        return "File is empty."

    lines = text.splitlines()
    start = max(int(offset or 1), 1)
    count = max(int(limit or READ_DEFAULT_LIMIT), 1)
    if start > len(lines):
        raise ToolError(f"offset {start} 超出文件总行数 {len(lines)}")

    chunk = lines[start - 1 : start - 1 + count]
    out = []
    for i, line in enumerate(chunk, start):
        if len(line) > READ_MAX_LINE_CHARS:
            line = line[:READ_MAX_LINE_CHARS] + " ...[行过长已截断]"
        out.append(f"{i:>6}:{line}")
    end = start + len(chunk) - 1
    result = "\n".join(out)
    if start > 1 or end < len(lines):
        result += f"\n\n[显示第 {start}-{end} 行，共 {len(lines)} 行]"
    return result


@registry.tool(
    name="write_to_file",
    description=(
        "Write a complete local file; this will overwrite the existing file.\n"
        "- For an existing file, you MUST read it first.\n"
        "- Prefer replace_in_file for partial changes.\n"
        "- Parent directories are created automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "Target file absolute path."},
            "content": {"type": "string", "description": "The complete content to write to the file."},
            "explanation": {"type": "string", "description": "One sentence explanation of why."},
        },
        "required": ["filePath", "content"],
    },
)
def write_to_file(ctx: ToolContext, filePath: str, content: str, explanation: str | None = None) -> str:
    path = ctx.resolve(filePath)
    ctx.ensure_in_workspace(path)
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    action = "已覆盖" if existed else "已创建"
    return f"{action} {path} ({len(content.splitlines())} 行)"


@registry.tool(
    name="replace_in_file",
    description=(
        "Replace exact text in an existing file; prefer this over rewriting the whole file.\n"
        "- Read the file before editing; after a failed match, re-read before retrying.\n"
        "- Match original whitespace and indentation exactly; strip LINE_NUMBER: prefixes.\n"
        "- old_str must be unique unless replace_all=true.\n"
        "- Empty new_str deletes the match."
    ),
    parameters={
        "type": "object",
        "properties": {
            "filePath": {"type": "string", "description": "Absolute path of an existing workspace file."},
            "old_str": {"type": "string", "description": "Non-empty exact text to match."},
            "new_str": {"type": "string", "description": "Replacement text, different from old_str."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
            "explanation": {"type": "string", "description": "Brief purpose of the edit."},
        },
        "required": ["filePath", "old_str", "new_str"],
    },
)
def replace_in_file(
    ctx: ToolContext,
    filePath: str,
    old_str: str,
    new_str: str,
    replace_all: bool = False,
    explanation: str | None = None,
) -> str:
    path = ctx.resolve(filePath)
    ctx.ensure_in_workspace(path)
    if not path.is_file():
        raise ToolError(f"文件不存在: {path}")
    if not old_str:
        raise ToolError("old_str 不能为空")
    if old_str == new_str:
        raise ToolError("old_str 与 new_str 相同，无需修改")

    text = path.read_bytes().decode("utf-8")
    count = text.count(old_str)
    if count == 0 and "\r\n" in text:
        old_str = old_str.replace("\r\n", "\n").replace("\n", "\r\n")
        new_str = new_str.replace("\r\n", "\n").replace("\n", "\r\n")
        count = text.count(old_str)
    if count == 0:
        raise ToolError("未找到 old_str。请重新 read_file 确认内容（注意空白、缩进和标点）。")
    if count > 1 and not replace_all:
        raise ToolError(f"old_str 匹配到 {count} 处。请提供更多上下文使其唯一，或设置 replace_all=true。")

    new_text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
    path.write_bytes(new_text.encode("utf-8"))
    return f"已在 {path} 中替换 {count if replace_all else 1} 处"


@registry.tool(
    name="delete_file",
    description="Deletes a file at the specified path. Fails gracefully if the file doesn't exist or is outside the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "target_file": {"type": "string", "description": "The absolute path of the file to delete."},
            "explanation": {"type": "string", "description": "Brief purpose of this action."},
        },
        "required": ["target_file"],
    },
)
def delete_file(ctx: ToolContext, target_file: str, explanation: str | None = None) -> str:
    path: Path = ctx.resolve(target_file)
    ctx.ensure_in_workspace(path)
    if not path.exists():
        raise ToolError(f"文件不存在: {path}")
    if path.is_dir():
        raise ToolError(f"不能删除目录: {path}")
    path.unlink()
    return f"已删除 {path}"
