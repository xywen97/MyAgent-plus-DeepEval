from __future__ import annotations

import re
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path

from .base import ToolContext, ToolError, is_binary, iter_files, registry

MAX_RESULT_LINES = 500


def _ripgrep(rg: str, root: Path, pattern: str, glob: str | None, mode: str,
             before: int, after: int, around: int, case: bool) -> list[str]:
    cmd = [rg, "--color=never", "--no-heading", "--max-columns=500"]
    if mode == "files_with_matches":
        cmd.append("-l")
    elif mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if around:
            cmd += ["-C", str(around)]
        else:
            if before:
                cmd += ["-B", str(before)]
            if after:
                cmd += ["-A", str(after)]
    if not case:
        cmd.append("-i")
    if glob:
        cmd += ["--glob", glob]

    if root.is_file():
        cmd += ["--with-filename", "--", pattern, str(root)]
        cwd = root.parent
    else:
        cmd += ["--", pattern]
        cwd = root

    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    if proc.returncode == 1:
        return []
    if proc.returncode != 0:
        raise ToolError(proc.stderr.strip() or f"rg 退出码 {proc.returncode}")
    return proc.stdout.splitlines()


def _python_grep(root: Path, pattern: str, glob: str | None, mode: str,
                 before: int, after: int, case: bool) -> list[str]:
    try:
        rx = re.compile(pattern, 0 if case else re.IGNORECASE)
    except re.error as e:
        raise ToolError(f"无效正则: {e}")

    files = [root] if root.is_file() else iter_files(root)
    out: list[str] = []
    for f in files:
        rel = f.name if root.is_file() else str(f.relative_to(root))
        if glob and not (fnmatch(f.name, glob) or fnmatch(rel, glob)):
            continue
        try:
            data = f.read_bytes()
        except OSError:
            continue
        if is_binary(data):
            continue
        lines = data.decode("utf-8", errors="replace").splitlines()
        hits = [i for i, line in enumerate(lines) if rx.search(line)]
        if not hits:
            continue
        if mode == "files_with_matches":
            out.append(rel)
        elif mode == "count":
            out.append(f"{rel}:{len(hits)}")
        else:
            hit_set, shown = set(hits), set()
            for i in hits:
                for j in range(max(0, i - before), min(len(lines), i + after + 1)):
                    if j in shown:
                        continue
                    shown.add(j)
                    sep = ":" if j in hit_set else "-"
                    out.append(f"{rel}:{j + 1}{sep}{lines[j][:500]}")
        if len(out) > MAX_RESULT_LINES:
            break
    return out


@registry.tool(
    name="search_content",
    description=(
        "Search file contents with ripgrep (falls back to Python regex); respects .gitignore when rg is available.\n"
        "- Provide a regex pattern. Escape metacharacters for literal matches.\n"
        "- Narrow path/glob before searching. outputMode: \"content\" (default), \"files_with_matches\", \"count\"."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "The regular expression pattern to search for."},
            "path": {"type": "string", "description": "Absolute file or directory to search. Defaults to workspace root."},
            "glob": {"type": "string", "description": 'Glob to filter files, e.g. "*.py".'},
            "outputMode": {"type": "string", "description": '"content", "files_with_matches" or "count".'},
            "contextBefore": {"type": "number", "description": "Lines before each match (content mode)."},
            "contextAfter": {"type": "number", "description": "Lines after each match (content mode)."},
            "contextAround": {"type": "number", "description": "Lines before and after each match (content mode)."},
            "caseSensitive": {"type": "boolean", "description": "Case-sensitive search. Default false."},
        },
        "required": ["pattern"],
    },
)
def search_content(
    ctx: ToolContext,
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    outputMode: str | None = None,
    contextBefore: int | None = None,
    contextAfter: int | None = None,
    contextAround: int | None = None,
    caseSensitive: bool = False,
) -> str:
    root = ctx.resolve(path) if path else ctx.workspace
    if not root.exists():
        raise ToolError(f"路径不存在: {root}")
    mode = outputMode or "content"
    if mode not in {"content", "files_with_matches", "count"}:
        raise ToolError(f"不支持的 outputMode: {mode}")

    before, after, around = int(contextBefore or 0), int(contextAfter or 0), int(contextAround or 0)
    rg = shutil.which("rg")
    if rg:
        lines = _ripgrep(rg, root, pattern, glob, mode, before, after, around, caseSensitive)
    else:
        if around:
            before = after = around
        lines = _python_grep(root, pattern, glob, mode, before, after, caseSensitive)

    if not lines:
        return "No matches found."
    truncated = len(lines) > MAX_RESULT_LINES
    body = "\n".join(lines[:MAX_RESULT_LINES])
    if truncated:
        body += f"\n\n[结果过多，仅显示前 {MAX_RESULT_LINES} 行，请缩小搜索范围]"
    return f"搜索根目录: {root}\n{body}"
