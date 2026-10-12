from __future__ import annotations

import json

from .base import ToolContext, ToolError, registry

TODO_STATUSES = {"pending", "in_progress", "completed", "cancelled"}


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


@registry.tool(
    name="todo_write",
    description=(
        "Create and manage a structured task list for the current session.\n"
        "Use proactively for complex multi-step tasks (3+ steps). Skip for trivial tasks.\n"
        "- States: pending, in_progress, completed, cancelled. Only ONE in_progress at a time.\n"
        "- Mark a task in_progress when starting it, completed IMMEDIATELY after finishing.\n"
        "- First creation uses merge=false; later updates use merge=true and include only changed items (with content).\n"
        "- Never include linting, testing or codebase searching as todos."
    ),
    parameters={
        "type": "object",
        "properties": {
            "merge": {"type": "boolean", "description": "Merge into existing todos by id (true) or replace them (false)."},
            "todos": {
                "type": "string",
                "description": 'JSON string array, each item has id, status, content. e.g. \'[{"id":"1","status":"in_progress","content":"Add login"}]\'',
            },
        },
        "required": ["merge", "todos"],
    },
)
def todo_write(ctx: ToolContext, merge, todos) -> str:
    try:
        items = json.loads(todos) if isinstance(todos, str) else todos
    except json.JSONDecodeError as e:
        raise ToolError(f"todos 不是合法 JSON: {e}")
    if not isinstance(items, list) or not items:
        raise ToolError("todos 必须是非空数组")
    for it in items:
        if not isinstance(it, dict) or not {"id", "status", "content"} <= it.keys():
            raise ToolError(f"每个 todo 必须包含 id/status/content: {it}")
        if it["status"] not in TODO_STATUSES:
            raise ToolError(f"非法状态 {it['status']}，可选: {', '.join(sorted(TODO_STATUSES))}")
        it["id"] = str(it["id"])

    if _as_bool(merge):
        index = {t["id"]: t for t in ctx.todos}
        for it in items:
            if it["id"] in index:
                index[it["id"]].update(it)
            else:
                ctx.todos.append(dict(it))
    else:
        ctx.todos = [dict(it) for it in items]

    ctx.ui.show_todos(ctx.todos)
    lines = [f"[{t['status']}] {t['id']}. {t['content']}" for t in ctx.todos]
    return "Todos 已更新:\n" + "\n".join(lines)


@registry.tool(
    name="ask_followup_question",
    description=(
        "Collect structured multiple-choice answers from the user. Provide 1-4 questions, each with 2-4 options; "
        "set multiSelect when appropriate. The user may also type a custom answer. "
        "Only use when highly uncertain and blocked; prefer resolving through reasoning and tools."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Optional title for the questions form."},
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "header": {"type": "string", "description": "Very short label (max 12 chars)."},
                        "question": {"type": "string"},
                        "multiSelect": {"type": "boolean"},
                        "options": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},
                                            "description": {"type": "string"},
                                        },
                                        "required": ["label"],
                                    },
                                ]
                            },
                        },
                    },
                    "required": ["question", "options"],
                },
            },
        },
        "required": ["questions"],
    },
)
def ask_followup_question(ctx: ToolContext, questions: list[dict], title: str | None = None) -> str:
    if not isinstance(questions, list) or not 1 <= len(questions) <= 4:
        raise ToolError("questions 必须包含 1-4 个问题")
    for i, q in enumerate(questions):
        if not isinstance(q, dict) or "question" not in q or not q.get("options"):
            raise ToolError(f"第 {i + 1} 个问题缺少 question 或 options")
        q.setdefault("id", f"q{i + 1}")
        q["options"] = [o if isinstance(o, dict) else {"label": str(o)} for o in q["options"]]

    answers = ctx.ui.ask_questions(questions, title)
    return json.dumps({"answers": answers}, ensure_ascii=False, indent=2)
