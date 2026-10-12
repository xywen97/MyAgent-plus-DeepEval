from __future__ import annotations

import json
from contextlib import contextmanager

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm

SUMMARY_KEYS = ("command", "filePath", "target_file", "target_directory", "path", "pattern")
TODO_ICONS = {"completed": "[green][x][/green]", "in_progress": "[yellow][>][/yellow]",
              "pending": "[ ]", "cancelled": "[dim][-][/dim]"}


class ConsoleUI:
    def __init__(self) -> None:
        self.console = Console()

    @contextmanager
    def thinking(self):
        with self.console.status("[dim]思考中...[/dim]", spinner="dots"):
            yield

    def stream_text(self, text: str) -> None:
        self.console.print(text, end="", markup=False, highlight=False, soft_wrap=True)
        self._stream_active = True

    def stream_end(self) -> None:
        if getattr(self, "_stream_active", False):
            self.console.print()
        self._stream_active = False

    def assistant(self, text: str) -> None:
        if text.strip():
            self.console.print(Markdown(text))

    def info(self, text: str) -> None:
        self.console.print(f"[dim]{escape(text)}[/dim]")

    def error(self, text: str) -> None:
        self.console.print(f"[red]{escape(text)}[/red]")

    def tool_call(self, name: str, args: dict) -> None:
        summary = next((str(args[k]) for k in SUMMARY_KEYS if k in args), "")
        if not summary and args:
            summary = json.dumps(args, ensure_ascii=False)
        if len(summary) > 120:
            summary = summary[:117] + "..."
        self.console.print(f"[bold magenta]●[/bold magenta] [bold]{name}[/bold] [cyan]{escape(summary)}[/cyan]")

    def tool_result(self, name: str, result: str) -> None:
        first = result.strip().splitlines()[0] if result.strip() else "(无输出)"
        if len(first) > 120:
            first = first[:117] + "..."
        n = result.count("\n") + 1
        style = "red" if result.startswith("Error") else "dim"
        self.console.print(f"  [{style}]└ {escape(first)}[/{style}]" + (f" [dim](+{n - 1} 行)[/dim]" if n > 1 else ""))

    def confirm_command(self, command: str, explanation: str | None) -> bool:
        body = f"[bold]{escape(command)}[/bold]"
        if explanation:
            body += f"\n[dim]{escape(explanation)}[/dim]"
        self.console.print(Panel(body, title="需要确认执行命令", border_style="yellow"))
        try:
            return Confirm.ask("允许执行?", default=False, console=self.console)
        except (EOFError, KeyboardInterrupt):
            return False

    def show_todos(self, todos: list[dict]) -> None:
        lines = [f"{TODO_ICONS.get(t['status'], '[ ]')} {escape(t['content'])}" for t in todos]
        self.console.print(Panel("\n".join(lines), title="Todos", border_style="blue", expand=False))

    def ask_questions(self, questions: list[dict], title: str | None) -> list[dict]:
        if title:
            self.console.print(f"[bold]{escape(title)}[/bold]")
        answers = []
        for q in questions:
            header = f"[{escape(q['header'])}] " if q.get("header") else ""
            self.console.print(f"\n[bold cyan]?[/bold cyan] {header}{escape(q['question'])}")
            options = q["options"]
            for i, opt in enumerate(options, 1):
                desc = f" [dim]- {escape(opt['description'])}[/dim]" if opt.get("description") else ""
                self.console.print(f"  {i}. {escape(opt['label'])}{desc}")
            multi = bool(q.get("multiSelect"))
            hint = "输入序号（多选用逗号分隔），或直接输入自定义回答" if multi else "输入序号，或直接输入自定义回答"
            try:
                raw = self.console.input(f"[dim]{hint}:[/dim] ").strip()
            except (EOFError, KeyboardInterrupt):
                raw = ""
            answers.append({"id": q["id"], "question": q["question"], "answer": self._parse_choice(raw, options, multi)})
        return answers

    @staticmethod
    def _parse_choice(raw: str, options: list[dict], multi: bool) -> list[str]:
        if not raw:
            return []
        parts = [p.strip() for p in raw.replace("，", ",").split(",")] if multi else [raw]
        if all(p.isdigit() and 1 <= int(p) <= len(options) for p in parts):
            return [options[int(p) - 1]["label"] for p in parts]
        return [raw]
