from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import Config
from .ui import ConsoleUI

HELP = """命令:
  /help   显示帮助
  /reset  清空对话上下文
  /todos  查看当前任务清单
  /exit   退出"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="myagent", description="MyAgent - 终端里的 AI 编码助手")
    parser.add_argument("prompt", nargs="*", help="单次执行的任务；不提供则进入交互模式")
    parser.add_argument("-w", "--workspace", help="工作区目录（默认当前目录）")
    parser.add_argument("-m", "--model", help="模型名称")
    parser.add_argument("--auto-approve", action="store_true", default=None, help="自动批准所有命令（谨慎使用）")
    args = parser.parse_args(argv)

    ui = ConsoleUI()
    cfg = Config.from_env(workspace=args.workspace, model=args.model, auto_approve=args.auto_approve)
    if not cfg.api_key:
        ui.error("未设置 API Key：请在 .env 中配置 MYAGENT_API_KEY（参考 .env.example）。")
        return 1
    if not cfg.workspace.is_dir():
        ui.error(f"工作区不存在: {cfg.workspace}")
        return 1

    agent = Agent(cfg, ui)

    if args.prompt:
        agent.run(" ".join(args.prompt))
        return 0

    try:
        import readline  # noqa: F401  (输入历史与行编辑)
    except ImportError:
        pass

    ui.console.print(f"[bold]MyAgent[/bold] [dim]model={cfg.model}  workspace={cfg.workspace}[/dim]")
    ui.console.print("[dim]输入 /help 查看命令，Ctrl+C 中断当前任务，Ctrl+D 退出。[/dim]")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return 0
        if not text:
            continue
        if text in {"/exit", "/quit"}:
            return 0
        if text == "/help":
            ui.console.print(HELP)
            continue
        if text == "/reset":
            agent.reset()
            ui.info("上下文已清空。")
            continue
        if text == "/todos":
            if agent.ctx.todos:
                ui.show_todos(agent.ctx.todos)
            else:
                ui.info("暂无任务。")
            continue
        agent.run(text)


if __name__ == "__main__":
    sys.exit(main())
