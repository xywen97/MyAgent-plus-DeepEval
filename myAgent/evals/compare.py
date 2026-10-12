"""对比两次 deepeval 评测结果，发现回归。

用法:
  python -m evals.compare                    # 最近两次
  python -m evals.compare BASE.json NEW.json  # 指定两次
  python -m evals.compare --fail-on-regression
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

RESULTS_DIR = Path(os.getenv("DEEPEVAL_RESULTS_FOLDER") or Path(__file__).parent / "results")
REGRESSION_DELTA = 0.1


def load_run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases: dict[str, dict] = {}
    for tc in data.get("testCases") or []:
        name = tc.get("name") or tc.get("input", "")[:40]
        metrics = {m["name"]: m for m in tc.get("metricsData") or []}
        cases[name] = {"success": bool(tc.get("success")), "metrics": metrics, "metadata": tc.get("metadata") or {}}
    return {"path": path, "hyper": data.get("hyperparameters") or {}, "cases": cases}


def _fmt(score) -> str:
    return "-" if score is None else f"{score:.2f}"


def _delta(old, new) -> str:
    if old is None or new is None:
        return ""
    d = new - old
    if abs(d) < 1e-9:
        return "[dim]=[/dim]"
    color = "green" if d > 0 else "red"
    return f"[{color}]{d:+.2f}[/{color}]"


def summarize(run: dict) -> tuple[int, int]:
    total = len(run["cases"])
    return sum(c["success"] for c in run["cases"].values()), total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="对比两次 MyAgent 评测结果")
    parser.add_argument("runs", nargs="*", type=Path, help="BASE.json NEW.json；默认取 results 目录最近两次")
    parser.add_argument("--fail-on-regression", action="store_true", help="存在回归时以退出码 1 结束")
    args = parser.parse_args(argv)
    console = Console()

    if len(args.runs) == 2:
        base_path, new_path = args.runs
    elif not args.runs:
        files = sorted(RESULTS_DIR.glob("test_run_*.json"), key=lambda p: p.stat().st_mtime)
        if len(files) < 2:
            console.print(f"[yellow]{RESULTS_DIR} 下评测结果不足两次，无法对比。[/yellow]")
            return 0
        base_path, new_path = files[-2], files[-1]
    else:
        parser.error("需要 0 个或 2 个结果文件")

    base, new = load_run(base_path), load_run(new_path)

    for label, run in (("BASE", base), ("NEW ", new)):
        h = run["hyper"]
        passed, total = summarize(run)
        console.print(
            f"[bold]{label}[/bold] {run['path'].name}  pass {passed}/{total}  "
            f"[dim]version={h.get('agent_version')} commit={h.get('git_commit')} model={h.get('model')} "
            f"prompt={h.get('system_prompt_sha')} {h.get('run_label') or ''}[/dim]"
        )

    metric_names = sorted({m for r in (base, new) for c in r["cases"].values() for m in c["metrics"]})
    table = Table(show_lines=False)
    table.add_column("case")
    table.add_column("pass")
    for m in metric_names:
        table.add_column(m, justify="right")
    table.add_column("steps", justify="right")

    regressions: list[str] = []
    for name in sorted(set(base["cases"]) | set(new["cases"])):
        b, n = base["cases"].get(name), new["cases"].get(name)
        if n is None:
            table.add_row(name, "[dim]removed[/dim]", *[""] * len(metric_names), "")
            continue
        if b is None:
            status = "[cyan]new[/cyan] " + ("[green]PASS[/green]" if n["success"] else "[red]FAIL[/red]")
        elif b["success"] and not n["success"]:
            status = "[red]PASS→FAIL[/red]"
            regressions.append(f"{name}: 由通过变为失败")
        elif not b["success"] and n["success"]:
            status = "[green]FAIL→PASS[/green]"
        else:
            status = "[green]PASS[/green]" if n["success"] else "[red]FAIL[/red]"

        cells = []
        for m in metric_names:
            old_s = (b or {}).get("metrics", {}).get(m, {}).get("score") if b else None
            new_s = n["metrics"].get(m, {}).get("score")
            cells.append(f"{_fmt(new_s)} {_delta(old_s, new_s)}".strip())
            if old_s is not None and new_s is not None and old_s - new_s >= REGRESSION_DELTA:
                regressions.append(f"{name}: {m} {old_s:.2f} → {new_s:.2f}")
        old_steps = (b or {}).get("metadata", {}).get("steps") if b else None
        new_steps = n["metadata"].get("steps")
        steps = "-" if new_steps is None else str(new_steps)
        if old_steps is not None and new_steps is not None and old_steps != new_steps:
            steps += f" ({new_steps - old_steps:+d})"
        table.add_row(name, status, *cells, steps)

    console.print(table)
    if regressions:
        console.print("[bold red]发现回归:[/bold red]")
        for r in regressions:
            console.print(f"  - {r}")
    else:
        console.print("[green]未发现回归。[/green]")
    return 1 if regressions and args.fail_on_regression else 0


if __name__ == "__main__":
    sys.exit(main())
