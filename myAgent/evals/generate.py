"""Synthesizer 生成候选题；人工审核后导入正式题库，不执行生成的命令。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig, FiltrationConfig, EvolutionConfig

from my_agent.config import PROJECT_ROOT
from my_agent.prompts import SYSTEM_PROMPT_TEMPLATE
from my_agent.tools import registry

from .harness import EvalCase, load_cases
from .judge import get_judge

DRAFTS = PROJECT_ROOT / "evals" / "drafts"
FORMAT = '''只返回合法 JSON 对象，不要 Markdown：
{"files":{"相对路径":"初始文件内容"},
 "reference_files":{"相对路径":"完成任务后的参考实现"},
 "checks":[{"type":"file_exists","path":"相对路径"},
           {"type":"command","cmd":"{python} -m unittest discover -s tests"}],
 "expected_tools":["read_file"],"criteria":"具体验收要求"}
请提供可重复运行的 Python 标准库 unittest 测试，并放在 files 的 tests/ 目录中。
参考实现必须满足 input，测试要覆盖普通输入和边界情况，且不能要求实现文本完全一致。
用 unchanged 检查保护 tests 中的测试文件，避免被测 Agent 修改验收测试。
只使用标准库和工作区相对路径。禁止网络、安装依赖、系统路径、删除命令和读写真实数据。
不要把参考实现透露给 Agent。'''


def parse_spec(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value)
    spec = json.loads(value)
    if not isinstance(spec, dict):
        raise ValueError("期望输出必须是 JSON 对象")
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    for key in ("files", "reference_files"):
        if not isinstance(spec.get(key), dict) or not spec[key]:
            raise ValueError(f"{key} 必须是非空文件映射")
        for rel, content in spec[key].items():
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts or not isinstance(content, str):
                raise ValueError(f"非法候选文件路径或内容: {rel}")
    if not isinstance(spec.get("checks"), list) or not spec["checks"]:
        raise ValueError("必须有验收检查")
    supported = {"completed", "file_exists", "file_not_exists", "file_contains", "file_not_contains",
                 "unchanged", "output_contains_any", "tool_called", "tool_not_called", "max_tool_calls", "command"}
    for check in spec["checks"]:
        if not isinstance(check, dict) or check.get("type") not in supported:
            raise ValueError(f"未知检查: {check}")
    for name in spec.get("expected_tools", []):
        if name not in registry.names():
            raise ValueError(f"未知工具: {name}")


def generate(args) -> Path:
    judge = get_judge()
    contexts = [[SYSTEM_PROMPT_TEMPLATE, json.dumps(registry.schemas(), ensure_ascii=False)]]
    for source in args.context:
        path = source.resolve()
        if path.suffix not in {".py", ".md", ".txt", ".json"}:
            raise ValueError(f"只支持明确指定的文本上下文: {path}")
        text = path.read_text(encoding="utf-8")
        if len(text) > 30000:
            raise ValueError(f"上下文过长，请选择更小的文件: {path}")
        contexts[0].append(f"资料 {path.name}:\n{text}")
    synthesizer = Synthesizer(
        model=judge, async_mode=False,
        filtration_config=FiltrationConfig(critic_model=judge),
        evolution_config=EvolutionConfig(num_evolutions=1),
        styling_config=StylingConfig(
            scenario="在隔离临时目录工作的 Python 编码 Agent，有文件读写、搜索、命令工具。",
            task="生成不同的、可用标准库测试验收的编程任务，覆盖创建、修复、只读分析。",
            input_format="简体中文用户任务，仅描述需求和已有文件，不包含参考答案；明确要求不修改测试文件。",
            expected_output_format=FORMAT,
        ),
    )
    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts, max_goldens_per_context=args.count,
        include_expected_output=True, _send_data=False,
    )
    drafts = []
    for index, golden in enumerate(goldens, 1):
        raw = golden.expected_output or ""
        try:
            spec = parse_spec(raw)
            error = None
        except (ValueError, TypeError) as exc:
            spec, error = None, str(exc)
        drafts.append({
            "id": f"generated_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{index}",
            "reviewed": False, "input": golden.input, "spec": spec, "parse_error": error,
            "raw_expected_output": raw,
            "provenance": {"generator": "deepeval.Synthesizer", "model": judge.get_model_name(),
                           "context_sha": hashlib.sha256(json.dumps(contexts).encode()).hexdigest()[:12],
                           "source_files": [str(p) for p in args.context]},
        })
    output = args.output or DRAFTS / f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if output.exists():
        raise ValueError(f"目标文件已存在，拒绝覆盖: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 {len(drafts)} 道候选题: {output}，解析失败 {sum(d['parse_error'] is not None for d in drafts)} 道。")
    print("未执行任何参考代码或命令。请审核 spec，并自行在沙箱验证参考实现及测试，再设置 reviewed=true。")
    return output


def import_reviewed(path: Path) -> Path:
    drafts = json.loads(path.read_text(encoding="utf-8"))
    existing = {c.id for c in load_cases()}
    cases = []
    for draft in drafts:
        if draft.get("reviewed") is not True:
            continue
        spec = draft.get("spec")
        if not isinstance(spec, dict):
            raise ValueError(f"{draft['id']} 缺少合法 spec")
        validate_spec(spec)
        if draft["id"] in existing:
            raise ValueError(f"重复 id: {draft['id']}")
        existing.add(draft["id"])
        case = {"id": draft["id"], "input": draft["input"], "files": spec["files"],
                "checks": spec["checks"], "expected_tools": spec.get("expected_tools", []),
                "criteria": spec.get("criteria"), "tags": ["generated", "reviewed"],
                "approve_commands": False}
        EvalCase(**case)
        cases.append(case)
    if not cases:
        raise ValueError("没有 reviewed=true 的候选题")
    output = PROJECT_ROOT / "evals" / "datasets" / f"{path.stem}_reviewed.json"
    if output.exists():
        raise ValueError(f"目标文件已存在，拒绝覆盖: {output}")
    output.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导入 {len(cases)} 道审核题: {output}，参考答案仅保留在草稿中。")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--context", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--import-reviewed", type=Path)
    args = parser.parse_args()
    if args.import_reviewed:
        import_reviewed(args.import_reviewed)
    else:
        if not 1 <= args.count <= 30:
            parser.error("count 必须在 1 到 30 之间")
        generate(args)


if __name__ == "__main__":
    main()
