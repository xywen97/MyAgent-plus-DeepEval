# MyAgent

一个运行在终端里的 AI 编码助手，参考 CodeBuddy 的 system prompt 与工具设计，基于 OpenAI Chat Completions 协议（兼容 OpenAI / DeepSeek / 通义 / Kimi 等）。

## 快速开始

```bash
cd myAgent
source .venv/bin/activate          # Python 3.12
uv pip install -r requirements.txt
cp .env.example .env               # 填写 MYAGENT_API_KEY / MYAGENT_BASE_URL / MYAGENT_MODEL

python -m my_agent -w /path/to/project          # 交互模式
python -m my_agent -w /path/to/project "解释一下这个项目的结构"   # 单次任务
```

交互命令：`/help`、`/reset`、`/todos`、`/exit`。

## 结构

```
my_agent/
  __main__.py      CLI 入口与 REPL
  agent.py         Agent 主循环（模型调用 -> 工具执行 -> 回填结果）
  prompts.py       System Prompt
  config.py        配置（.env / 环境变量 / 命令行参数）
  ui.py            终端渲染、命令确认、提问交互
  tools/
    base.py        工具注册表、ToolContext
    fs.py          list_dir / search_file / read_file / write_to_file / replace_in_file / delete_file
    search.py      search_content（优先 ripgrep，回退 Python 正则）
    shell.py       execute_command（需审批）
    interaction.py todo_write / ask_followup_question
```

## 扩展工具

```python
from my_agent.tools.base import registry, ToolContext

@registry.tool(name="hello", description="Say hello.",
               parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
def hello(ctx: ToolContext, name: str) -> str:
    return f"hello {name}"
```

然后在 `my_agent/tools/__init__.py` 中导入该模块即可。

## 安全

- 写入/删除仅限工作区内。
- `requires_approval=true` 的命令需要在终端确认；`--auto-approve` 会跳过确认，谨慎使用。
