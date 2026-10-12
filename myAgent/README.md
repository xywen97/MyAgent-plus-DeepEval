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

## 评测（deepeval）

每次修改 Agent（prompt、工具、主循环、模型）后运行：

```bash
uv pip install --python .venv/bin/python -r requirements-eval.txt   # 首次
./eval.sh                          # 全量评测 + 与上次结果对比
./eval.sh -k fix_bug               # 只跑部分用例（pytest -k）
./eval.sh --label "prompt-v2"      # 给本次运行打标签
.venv/bin/python -m evals.compare A.json B.json   # 对比任意两次
```

流程：每个用例在独立的临时工作区中无头运行 Agent（命令审批、提问由用例配置自动应答），把最终回复、工具调用轨迹和文件 diff 转成 deepeval 的 `LLMTestCase`，用以下指标打分：

| 指标 | 类型 | 说明 |
|---|---|---|
| `Checks` | 确定性 | 用例中的 `checks` 全部通过（文件状态、运行命令验证、工具调用约束等） |
| `Task Completion [GEval]` | LLM 裁判 | 结合工具结果与 diff 判断任务是否真正完成 |
| `Response Quality [GEval]` | LLM 裁判 | 回复是否简洁、如实、使用中文 |
| `Tool Correctness` | 确定性 | 是否调用了 `expected_tools` |

结果写入 `evals/results/test_run_*.json`，包含 `hyperparameters`（版本、git commit、模型、system prompt 哈希、工具列表），`evals.compare` 据此做版本间对比；用例由通过变失败或指标下降 ≥0.1 视为回归，`eval.sh` 以退出码 1 结束，可直接接入 CI。

### 新增用例

在 `evals/datasets/*.json` 中追加：

```json
{
  "id": "唯一 id",
  "input": "给 Agent 的任务，可用 {workspace} 引用工作区路径",
  "files": {"相对路径": "初始文件内容"},
  "expected_tools": ["read_file"],
  "criteria": "给 LLM 裁判的具体判定标准",
  "checks": [
    {"type": "command", "cmd": "{python} -c \"from calc import add; assert add(1, 2) == 3\""}
  ],
  "approve_commands": false,
  "followup_answer": "当 Agent 提问时的自动回答",
  "max_steps": 20
}
```

`checks` 支持：`file_exists`、`file_not_exists`、`file_contains`、`file_not_contains`、`unchanged`、`output_contains_any`、`tool_called`、`tool_not_called`、`max_tool_calls`、`command`。
