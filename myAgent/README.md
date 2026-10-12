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

流程：每个用例在独立临时工作区运行 Agent；评测专用 `TracedAgent` 用 deepeval 的 `observe` 记录 Agent、每次 LLM 调用和工具 span。通过 `update_current_trace` 补充文件 diff 和验收要求，然后调用 `assert_test(golden=..., metrics=...)`，由 deepeval 自动构建测试用例并评分。正常 CLI 不依赖 tracing；`RunRecord` 仅保留用于运行统计和代码验收检查。

`evals/tracing.py` 是插桩适配层，不是新的自制 tracing 系统。命令审批和工具提问目前仍按固定配置回答，没有接入 `ConversationSimulator`。

| 指标 | 类型 | 说明 |
|---|---|---|
| `Checks` | 确定性 | 用例中的 `checks` 全部通过（文件状态、运行命令验证、工具调用约束等） |
| `Task Completion` | 原生轨迹指标 | `TaskCompletionMetric` 根据完整执行轨迹判断任务完成度 |
| `Step Efficiency`（可选） | 原生轨迹指标 | 设置 `MYAGENT_EVAL_STEP_EFFICIENCY=1` 分析步骤效率 |
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

### 自动生成候选题

```bash
.venv/bin/python -m evals.generate --count 5
.venv/bin/python -m evals.generate --count 5 --context my_agent/tools/fs.py
```

使用 deepeval 原生 `Synthesizer`，默认以 Agent 职责和全部工具 schema 为依据，支持额外指定代码或文档。`StylingConfig` 定义编码场景、题目格式和参考输出格式；`FiltrationConfig` 做模型质量过滤，`EvolutionConfig` 做题目演化。生成模型目前复用 `MYAGENT_JUDGE_*` 配置。

候选题写到 `evals/drafts/`，包括任务、初始文件、参考实现、检查、生成模型和来源哈希。结构校验不是代码正确性或安全认证。不会自动执行候选代码，也不会自动加入正式题库。请审核需求和测试，在容器/沙箱验证参考实现通过、错误实现不通过，确认不含危险命令，再把该题的 `reviewed` 设置为 `true`：

```bash
.venv/bin/python -m evals.generate --import-reviewed evals/drafts/具体文件.json
./eval.sh
```

导入只包含初始文件和验收标准，不把参考实现提供给 Agent。`_send_data=False` 禁止 Synthesizer 的数据集发送步骤，但指定上下文仍会发给配置的模型服务，请勿提供密钥或敏感代码。

### 查看轨迹与离线验证

安装可选查看器后，可用 `.venv/bin/deepeval inspect` 查看原生轨迹和评分原因（`uv pip install --python .venv/bin/python -e '../deepeval[inspect]'`）。

```bash
DEEPEVAL_RESULTS_FOLDER="$PWD/evals/.verification" .venv/bin/deepeval test run evals/test_native.py evals/test_generation.py
MYAGENT_EVAL_CHECKS_ONLY=1 ./eval.sh   # 仍运行真实 Agent，但不调用裁判模型
MYAGENT_EVAL_STEP_EFFICIENCY=1 ./eval.sh
```

原生指标替换了旧的完成度 GEval，首次运行作为新协议基线。新旧协议、裁判或指标开关不同，不做回归判定；不要把仅检查模式与全指标模式混合对比。临时目录不是安全沙箱，Agent 和检查命令仍在本机执行；执行自动生成代码前应人工审查或使用真正隔离的运行环境。
