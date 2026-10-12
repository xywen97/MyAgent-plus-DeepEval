# MyAgent + DeepEval

一个运行在终端里的 AI 编码助手（MyAgent），以及基于 [deepeval](https://github.com/confident-ai/deepeval) 的自动化评测体系，用于在每次修改 Agent 后量化效果、发现回归。

## 目录

- [特性](#特性)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装](#安装)
- [配置](#配置)
- [使用](#使用)
- [评测](#评测)
- [更新 deepeval 子模块](#更新-deepeval-子模块)
- [许可证](#许可证)

## 特性

- **终端编码助手**：支持文件读写、代码搜索、命令执行、待办管理、追问交互等工具。
- **协议兼容**：基于 OpenAI Chat Completions 协议，兼容 OpenAI / DeepSeek / 通义 / Kimi 等服务。
- **安全约束**：写入/删除仅限工作区内，高风险命令需要终端确认。
- **可扩展**：通过装饰器注册新工具。
- **自动化评测**：每个用例在隔离的临时工作区中无头运行，结合确定性检查与 LLM 裁判打分，并自动与上一次结果对比，可接入 CI。

## 项目结构

```
.
├── deepeval/                 # git submodule，指向 confident-ai/deepeval
└── myAgent/
    ├── my_agent/             # Agent 源码（CLI、主循环、Prompt、工具）
    ├── evals/                # 评测代码与数据集
    │   ├── datasets/         # 评测用例（JSON）
    │   └── results/          # 评测结果（已被 git 忽略）
    ├── eval.sh               # 一键评测 + 回归对比
    ├── requirements.txt      # Agent 运行依赖
    ├── requirements-eval.txt # 评测依赖（以可编辑模式安装本地 deepeval）
    └── .env.example          # 配置模板
```

## 环境要求

- Python 3.12
- Git
- [uv](https://github.com/astral-sh/uv)（推荐，也可用 pip）
- [ripgrep](https://github.com/BurntSushi/ripgrep)（可选，用于加速代码搜索）

## 安装

```bash
# 克隆仓库（必须带上子模块）
git clone --recursive https://github.com/xywen97/MyAgent-plus-DeepEval.git
cd MyAgent-plus-DeepEval

# 如果克隆时没有加 --recursive
git submodule update --init

# 创建虚拟环境并安装依赖
cd myAgent
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 如需运行评测，额外安装
uv pip install -r requirements-eval.txt
```

## 配置

```bash
cp .env.example .env
```

| 变量 | 必填 | 说明 |
|---|---|---|
| `MYAGENT_API_KEY` | 是 | 模型服务的 API Key |
| `MYAGENT_BASE_URL` | 否 | 服务地址，留空使用 OpenAI 官方地址 |
| `MYAGENT_MODEL` | 是 | 模型名称，如 `gpt-4o` |
| `MYAGENT_MAX_STEPS` | 否 | 单次任务最大步数 |
| `MYAGENT_COMMAND_TIMEOUT` | 否 | 命令执行超时（秒） |
| `MYAGENT_AUTO_APPROVE` | 否 | 是否跳过命令确认，谨慎使用 |
| `MYAGENT_JUDGE_MODEL` / `MYAGENT_JUDGE_API_KEY` / `MYAGENT_JUDGE_BASE_URL` | 否 | 评测裁判模型，未设置时复用 Agent 模型配置 |

完整说明见 [`myAgent/.env.example`](myAgent/.env.example)。

## 使用

```bash
cd myAgent

# 交互模式
python -m my_agent -w /path/to/project

# 单次任务
python -m my_agent -w /path/to/project "解释一下这个项目的结构"
```

交互命令：`/help`、`/reset`、`/todos`、`/exit`。

## 评测

```bash
cd myAgent

./eval.sh                          # 全量评测 + 与上次结果对比
./eval.sh -k fix_bug               # 只跑部分用例（pytest -k）
./eval.sh --label "prompt-v2"      # 给本次运行打标签
.venv/bin/python -m evals.compare A.json B.json   # 对比任意两次结果
```

评测指标：

| 指标 | 类型 | 说明 |
|---|---|---|
| `Checks` | 确定性 | 用例中定义的文件状态、命令验证、工具调用约束全部通过 |
| `Task Completion [GEval]` | LLM 裁判 | 结合工具结果与 diff 判断任务是否真正完成 |
| `Response Quality [GEval]` | LLM 裁判 | 回复是否简洁、如实、使用中文 |
| `Tool Correctness` | 确定性 | 是否调用了预期工具 |

用例由通过变为失败，或指标下降 ≥ 0.1，视为回归，`eval.sh` 以退出码 1 结束。

> 评测结果目录 `evals/results/` 不纳入版本控制，新克隆的仓库首次运行时没有对比基线。

新增用例格式及更多细节见 [`myAgent/README.md`](myAgent/README.md)。

## 更新 deepeval 子模块

```bash
# 更新到上游最新版本
cd deepeval
git fetch origin && git checkout main && git pull
cd ..
git add deepeval
git commit -m "chore: bump deepeval"
git push
```

如需修改 deepeval 源码，请先 fork 上游仓库，并将子模块地址改为你的 fork：

```bash
git config -f .gitmodules submodule.deepeval.url https://github.com/<your-name>/deepeval.git
git submodule sync
```

## 许可证

deepeval 遵循其上游仓库的 [Apache-2.0](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) 许可证。
