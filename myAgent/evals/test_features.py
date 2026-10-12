from contextlib import nullcontext
from types import SimpleNamespace

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.tracing import observe, update_current_trace
from openai.types.chat import ChatCompletionChunk

from my_agent.streaming import consume_stream


class FeatureMetric(BaseMetric):
    threshold = 1.0
    async_mode = False

    def __init__(self, passed):
        self.passed = passed

    def measure(self, test_case, **kwargs):
        self.score = float(self.passed)
        self.success = self.passed
        self.reason = "功能断言通过" if self.passed else "功能断言失败"
        return self.score

    async def a_measure(self, test_case, **kwargs):
        return self.measure(test_case)

    @property
    def __name__(self):
        return "Feature Correctness"


@observe(type="agent")
def evaluate_feature(name, passed):
    update_current_trace(input=name, output="通过" if passed else "失败")
    return passed


def assess(name, passed):
    evaluate_feature(name, passed)
    assert_test(golden=Golden(input=name, name=name), metrics=[FeatureMetric(passed)])


def chunk(delta, finish=None):
    return ChatCompletionChunk.model_validate({"id": "c", "object": "chat.completion.chunk", "created": 0,
        "model": "mock", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


def test_streaming():
    emitted = []
    def source():
        yield chunk({"content": "你"})
        assert emitted == ["你"]
        yield chunk({"content": "好", "tool_calls": [{"index": 0, "id": "call", "type": "function",
            "function": {"name": "read_", "arguments": '{"file'}}]})
        yield chunk({"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": 'Path":"a.py"}'}}]}, "tool_calls")
    response = consume_stream(source(), emitted.append)
    call = response.choices[0].message.tool_calls[0]
    assess("streaming", response.choices[0].message.content == "你好" and call.function.name == "read_file"
           and call.function.arguments == '{"filePath":"a.py"}')


def test_compression(tmp_path):
    from my_agent.agent import Agent
    from my_agent.config import Config
    from my_agent.context import validate_tool_pairs
    from .harness import HeadlessUI
    agent = Agent(Config(api_key="mock", base_url=None, model="mock", workspace=tmp_path,
                         context_budget=6000, context_keep_turns=1), HeadlessUI(False, None))
    agent.messages = [{"role": "system", "content": "rules"},
        {"role": "user", "content": "保留编号 ABC"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "function": {"name": "read_file", "arguments": "{}"}, "type": "function"}]},
        {"role": "tool", "tool_call_id": "a", "content": "x" * 30000},
        {"role": "assistant", "content": "已读"}, {"role": "user", "content": "最新任务"}]
    agent._summarize = lambda history: "编号 ABC，已读文件"
    agent._compress_context()
    validate_tool_pairs(agent.messages)
    assess("compression", len(agent.messages) == 3 and "ABC" in agent.messages[1]["content"]
           and agent.messages[-1]["content"] == "最新任务")


def test_subagent(tmp_path):
    from my_agent.agent import Agent
    from my_agent.config import Config
    from my_agent.subagent import READ_TOOLS
    from .harness import HeadlessUI
    child = Agent(Config(api_key="mock", base_url=None, model="mock", workspace=tmp_path), HeadlessUI(False, None))
    child.allowed_tools = READ_TOOLS
    child.read_only = True
    child.tool_budget = 3
    (tmp_path / "a.py").write_text("ANSWER=42")
    write = child._execute_tool("write_to_file", {"filePath": "a.py", "content": "wrong"})
    escape = child._execute_tool("read_file", {"filePath": str(tmp_path.parent / "escape")})
    read = child._execute_tool("read_file", {"filePath": "a.py"})
    recursive = child._execute_tool("delegate_task", {"task": "again"})
    child.tool_budget = 0
    exhausted = child._execute_tool("read_file", {"filePath": "a.py"})
    assess("subagent", all(x.startswith("Error") for x in (write, escape, recursive, exhausted))
           and "42" in read and (tmp_path / "a.py").read_text() == "ANSWER=42")


def test_delegate_roundtrip(tmp_path):
    from openai.types.chat import ChatCompletion
    from my_agent.agent import Agent, RunRecord
    from my_agent.config import Config
    from .harness import HeadlessUI
    import json
    parent = Agent(Config(api_key="mock", base_url=None, model="mock", workspace=tmp_path,
                          subagent_max_calls=1), HeadlessUI(False, None))
    parent.last_run = RunRecord(input="research")
    response = ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 0,
        "model": "mock", "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": "结论：a.py 包含常量"}}]})
    parent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response)))
    output = parent._execute_tool("delegate_task", {"task": "研究 a.py"})
    denied = parent._execute_tool("delegate_task", {"task": "再次研究"})
    assess("delegate_roundtrip", json.loads(output)["stop_reason"] == "completed"
           and denied.startswith("Error") and len(parent.messages) == 1)

