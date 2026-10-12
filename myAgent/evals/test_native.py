"""离线验证原生 tracing/Golden 链路，不调用任何真实模型。"""
from types import SimpleNamespace

from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.tracing import update_current_trace
from openai.types.chat import ChatCompletion

from my_agent.config import Config

from .harness import EvalCase, run_case
from .metrics import ChecksMetric


def test_native_trace(monkeypatch, tmp_path):
    import evals.harness as harness

    responses = [
        ChatCompletion.model_validate({
            "id": "one", "object": "chat.completion", "created": 0, "model": "offline",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_one", "type": "function", "function": {
                        "name": "read_file", "arguments": '{"filePath":"demo.py"}'}}]}}],
        }),
        ChatCompletion.model_validate({
            "id": "two", "object": "chat.completion", "created": 0, "model": "offline",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "文件定义了常量 ANSWER。"}}],
        }),
    ]
    original = harness.TracedAgent

    class OfflineAgent(original):
        def __init__(self, cfg, ui):
            cfg.stream = False
            super().__init__(cfg, ui)
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: responses.pop(0))))

    monkeypatch.setattr(harness, "TracedAgent", OfflineAgent)
    case = EvalCase(id="offline_native", input="解释文件", files={"demo.py": "ANSWER = 42\n"},
                    checks=[{"type": "unchanged"}, {"type": "tool_called", "name": "read_file"}])
    result = run_case(case, Config(api_key="offline", base_url=None, model="offline", workspace=tmp_path))
    try:
        from deepeval.tracing.context import current_trace_context
        trace = current_trace_context.get()
        assert trace.tools_called[0].name == "read_file"
        assert trace.tools_called[0].input_parameters == {"filePath": "demo.py"}
        golden = Golden(input=case.input, name=case.id, additional_metadata={"offline": True})
        update_current_trace(expected_tools=trace.tools_called)
        assert_test(golden=golden, metrics=[ChecksMetric(result)])
        assert trace.context and "diff" in trace.context[-1]
    finally:
        result.cleanup()
