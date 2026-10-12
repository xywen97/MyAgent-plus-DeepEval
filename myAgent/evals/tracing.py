from __future__ import annotations

import json

from deepeval.tracing import observe, update_current_span
from deepeval.tracing.context import current_span_context

from my_agent.agent import Agent


class TracedCompletions:
    def __init__(self, completions):
        self.completions = completions

    @observe(type="llm", name="chat_completion")
    def create(self, **kwargs):
        update_current_span(input=kwargs["messages"], metadata={"model": kwargs["model"]})
        response = self.completions.create(**kwargs)
        message = response.choices[0].message
        update_current_span(output=message.model_dump(exclude_none=True))
        return response


class TracedAgent(Agent):
    """Evaluation-only instrumentation; normal CLI has no deepeval dependency."""

    @observe(type="agent", name="MyAgent")
    def run(self, user_input: str) -> str:
        self.eval_span = current_span_context.get()
        update_current_span(input=user_input)
        output = super().run(user_input)
        update_current_span(output=output or f"(无最终回复，stop_reason={self.last_run.stop_reason})")
        return output

    @observe(type="llm", name="chat_completion")
    def _request(self):
        update_current_span(input=self.messages, metadata={"model": self.cfg.model, "stream": self.cfg.stream})
        response = super()._request()
        update_current_span(output=response.choices[0].message.model_dump(exclude_none=True))
        return response

    def _make_subagent(self, cfg, ui):
        return TracedAgent(cfg, ui)

    @observe(type="llm", name="context_summary")
    def _summarize(self, history):
        update_current_span(input=history)
        summary = super()._summarize(history)
        update_current_span(output=summary)
        return summary

    @observe(type="tool")
    def _run_tool(self, name: str, raw_args: str | None) -> str:
        try:
            args = json.loads(raw_args or "{}")
        except (ValueError, TypeError):
            args = {"invalid_arguments": raw_args}
        update_current_span(name=name, input=args)
        output = super()._run_tool(name, raw_args)
        update_current_span(output=output)
        return output
