from openai.types.chat import ChatCompletion


def consume_stream(stream, on_text):
    text = []
    calls = {}
    usage = None
    finished = False
    try:
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage.model_dump()
            for choice in chunk.choices:
                if choice.index != 0:
                    continue
                if choice.finish_reason is not None:
                    finished = True
                delta = choice.delta
                if delta.content:
                    text.append(delta.content)
                    on_text(delta.content)
                for part in delta.tool_calls or []:
                    call = calls.setdefault(part.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if part.id:
                        call["id"] += part.id
                    if part.function:
                        call["function"]["name"] += part.function.name or ""
                        call["function"]["arguments"] += part.function.arguments or ""
        if not finished:
            raise ValueError("模型流提前结束，没有 finish_reason；未执行任何未完成工具调用")
        message = {"role": "assistant", "content": "".join(text)}
        if calls:
            message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return ChatCompletion.model_validate({
            "id": "stream", "object": "chat.completion", "created": 0, "model": "stream",
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": usage,
        })
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()
