import json


def estimate_tokens(messages, tools=()):
    return len(json.dumps([messages, tools], ensure_ascii=False).encode("utf-8")) // 3 + 1


def split_history(messages, keep_turns):
    starts = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(starts) <= keep_turns:
        return [], messages[1:]
    cut = starts[-max(1, keep_turns)]
    return messages[1:cut], messages[cut:]


def validate_tool_pairs(messages):
    pending = set()
    for message in messages:
        if message["role"] == "tool":
            identifier = message["tool_call_id"]
            if identifier not in pending:
                raise ValueError("孤立工具结果")
            pending.remove(identifier)
        else:
            if pending:
                raise ValueError("工具结果缺失")
            pending.update(c["id"] for c in message.get("tool_calls", []))
    if pending:
        raise ValueError("工具结果缺失")
