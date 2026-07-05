import pytest

from edgeloop import Agent, HybridRouter, Tool, ToolError
from edgeloop.agent import parse_reply


def make_tool():
    return Tool(
        name="get_inventory",
        description="stock lookup",
        parameters={"type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"]},
        fn=lambda sku: f"{sku}: 42 units",
    )


class ScriptedLLM:
    """Replays canned replies — lets us test the loop without a model."""

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, prompt: str) -> str:
        return self.replies.pop(0)


def test_parse_reply_strictness():
    assert parse_reply('{"answer": "hi", "confidence": 0.9}')["answer"] == "hi"
    assert parse_reply('```json\n{"tool": "t", "args": {}}\n```')["tool"] == "t"
    with pytest.raises(ValueError):
        parse_reply('{"neither": true}')
    with pytest.raises(Exception):
        parse_reply("not json")


def test_happy_path_tool_then_answer():
    llm = ScriptedLLM([
        '{"tool": "get_inventory", "args": {"sku": "SKU-114"}}',
        '{"answer": "42 units of SKU-114.", "confidence": 0.95}',
    ])
    result = Agent(llm, [make_tool()]).run("stock for SKU-114?")
    assert "42" in result.answer and result.repairs == 0
    assert any(s.kind == "tool" for s in result.trace)


def test_repair_retry_recovers_from_bad_json():
    llm = ScriptedLLM([
        "sorry, here is the tool call you wanted",              # invalid
        '{"tool": "get_inventory", "args": {"sku": "SKU-114"}}',
        '{"answer": "42 units.", "confidence": 0.9}',
    ])
    result = Agent(llm, [make_tool()]).run("stock?")
    assert result.repairs == 1 and "42" in result.answer


def test_bad_tool_args_are_fed_back():
    llm = ScriptedLLM([
        '{"tool": "get_inventory", "args": {"item": "SKU-114"}}',  # wrong key
        '{"tool": "get_inventory", "args": {"sku": "SKU-114"}}',
        '{"answer": "42 units.", "confidence": 0.9}',
    ])
    result = Agent(llm, [make_tool()]).run("stock?")
    assert result.repairs == 1 and "42" in result.answer


def test_repair_budget_exhaustion():
    llm = ScriptedLLM(["garbage"] * 10)
    result = Agent(llm, [make_tool()], max_repairs=2).run("stock?")
    assert result.confidence == 0.0


class DeadLLM:
    def complete(self, prompt):
        raise ConnectionError("ollama down")


def test_router_escalates_when_local_dead():
    router = HybridRouter(local=DeadLLM(), escalate=lambda p: "cloud says hi")
    assert router.complete("x") == "cloud says hi"
    assert router.last_used == "cloud"


def test_router_local_only_degradation():
    router = HybridRouter(local=DeadLLM(), escalate=None)
    with pytest.raises(RuntimeError):
        router.complete("x")
    assert router.should_escalate(confidence=0.1, repairs=5) is False


def test_router_should_escalate_thresholds():
    router = HybridRouter(local=DeadLLM(), escalate=lambda p: "c",
                          confidence_floor=0.4, max_local_retries=2)
    assert router.should_escalate(0.2, 0) is True
    assert router.should_escalate(0.9, 3) is True
    assert router.should_escalate(0.9, 0) is False
