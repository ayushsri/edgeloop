"""The agent loop: strict-JSON protocol, repair retries, bounded steps."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LLM
from .tools import Tool, ToolError, ToolRegistry

SYSTEM = """You are a precise on-device assistant. You MUST reply with exactly \
one JSON object and nothing else.

To use a tool:
  {{"tool": "<tool_name>", "args": {{...}}}}
To answer the user:
  {{"answer": "<final answer>", "confidence": <0.0-1.0>}}

Available tools:
{tools}

Rules: use a tool when it can ground your answer in real data; report honest \
confidence; never invent tool names or arguments."""


@dataclass
class Step:
    kind: str          # "model" | "tool" | "repair" | "error"
    content: str

    def __repr__(self) -> str:
        return f"[{self.kind}] {self.content[:200]}"


@dataclass
class AgentResult:
    answer: str
    confidence: float
    trace: List[Step] = field(default_factory=list)
    escalated: bool = False
    repairs: int = 0


def parse_reply(text: str) -> dict:
    """Strictly parse the model's single-JSON-object reply."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    obj = json.loads(text)  # raises on malformed JSON
    if not isinstance(obj, dict):
        raise ValueError("reply must be a JSON object")
    if "tool" in obj:
        if not isinstance(obj.get("args"), dict):
            raise ValueError('tool call must include an "args" object')
        return obj
    if "answer" in obj:
        obj.setdefault("confidence", 0.5)
        return obj
    raise ValueError('reply must contain "tool" or "answer"')


class Agent:
    """Bounded tool-use loop with repair retries.

    On malformed output or bad tool arguments, the error is appended to the
    conversation and the model retries — small models recover from a large
    share of their structured-output mistakes when shown the exact error.
    """

    def __init__(self, llm: LLM, tools: List[Tool],
                 max_steps: int = 6, max_repairs: int = 3):
        self.llm = llm
        self.registry = ToolRegistry(tools)
        self.max_steps, self.max_repairs = max_steps, max_repairs

    def run(self, query: str) -> AgentResult:
        trace: List[Step] = []
        repairs = 0
        convo = (SYSTEM.format(tools=self.registry.render_for_prompt())
                 + f"\n\nUser: {query}\nReply with one JSON object:")

        for _ in range(self.max_steps + self.max_repairs):
            raw = self.llm.complete(convo)
            trace.append(Step("model", raw))
            try:
                reply = parse_reply(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                repairs += 1
                trace.append(Step("repair", str(exc)))
                if repairs > self.max_repairs:
                    return AgentResult(
                        answer="I could not produce a reliable answer.",
                        confidence=0.0, trace=trace, repairs=repairs)
                convo += (f"\n\nYour last reply was invalid ({exc}). "
                          "Reply again with exactly one valid JSON object:")
                continue

            if "answer" in reply:
                return AgentResult(
                    answer=str(reply["answer"]),
                    confidence=float(reply.get("confidence", 0.5)),
                    trace=trace, repairs=repairs)

            # tool call
            try:
                tool = self.registry.get(reply["tool"])
                observation = tool.call(reply["args"])
                trace.append(Step("tool", f"{reply['tool']} -> {observation}"))
                convo += (f"\n\nTool {reply['tool']} returned:\n{observation}\n"
                          "Now reply with one JSON object (tool call or final answer):")
            except ToolError as exc:
                repairs += 1
                trace.append(Step("repair", str(exc)))
                if repairs > self.max_repairs:
                    return AgentResult(
                        answer="I could not complete the tool call reliably.",
                        confidence=0.0, trace=trace, repairs=repairs)
                convo += (f"\n\nTool call failed: {exc}. "
                          "Reply again with one valid JSON object:")

        return AgentResult(answer="Step limit reached without a final answer.",
                           confidence=0.0, trace=trace, repairs=repairs)
