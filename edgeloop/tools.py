"""Tool registry with JSON-schema argument validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List


class ToolError(Exception):
    """Raised for unknown tools or invalid arguments (fed back to the model)."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema for the arguments object
    fn: Callable[..., Any]

    def validate(self, args: Dict[str, Any]) -> None:
        import jsonschema
        try:
            jsonschema.validate(args, self.parameters)
        except jsonschema.ValidationError as exc:
            raise ToolError(f"invalid args for {self.name}: {exc.message}") from exc

    def call(self, args: Dict[str, Any]) -> str:
        self.validate(args)
        result = self.fn(**args)
        return result if isinstance(result, str) else json.dumps(result)


class ToolRegistry:
    def __init__(self, tools: List[Tool]):
        self._tools = {t.name: t for t in tools}
        if len(self._tools) != len(tools):
            raise ValueError("duplicate tool names")

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(
                f"unknown tool {name!r}; available: {sorted(self._tools)}")
        return self._tools[name]

    def render_for_prompt(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}\n  args schema: {json.dumps(t.parameters)}"
            for t in self._tools.values())
