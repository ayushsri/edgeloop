"""edgeloop: local-first agent loop for small LLMs."""

from .llm import OllamaClient
from .tools import Tool, ToolError
from .agent import Agent, AgentResult
from .router import HybridRouter

__version__ = "0.1.0"
__all__ = ["Agent", "AgentResult", "OllamaClient", "Tool", "ToolError",
           "HybridRouter"]
