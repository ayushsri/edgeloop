"""LLM clients. OllamaClient is stdlib-only (urllib) to stay edge-friendly."""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class OllamaClient:
    """Minimal client for a local Ollama server (default http://localhost:11434).

    Uses /api/generate with streaming disabled and JSON-format hinting, which
    materially improves structured-output reliability on small models.
    """

    def __init__(self, model: str = "llama3.2:3b",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.0,
                 json_mode: bool = True,
                 timeout: float = 120.0):
        self.model, self.host = model, host.rstrip("/")
        self.temperature, self.json_mode, self.timeout = temperature, json_mode, timeout

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.json_mode:
            payload["format"] = "json"
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["response"]
