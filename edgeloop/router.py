"""HybridRouter: run local first, escalate to cloud on weak signals."""

from __future__ import annotations

from typing import Callable, Optional

from .llm import LLM


class HybridRouter:
    """An LLM wrapper that answers locally and escalates when the local model
    struggles.

    Escalation signals (checked by the caller via `should_escalate`, or used
    transparently through `complete_with_fallback`):
      - the local call raised (Ollama down, timeout)
      - repeated repair retries upstream (tracked by the Agent)

    `escalate` is any Callable[[str], str] — an API client for a cloud model.
    If it is None, the router degrades gracefully to local-only.
    """

    def __init__(self, local: LLM,
                 escalate: Optional[Callable[[str], str]] = None,
                 confidence_floor: float = 0.4,
                 max_local_retries: int = 2):
        self.local, self.escalate = local, escalate
        self.confidence_floor = confidence_floor
        self.max_local_retries = max_local_retries
        self.last_used = "local"

    def complete(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for _ in range(self.max_local_retries):
            try:
                self.last_used = "local"
                return self.local.complete(prompt)
            except Exception as exc:  # connection refused, timeout, ...
                last_err = exc
        if self.escalate is not None:
            self.last_used = "cloud"
            return self.escalate(prompt)
        raise RuntimeError(
            f"local model unavailable and no escalation configured: {last_err}")

    def should_escalate(self, confidence: float, repairs: int) -> bool:
        """Post-hoc check for Agent results: answer weak enough to redo in cloud?"""
        if self.escalate is None:
            return False
        return confidence < self.confidence_floor or repairs > self.max_local_retries
