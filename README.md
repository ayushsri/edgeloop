# edgeloop

**A local-first agent loop for small LLMs.** Run tool-calling agents on-device with Ollama, get reliable structured outputs from 1–8B models, and escalate to a cloud model only when the local one isn't confident.

Zero heavy dependencies — the Ollama client is stdlib-only.

## Why

Small models are cheap, private, and fast — but they fail at tool calling in ways GPT/Claude don't: malformed JSON, hallucinated tool names, wrong argument types. `edgeloop` treats that as an engineering problem:

- **Strict JSON tool calls** with schema validation and automatic *repair retries* (the model sees its own parse error and tries again)
- **Confidence-based escalation** — parse failures, retry counts, and self-reported confidence feed a router that decides local vs. cloud
- **Graceful offline degradation** — no cloud handler configured? The agent still answers with what the local model can do

```
            ┌──────────── edge device ────────────┐
 user ──▶ Agent loop ──▶ Ollama (SLM) ──▶ tool registry
            │   ▲             │
            │   └── repair retry on bad JSON
            ▼
      HybridRouter ── low confidence ──▶ cloud LLM (optional)
```

## Install

```bash
pip install -e .          # needs a running Ollama: https://ollama.com
ollama pull llama3.2:3b
```

## Quickstart

```python
from edgeloop import Agent, OllamaClient, Tool, HybridRouter

def get_inventory(sku: str) -> str:
    return f"SKU {sku}: 42 units in stock"

tools = [Tool(
    name="get_inventory",
    description="Look up stock level for a SKU",
    parameters={"type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"]},
    fn=get_inventory,
)]

agent = Agent(llm=OllamaClient(model="llama3.2:3b"), tools=tools)
result = agent.run("How many units of SKU-114 do we have?")
print(result.answer)         # grounded in the tool result
print(result.trace)          # every step: model output, tool calls, retries
```

Add cloud escalation:

```python
router = HybridRouter(
    local=OllamaClient(model="llama3.2:3b"),
    escalate=my_cloud_completion_fn,     # any Callable[[str], str]
    max_local_retries=2,
)
agent = Agent(llm=router, tools=tools)
```

## How reliability works

1. The system prompt forces a single-JSON-object protocol: `{"tool": ..., "args": ...}` or `{"answer": ..., "confidence": 0-1}`.
2. Output is parsed strictly; args are validated against the tool's JSON schema.
3. On failure, the error is fed back to the model (repair retry) — this fixes a large share of small-model mistakes.
4. Repeated failures or low self-reported confidence trigger the router's escalation path (if configured).

Pairs well with [agentgauge](https://github.com/ayushsri/agentgauge) for measuring tool-call reliability across models and quantizations, and with [mcp-tabular](https://github.com/ayushsri/mcp-tabular) as a local data tool.

## Roadmap

- Grammar-constrained decoding (GBNF) backend
- On-device RAG helper (SQLite-vec)
- Telemetry sampling for fleet eval

## License

MIT
