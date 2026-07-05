"""Fully local tool-calling agent. Requires: `ollama pull llama3.2:3b`."""

from edgeloop import Agent, OllamaClient, Tool

STOCK = {"SKU-114": 42, "SKU-233": 7, "SKU-987": 0}


def get_inventory(sku: str) -> str:
    if sku not in STOCK:
        return f"unknown SKU {sku}; known: {sorted(STOCK)}"
    return f"{sku}: {STOCK[sku]} units in stock"


tools = [Tool(
    name="get_inventory",
    description="Look up current stock level for a SKU (e.g. 'SKU-114')",
    parameters={"type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"]},
    fn=get_inventory,
)]

if __name__ == "__main__":
    agent = Agent(llm=OllamaClient(model="llama3.2:3b"), tools=tools)
    result = agent.run("Is SKU-987 in stock? How many units?")
    print("answer:    ", result.answer)
    print("confidence:", result.confidence)
    print("repairs:   ", result.repairs)
    for step in result.trace:
        print(" ", step)
