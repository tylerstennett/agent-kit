# Agent Kit

Agent Kit is a typed Python toolkit for building stateful, extensible agents on top of LangGraph concepts.

## Install

```bash
uv pip install -e .
```

## Quickstart

```python
from agent_kit import Agent, tool
from tests.fakes import FakeModelAdapter, make_response

@tool(name="echo")
def echo(state, text: str) -> dict[str, str]:
    return {"echo": text}

model = FakeModelAdapter(
    responses=[
        make_response(
            "using tool",
            tool_calls=[{"id": "1", "name": "echo", "args": {"text": "hello"}}],
        ),
        make_response("done"),
    ]
)

agent = Agent(model=model, tools=[echo])
state = agent.run("say hello")
print(state["termination"])
```

## Highlights

- Stateful tools that receive full `AgentState`.
- Tool and agent lifecycle interception via hooks and middleware.
- Swappable graph builders with a default ReAct loop.
- Sync, async, and streaming invocation APIs.
- Nested agent composition through policy-driven `as_tool()` adapters.
