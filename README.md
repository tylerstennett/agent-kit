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
- Automatic tool schema binding for `BaseChatModel` inputs (`AgentConfig.model_tool_binding_mode="auto"`).
- Strict schema conversion by default (`AgentConfig.tool_schema_sync_policy="strict"`).
- Tool and agent lifecycle interception via hooks and middleware.
- Swappable graph builders with a default ReAct loop.
- Sync, async, and streaming invocation APIs.
- Nested agent composition through policy-driven `as_tool()` adapters.

## Tool Binding Behavior

- Default behavior auto-binds runtime tools to `BaseChatModel` instances.
- `model_tool_binding_mode` options:
  - `"auto"`: bind tools and fail fast if binding is unavailable.
  - `"manual"`: skip binding and assume the model is already configured.
  - `"off"`: disable binding checks entirely.
- Custom `ModelAdapter` implementations are treated as pre-bound and bypass auto-binding.

## Validation Repair Budget

- Runtime argument validation is always enforced before tool execution.
- Validation failures produce `ToolError(code="validation_error", details=...)`.
- The ReAct loop allows bounded repair turns for consecutive validation failures:
  - Default budget: `GraphBuildConfig.default_validation_repair_turns=1`
  - Per-invocation override: `InvocationConfig.validation_repair_turns`
  - If exceeded, termination reason is `validation_repair_exhausted`.
