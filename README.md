# Agent Kit

Agent Kit is a typed Python toolkit for building stateful, extensible agents on top of LangGraph concepts.

## Install

Requires Python 3.11+.

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
- First-class `conduit` model support with automatic `Conduit`/`SyncConduit` adapter wrapping.
- Strict schema conversion by default (`AgentConfig.tool_schema_sync_policy="strict"`).
- Tool and agent lifecycle interception via hooks and middleware.
- Swappable graph builders with a default ReAct loop.
- Sync, async, and streaming invocation APIs.
- Nested agent composition through policy-driven `as_tool()` adapters.

## Tool Binding Behavior

- Default behavior auto-binds runtime tools to `BaseChatModel` instances.
- The same `model_tool_binding_mode` behavior applies to `conduit` models:
  - `"auto"`: convert runtime tools to conduit `ToolDefinition` payloads.
  - `"manual"` and `"off"`: do not send runtime tool schemas to conduit.
- `model_tool_binding_mode` options:
  - `"auto"`: bind tools and fail fast if binding is unavailable.
  - `"manual"`: skip binding and assume the model is already configured.
  - `"off"`: disable binding checks entirely.
- Custom `ModelAdapter` implementations are treated as pre-bound and bypass auto-binding.

## Conduit Usage

- `Agent(model=Conduit(...))` is async-only. Use `arun`, `ainvoke`, and `astream`.
- `Agent(model=SyncConduit(...))` is sync-only. Use `run`, `invoke`, and `stream`.
- Conduit request mapping behavior:
  - `InvocationConfig.configurable` maps to conduit `config_overrides`.
  - `InvocationConfig.thread_id` and `InvocationConfig.tags` map to conduit `RequestContext`.
  - Reserved configurable keys:
    - `conduit_context_metadata`: merged into `RequestContext.metadata`.
    - `conduit_runtime_overrides`: passed to conduit `runtime_overrides`.
- Adapter interfaces now live under `agent_kit.llm` (for example `agent_kit.llm.ModelAdapter`).

## Validation Repair Budget

- Runtime argument validation is always enforced before tool execution.
- Validation failures produce `ToolError(code="validation_error", details=...)`.
- The ReAct loop allows bounded repair turns for consecutive validation failures:
  - Default budget: `GraphBuildConfig.default_validation_repair_turns=1`
  - Per-invocation override: `InvocationConfig.validation_repair_turns`
  - If exceeded, termination reason is `validation_repair_exhausted`.
