# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Kit is a typed Python toolkit (Python 3.11+) for building stateful, extensible agents. It implements a ReAct (Reason + Act) loop on top of LangGraph concepts, with first-class support for LangChain `BaseChatModel` and `conduit` (`Conduit`/`SyncConduit`) model backends.

## Commands

All commands use `uv run` to execute inside the project virtualenv without manual activation.

```bash
# Install (editable, with dev dependencies)
uv pip install -e ".[dev]"

# Run all unit tests
uv run pytest

# Run a single test file
uv run pytest tests/test_agent_invocation.py

# Run a single test by name
uv run pytest tests/test_agent_invocation.py -k "test_name"

# Run integration tests (requires OPENROUTER_API_KEY in .env)
uv run pytest tests/integration/ -v

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

## Architecture

### Core Loop

`Agent` is the top-level entry point. It accepts a model, tools, and optional config. On each invocation:

1. `Agent._resolve_model_adapter()` wraps the raw model into a `ModelAdapter` (either `LangChainModelAdapter` or `ConduitModelAdapter`). Custom `ModelAdapter` implementations pass through directly.
2. `Agent._compiled_graph()` lazily builds a `CompiledGraph` via a `GraphBuilder` (default: `ReActGraphBuilder`). The graph is rebuilt only when the agent is marked dirty (tools/hooks/config changed).
3. The compiled graph runs a **ReAct loop** (`graph/react.py`): alternating `reasoning_step` (LLM call) → `action_step` (tool execution) until no tool calls remain, `max_steps` is hit, or validation repair budget is exhausted.

### Model Adapter Layer (`llm/`)

`ModelAdapter` is a `Protocol` with `complete` and `acomplete` methods returning `ModelResponse`. Two concrete adapters exist:
- `LangChainModelAdapter` — wraps `BaseChatModel`, handles `bind_tools` for auto-binding.
- `ConduitModelAdapter` — wraps `Conduit`/`SyncConduit`, maps `InvocationConfig` fields to conduit request context and config overrides.

Tool binding mode (`AgentConfig.model_tool_binding_mode`): `"auto"` (default) binds tool schemas to the model at graph compile time. `"manual"` and `"off"` skip binding.

### Tool System (`tools/`)

- `BaseTool` — abstract base with lifecycle hooks: `pre_execute` → `execute` → `post_execute` (each has async variants). Tools receive full `AgentState` plus kwargs.
- `FunctionTool` / `@tool` decorator — wraps plain functions. First parameter must be `state: AgentState`.
- `ToolExecutor` — orchestrates tool calls in sequential or parallel mode, running before/after hooks and middleware around each call. Runtime argument validation is always enforced (`validate_args`).
- `ToolRegistry` — name-keyed lookup for registered tools.

### State (`state.py`)

`AgentState` is a `TypedDict` with annotated reducer fields: `messages`, `tool_outputs`, `metadata`, `routing_hints`, `termination`. Reducers (`append_messages`, `merge_dict`, etc.) define how values merge across steps.

### Middleware (`middleware.py`)

Middleware wraps tool execution as `(ToolInvocationContext, NextCallable) -> ToolExecutionOutcome`. Built-in: `logging_middleware`, `RetryMiddleware`, `TimeoutMiddleware`, `MetricsMiddleware`. Middleware chains execute in registration order (outermost first).

### Nested Agents (`nested.py`)

`Agent.as_tool()` / `agent_as_tool()` adapts a child agent into a `NestedAgentTool`. The `NestedAgentPolicy` controls state bridging (`copy_merge` / `shared` / `isolated`), message merging, failure mode, and max depth.

## Testing Conventions

- Tests use `FakeModelAdapter` and `make_response()` from `tests/fakes.py` to simulate LLM responses without real API calls.
- `conftest.py` adds `src/` to `sys.path` so imports resolve as `agent_kit.*`.
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio`).
- Fake tools (`EchoTool`, `MetadataTool`, `SlowTool`) are in `tests/fakes.py`.
- Integration tests live in `tests/integration/` and are marked `@pytest.mark.integration`. They require `OPENROUTER_API_KEY` in `.env` and are auto-skipped when the key is absent.

## Conduit Dependency

Conduit (`conduit @ git+https://github.com/tylerstennett/conduit.git@main`) is maintained by the same author as Agent Kit. When working on Agent Kit, if you identify improvements, bugs, or missing features in Conduit itself (e.g., message format issues, missing API surface, tool schema gaps, streaming behavior), **call these out explicitly in your output**. We have direct commit access to the Conduit repository and can make changes there in tandem with Agent Kit work.

## Commit Format

`<prefix>: <msg>` — prefix is one of `feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`. Message is a concise single sentence.

## Style

- Ruff: line length 100, target Python 3.11, lint rules `E, F, I, UP, B, SIM`.
- mypy: strict mode.
- All dataclasses use `slots=True`.
- All imports are absolute (`from agent_kit.x import Y`), never relative.
- Sync and async paths are always implemented side by side (e.g., `invoke`/`ainvoke`, `execute`/`aexecute`).
