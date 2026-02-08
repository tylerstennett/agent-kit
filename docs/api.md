# API Reference

## Core

- `agent_kit.Agent`: top-level orchestration API.
- `agent_kit.AgentConfig`: agent configuration.
- `agent_kit.ToolBindingMode`: `"auto" | "manual" | "off"`.
- `agent_kit.ToolSchemaSyncPolicy`: `"strict" | "warn" | "ignore"`.
- `agent_kit.InvocationRequest`: flexible invocation input object.
- `agent_kit.InvocationConfig`: per-invocation behavior settings.

`Agent.model` accepts:

- LangChain `BaseChatModel`
- `llm-conduit` `Conduit` (async-only)
- `llm-conduit` `SyncConduit` (sync-only)
- Any custom `agent_kit.llm.ModelAdapter`

## LLM Adapters

- `agent_kit.llm.ModelAdapter`: protocol for sync/async completion.
- `agent_kit.llm.ModelResponse`: normalized model response payload.
- `agent_kit.llm.ToolCall`: normalized tool call payload (`id`, `name`, `args`).
- `agent_kit.llm.LangChainModelAdapter`: adapter for LangChain runnable chat models.
- `agent_kit.llm.ConduitModelAdapter`: adapter for `llm-conduit` models.
- Conduit-specific configurable keys:
  - `agent_kit.llm.CONDUIT_CONTEXT_METADATA_KEY` (`"conduit_context_metadata"`)
  - `agent_kit.llm.CONDUIT_RUNTIME_OVERRIDES_KEY` (`"conduit_runtime_overrides"`)
- `agent_kit.model_adapter` is no longer the canonical adapter module path.

For conduit model invocations:

- `InvocationConfig.configurable` maps to conduit `config_overrides`.
- `InvocationConfig.thread_id` and `InvocationConfig.tags` map to conduit `RequestContext`.
- `InvocationConfig.configurable[CONDUIT_CONTEXT_METADATA_KEY]` maps to `RequestContext.metadata`.
- `InvocationConfig.configurable[CONDUIT_RUNTIME_OVERRIDES_KEY]` maps to conduit `runtime_overrides`.

## Tools

- `agent_kit.BaseTool`: class-based stateful tool base class.
- `agent_kit.tool`: decorator for function tools.
- `agent_kit.ToolResult`: normalized tool execution result.
- `agent_kit.ToolDecision`: pre-execute control signal (`continue` or `skip`).

Tool validation failures use `ToolError(code="validation_error")` and include structured
`details` payloads with:

- `kind`: `unexpected_args` | `missing_args` | `invalid_type`
- `unexpected`: list of unknown argument names
- `missing`: list of missing required argument names
- `arg_name`, `expected`, `actual_type` for type mismatches

## Middleware

- `agent_kit.logging_middleware(...)`
- `agent_kit.RetryMiddleware(...)`
- `agent_kit.TimeoutMiddleware(...)`
- `agent_kit.MetricsMiddleware(...)`

## Nested Agents

- `agent_kit.NestedAgentPolicy`: policy object for bridge, message merge, and depth limits.
- `Agent.as_tool(...)`: adapt an agent into a tool for parent agents.

## Events

Streaming emits typed events:

- `RunStartEvent`
- `LLMTokenEvent`
- `ToolStartEvent`
- `ToolEndEvent`
- `StateUpdateEvent`
- `RunEndEvent`
- `ErrorEvent`

## Errors

- `agent_kit.ModelToolBindingError`: raised when auto-binding fails or is unsupported.
- `agent_kit.ToolSchemaConversionError`: raised for unsupported strict schema conversion.
- `agent_kit.ToolValidationError`: validation exception with optional `details`.
