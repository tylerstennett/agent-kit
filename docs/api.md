# API Reference

## Core

- `agent_kit.Agent`: top-level orchestration API.
- `agent_kit.AgentConfig`: agent configuration.
- `agent_kit.InvocationRequest`: flexible invocation input object.
- `agent_kit.InvocationConfig`: per-invocation behavior settings.

## Tools

- `agent_kit.BaseTool`: class-based stateful tool base class.
- `agent_kit.tool`: decorator for function tools.
- `agent_kit.ToolResult`: normalized tool execution result.
- `agent_kit.ToolDecision`: pre-execute control signal (`continue` or `skip`).

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
