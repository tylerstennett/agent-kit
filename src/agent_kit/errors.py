from __future__ import annotations


class AgentKitError(Exception):
    pass


class GraphBuildError(AgentKitError):
    pass


class InvocationValidationError(AgentKitError):
    pass


class SyncInAsyncContextError(AgentKitError):
    pass


class ToolValidationError(AgentKitError):
    pass
