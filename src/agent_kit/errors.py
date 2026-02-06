from __future__ import annotations


class AgentKitError(Exception):
    pass


class GraphBuildError(AgentKitError):
    pass


class InvocationValidationError(AgentKitError):
    pass


class SyncInAsyncContextError(AgentKitError):
    pass


class ModelToolBindingError(GraphBuildError):
    pass


class ToolSchemaConversionError(GraphBuildError):
    pass


class ToolValidationError(AgentKitError):
    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.details = dict(details or {})
