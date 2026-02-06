from __future__ import annotations

from collections.abc import Iterable

from .base import BaseTool


class ToolRegistry:
    def __init__(self, tools: Iterable[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        if tools:
            self.set_tools(tools)

    def add_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def set_tools(self, tools: Iterable[BaseTool]) -> None:
        self._tools = {}
        for tool in tools:
            self.add_tool(tool)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def values(self) -> list[BaseTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools)
