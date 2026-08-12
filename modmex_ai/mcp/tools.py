"""Remote MCP tools adapted to the existing modmex-ai Tool contract."""

from __future__ import annotations

from typing import Any, Callable

from modmex_ai.tools.tool import Tool


class RemoteTool(Tool):
    def __init__(self, client: Any, definition: dict[str, Any]) -> None:
        self.client = client
        self.definition = definition
        async def invoke(**arguments: Any) -> Any:
            return await client.call_tool(definition["name"], arguments)
        super().__init__(invoke, name=definition["name"], description=definition.get("description", ""))

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.definition.get("inputSchema", {"type": "object", "properties": {}}),
        }

    def _coerce_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Preserve the remote MCP schema instead of inspecting ``**arguments``."""
        return dict(arguments)
