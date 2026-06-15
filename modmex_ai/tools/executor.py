from __future__ import annotations

from modmex_ai.errors import UnknownToolError
from modmex_ai.models import ToolCall
from modmex_ai.tools.tool import Tool, ToolResult


class ToolExecutor:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(self, tool_call: ToolCall, *, context: object = None) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool: {tool_call.name}")
        state = getattr(context, "state", None)
        if state is not None:
            state["current_tool_call_id"] = tool_call.tool_call_id
        try:
            return ToolResult(tool_call_id=tool_call.tool_call_id, name=tool_call.name, output=tool.run(tool_call.arguments, context=context))
        finally:
            if state is not None:
                state.pop("current_tool_call_id", None)

    async def execute_async(
        self,
        tool_call: ToolCall,
        *,
        context: object = None,
    ) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool: {tool_call.name}")
        state = getattr(context, "state", None)
        if state is not None:
            state["current_tool_call_id"] = tool_call.tool_call_id
        try:
            return ToolResult(tool_call_id=tool_call.tool_call_id, name=tool_call.name, output=await tool.arun(tool_call.arguments, context=context))
        finally:
            if state is not None:
                state.pop("current_tool_call_id", None)
