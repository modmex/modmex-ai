from __future__ import annotations

from modmex_ai.errors import UnknownToolError, ToolExecutionError, ToolValidationError
from modmex_ai.budget import bounded_await, remaining_seconds
from time import perf_counter
from modmex_ai.models import ToolCall
from modmex_ai.tools.tool import Tool, ToolResult, _current_tool_call_id


class ToolExecutor:
    def __init__(self, tools: list[Tool], *, error_policy: str = "raise") -> None:
        if error_policy not in {"raise", "return_error"}:
            raise ValueError("Invalid tool error policy")
        self.error_policy = error_policy
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def execute(self, tool_call: ToolCall, *, context: object = None) -> ToolResult:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            raise UnknownToolError(f"Unknown tool: {tool_call.name}")
        remaining_seconds()
        state = getattr(context, "state", None)
        if state is not None:
            state["current_tool_call_id"] = tool_call.tool_call_id
        started = perf_counter()
        outcome = "success"
        token = _current_tool_call_id.set(tool_call.tool_call_id)
        try:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                output=tool.run(tool_call.arguments, context=context),
            )
        except (ToolExecutionError, ToolValidationError) as error:
            outcome = type(error).__name__
            return self.failure(tool_call, error)
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            _current_tool_call_id.reset(token)
            self._trace(context, tool_call, started, outcome)
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
        remaining_seconds()
        started = perf_counter()
        outcome = "success"
        token = _current_tool_call_id.set(tool_call.tool_call_id)
        try:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                output=await bounded_await(
                    tool.arun(tool_call.arguments, context=context)
                ),
            )
        except (ToolExecutionError, ToolValidationError) as error:
            outcome = type(error).__name__
            return self.failure(tool_call, error)
        except BaseException as error:
            outcome = type(error).__name__
            raise
        finally:
            _current_tool_call_id.reset(token)
            self._trace(context, tool_call, started, outcome)

    def failure(self, call: ToolCall, error: Exception) -> ToolResult:
        if self.error_policy == "raise":
            raise error
        # Remote requests for human input are control flow, not recoverable outages.
        from modmex_ai.mcp.errors import MCPInputRequired

        cause = error
        while cause is not None:
            if isinstance(cause, MCPInputRequired):
                raise cause
            cause = cause.__cause__
        code = (
            "invalid_arguments"
            if isinstance(error, ToolValidationError)
            else "tool_failed"
        )
        return ToolResult(
            tool_call_id=call.tool_call_id,
            name=call.name,
            output={
                "ok": False,
                "error": {
                    "code": code,
                    "retryable": False,
                    "message": "Tool arguments are invalid."
                    if code == "invalid_arguments"
                    else "The operation could not be confirmed. Do not claim success or repeat writes without verification.",
                },
            },
        )

    @staticmethod
    def _trace(context, call, started, outcome):
        trace = getattr(context, "trace", None)
        if trace is not None:
            try:
                trace.add(
                    "tool_execution",
                    call.name,
                    tool_call_id=call.tool_call_id,
                    duration_ms=(perf_counter() - started) * 1000,
                    outcome=outcome,
                )
            except Exception:
                pass
