import asyncio
import time

import pytest

from modmex_ai import Agent, FakeModel, ModelResponse, ToolCall, tool
from modmex_ai.agents.context import RunContext
from modmex_ai.budget import execution_budget, remaining_seconds
from modmex_ai.errors import (
    ApprovalRequired,
    GuardrailTriggered,
    RunTimeoutError,
    ToolExecutionError,
)


def model_for_tool(name="operation"):
    return FakeModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(tool_call_id="call-1", name=name, arguments={})]
            ),
            "Could not complete the operation.",
        ]
    )


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async_stream"])
def test_tool_failure_continues_without_leaking_exception(mode):
    def operation():
        raise RuntimeError("secret credential and internal URL")

    model = model_for_tool()
    agent = Agent(
        name="driver",
        instructions="Handle failures",
        tools=[operation],
        model=model,
        tool_error_policy="return_error",
    )

    async def execute():
        if mode == "async":
            return await agent.run_async("hello")
        events = [event async for event in agent.arun_stream("hello")]
        return events[-1].result

    if mode == "sync":
        result = agent.run("hello")
    elif mode == "stream":
        result = list(agent.run_stream("hello"))[-1].result
    else:
        # FakeModel has sync streaming; supply a native async stream for this test.
        async def astream(request):
            for event in model.stream(request):
                yield event

        model.astream = astream
        result = asyncio.run(execute())
    assert result.output == "Could not complete the operation."
    error = model.requests[-1].messages[-1].output
    assert error["ok"] is False
    assert error["error"]["retryable"] is False
    assert "secret" not in str(error)
    assert len(model.requests) == 2
    assert any(
        step.type == "tool_execution" and step.data["duration_ms"] >= 0
        for step in result.trace.steps
    )


def test_default_still_raises_and_approval_and_guardrails_never_become_tool_errors():
    def operation():
        raise RuntimeError("failure")

    with pytest.raises(ToolExecutionError):
        Agent(
            name="test", instructions="", tools=[operation], model=model_for_tool()
        ).run("hello")
    for error in (GuardrailTriggered("blocked"), RunTimeoutError("expired")):

        def operation():
            raise error

        with pytest.raises(type(error)):
            Agent(
                name="test",
                instructions="",
                tools=[operation],
                model=model_for_tool(),
                tool_error_policy="return_error",
            ).run("hello")

    @tool(requires_approval=True)
    def operation():
        raise AssertionError("must never execute")

    with pytest.raises(ApprovalRequired):
        Agent(
            name="test",
            instructions="",
            tools=[operation],
            model=model_for_tool(),
            tool_error_policy="return_error",
        ).run("hello")


def test_async_budget_cancels_tool_without_followup_or_duplicate_write():
    cancelled = []

    async def operation():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    model = model_for_tool()
    agent = Agent(
        name="test",
        instructions="",
        tools=[operation],
        model=model,
        execution_timeout=0.02,
        tool_error_policy="return_error",
    )
    with pytest.raises(RunTimeoutError):
        asyncio.run(agent.run_async("hello"))
    assert cancelled == [True]
    assert len(model.requests) == 1
    assert remaining_seconds() is None


def test_sync_budget_detects_overrun_and_nested_budget_cannot_extend_parent():
    class SlowModel(FakeModel):
        def complete(self, request):
            time.sleep(0.02)
            return super().complete(request)

    with pytest.raises(RunTimeoutError):
        Agent(
            name="test",
            instructions="",
            model=SlowModel(["late"]),
            execution_timeout=0.001,
        ).run("hello")
    with execution_budget(1):
        before = remaining_seconds()
        with execution_budget(100):
            assert remaining_seconds() <= before
    assert remaining_seconds() is None


def test_trace_reports_context_sizes_without_content():
    context = RunContext(input="hello")
    result = Agent(
        name="test", instructions="private instructions", model=FakeModel(["done"])
    ).run("private content", context=context)
    request = next(step for step in result.trace.steps if step.type == "model_request")
    assert request.data["context_bytes"] > 0
    assert request.data["message_count"] == 2
    assert "private" not in str(request.data)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_budget(timeout):
    with pytest.raises(ValueError):
        Agent(name="test", instructions="", execution_timeout=timeout)


@pytest.mark.parametrize("streaming", [False, True])
def test_remote_mcp_failure_recovers_and_input_request_propagates(streaming):
    from modmex_ai.mcp.tools import RemoteTool
    from modmex_ai.mcp.errors import MCPError, MCPInputRequired

    class Client:
        async def call_tool(self, *args, **kwargs):
            raise self.error

    class StreamingModel(FakeModel):
        async def astream(self, request):
            yield self.complete(request)

    client = Client()
    definition = {
        "name": "operation",
        "inputSchema": {"type": "object", "properties": {}},
    }

    async def run(error):
        client.error = error
        base = model_for_tool()
        agent = Agent(
            name="driver",
            instructions="",
            tools=[RemoteTool(client, definition)],
            model=StreamingModel(base._responses),
            tool_error_policy="return_error",
        )
        if streaming:
            events = [event async for event in agent.arun_stream("hello")]
            return events[-1].result
        return await agent.run_async("hello")

    assert asyncio.run(run(MCPError(-32603, "private failure"))).output
    with pytest.raises(MCPInputRequired):
        asyncio.run(run(MCPInputRequired([])))


def test_async_stream_deadline_closes_provider_stream():
    closed = []

    class StreamingModel(FakeModel):
        async def astream(self, request):
            try:
                await asyncio.sleep(10)
                yield self.complete(request)
            finally:
                closed.append(True)

    async def run():
        agent = Agent(
            name="test",
            instructions="",
            model=StreamingModel(["late"]),
            execution_timeout=0.02,
        )
        return [event async for event in agent.arun_stream("hello")]

    with pytest.raises(RunTimeoutError):
        asyncio.run(run())
    assert closed == [True]


def test_parallel_recovery_keeps_successful_results():
    async def failed():
        raise ValueError("failed")

    async def successful():
        return {"saved": True}

    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(tool_call_id="1", name="failed", arguments={}),
                    ToolCall(tool_call_id="2", name="successful", arguments={}),
                ]
            ),
            "Partial completion",
        ]
    )
    result = asyncio.run(
        Agent(
            name="test",
            instructions="",
            model=model,
            tools=[failed, successful],
            parallel_tool_calls=True,
            tool_error_policy="return_error",
        ).run_async("hello")
    )
    outputs = [
        item.output for item in result.items if item.type == "function_call_output"
    ]
    assert outputs[0]["ok"] is False
    assert outputs[1] == {"saved": True}
