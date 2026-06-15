import asyncio

import pytest

from modmex_ai import (
    Agent, ApprovalDecision, ApprovalDecisionType, ApprovalPolicy, FakeModel,
    Flow, FlowSuspended, InMemorySession, ModelResponse, ProviderState, ToolCall, tool,
)


class AsyncFake(FakeModel):
    async def acomplete(self, request):
        return self.complete(request)


def test_flow_run_async_uses_native_handoff_session_and_provider_state():
    async def run():
        final = Agent(name="final", instructions="Finish.", model=AsyncFake([
            ModelResponse(output_text="done", provider_state=ProviderState(previous_response_id="response-2")),
        ]))
        first = Agent(name="first", instructions="Route.", handoffs=["final"], model=AsyncFake([
            ModelResponse(tool_calls=[ToolCall(tool_call_id="h", name="transfer_to_final", arguments={})]),
        ]))
        session = InMemorySession(session_id="session")
        result = await Flow(name="f", entrypoint=first, agents=[final], emit=lambda output, results: [{"n": len(results)}]).run_async("go", session=session)
        return result, session

    result, session = asyncio.run(run())
    assert result.output == "done"
    assert result.last_agent_name == "final"
    assert result.events == [{"n": 2}]
    assert result.provider_state.previous_response_id == "response-2"
    assert session.get_items()


def test_flow_resume_async_executes_approved_async_tool():
    class Policy(ApprovalPolicy):
        def verify(self, request, decision):
            return decision.signature == "ok"

    called = []

    @tool(requires_approval=True)
    async def send(value: str):
        called.append(value)
        return {"sent": value}

    async def run():
        agent = Agent(name="a", instructions="Help.", tools=[send], model=AsyncFake([
            ModelResponse(tool_calls=[ToolCall(tool_call_id="tool-1", name="send", arguments={"value": "x"})]),
            ModelResponse(output_text="done"),
        ]))
        flow = Flow(name="f", entrypoint=agent, agents=[])
        with pytest.raises(FlowSuspended) as error:
            await flow.run_async("go", flow_instance_id="flow")
        return await flow.resume_async(error.value.state, ApprovalDecision(request_id="tool-1", decision=ApprovalDecisionType.APPROVED, signature="ok"), approval_policy=Policy())

    assert asyncio.run(run()).output == "done"
    assert called == ["x"]
