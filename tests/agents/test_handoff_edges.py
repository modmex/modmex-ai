import asyncio

import pytest

from modmex_ai import Agent, FakeModel, Handoff, ModelResponse, ToolCall
from modmex_ai.agents.handoff import HandoffInputData
from modmex_ai.sessions import SessionItem


def test_handoff_validates_callback_and_filters_history():
    with pytest.raises(ValueError):
        Handoff("a", input_type=dict)
    seen = []
    handoff = Handoff("Billing Team!", input_type=dict, on_handoff=lambda ctx, value: seen.append(value), input_filter=lambda data: HandoffInputData(history=data.history[-1:], input=data.input))
    assert handoff.name == "transfer_to_billing_team"
    assert handoff.invoke(None, {"id": 1}) == {"id": 1}
    assert seen == [{"id": 1}]
    assert len(handoff.filter_history([SessionItem(role="user", content="x")], {})) == 1


def test_agent_decorator_nested_tool_and_stream_errors():
    outer = Agent(name="outer", instructions="x", model=FakeModel(["inner"]))
    nested = outer.as_tool()
    assert nested.run({"input": "go"}) == "inner"
    agent = Agent(name="a", instructions="x")
    @agent.tool(name="hello")
    def hello(value: str):
        return value
    assert agent.tools[0].name == "hello"
    with pytest.raises(ValueError):
        Agent(name="bad", instructions="x", max_output_guardrail_retries=-1)
    with pytest.raises(ValueError):
        agent.run("x")
    with pytest.raises(NotImplementedError):
        asyncio.run(_consume(agent.arun_stream("x", model=FakeModel(["x"]))))


async def _consume(stream):
    return [item async for item in stream]
