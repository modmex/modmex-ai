import pytest
from modmex import BaseModel

from modmex_ai import Agent, FakeModel, Handoff, ModelResponse, ToolCall, Flow
from modmex_ai.errors import MaxHandoffsExceeded, UnknownAgentError


def test_flow_rejects_unknown_entrypoint():
    flow = Flow(name="broken", entrypoint="missing", agents=[], model=FakeModel([]))

    with pytest.raises(UnknownAgentError):
        flow.run({})


def test_flow_enforces_max_handoffs():
    first = Agent(
        name="first",
        instructions="handoff",
        handoffs=[Handoff("second")],
    )
    second = Agent(
        name="second",
        instructions="handoff",
        handoffs=[Handoff("first")],
    )
    flow = Flow(
        name="loop",
        entrypoint=first,
        agents=[second],
        model=FakeModel([
            ModelResponse(tool_calls=[ToolCall(tool_call_id="handoff-1", name="transfer_to_second", arguments={})]),
            ModelResponse(tool_calls=[ToolCall(tool_call_id="handoff-2", name="transfer_to_first", arguments={})]),
        ]),
        max_handoffs=1,
    )

    with pytest.raises(MaxHandoffsExceeded):
        flow.run({})
