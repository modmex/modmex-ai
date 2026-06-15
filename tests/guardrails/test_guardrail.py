import pytest

from modmex_ai import Agent, FakeModel, Flow, GuardrailResult, Handoff, ModelResponse, ToolCall, tool
from modmex_ai.errors import (
    InputGuardrailTriggered,
    OutputGuardrailTriggered,
    ToolInputGuardrailTriggered,
    ToolOutputGuardrailTriggered,
)


class RejectingGuardrail:
    name = "rejecting"

    def check(self, _value, context=None):
        return GuardrailResult(passed=False, reason="nope")


class RejectOnceGuardrail:
    name = "reject_once"

    def __init__(self) -> None:
        self.calls = 0

    def check(self, _value, context=None):
        self.calls += 1
        return GuardrailResult(
            passed=self.calls > 1,
            reason="correct the output" if self.calls == 1 else None,
        )


def test_enforce_guardrails_raises_when_triggered():
    with pytest.raises(InputGuardrailTriggered):
        from modmex_ai.guardrails import enforce_input_guardrails

        enforce_input_guardrails("value", [RejectingGuardrail()])


def test_agent_input_guardrail_blocks_before_the_model_runs():
    model = FakeModel(["not used"])
    agent = Agent(
        name="guarded",
        instructions="Reply.",
        model=model,
        input_guardrails=[RejectingGuardrail()],
    )

    with pytest.raises(InputGuardrailTriggered):
        agent.run("blocked")

    assert model.requests == []


def test_agent_output_guardrail_blocks_parsed_output():
    agent = Agent(
        name="guarded",
        instructions="Reply.",
        model=FakeModel(["blocked output"]),
        output_guardrails=[RejectingGuardrail()],
    )

    with pytest.raises(OutputGuardrailTriggered):
        agent.run("hello")


def test_agent_retries_a_rejected_output_once_when_configured():
    guardrail = RejectOnceGuardrail()
    model = FakeModel(["invalid output", "corrected output"])
    agent = Agent(
        name="guarded",
        instructions="Reply.",
        model=model,
        output_guardrails=[guardrail],
        max_output_guardrail_retries=1,
    )

    result = agent.run("hello")

    assert result.output == "corrected output"
    assert guardrail.calls == 2
    assert len(model.requests) == 2
    assert model.requests[1].messages[-1].role == "developer"
    assert "correct the output" in model.requests[1].messages[-1].content


def test_flow_applies_output_guardrails_only_to_the_final_agent():
    triage = Agent(
        name="triage",
        instructions="Route.",
        handoffs=[Handoff("support")],
    )
    support = Agent(
        name="support",
        instructions="Help.",
        output_guardrails=[RejectingGuardrail()],
    )
    flow = Flow(
        name="guarded-flow",
        entrypoint=triage,
        agents=[support],
        model=FakeModel([
            ModelResponse(
                tool_calls=[ToolCall(tool_call_id="handoff-1", name="transfer_to_support", arguments={})],
            ),
            "blocked support output",
        ]),
    )

    with pytest.raises(OutputGuardrailTriggered):
        flow.run("I need help")


def test_tool_guardrails_run_before_and_after_the_tool_side_effect():
    calls: list[str] = []

    @tool(input_guardrails=[RejectingGuardrail()])
    def blocked_before() -> str:
        calls.append("before")
        return "never"

    with pytest.raises(ToolInputGuardrailTriggered):
        blocked_before.run({})
    assert calls == []

    @tool(output_guardrails=[RejectingGuardrail()])
    def blocked_after() -> str:
        calls.append("after")
        return "unsafe result"

    with pytest.raises(ToolOutputGuardrailTriggered):
        blocked_after.run({})
    assert calls == ["after"]
