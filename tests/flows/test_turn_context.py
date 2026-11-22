import pytest

from modmex_ai import (
    Agent,
    ApprovalDecision,
    ApprovalDecisionType,
    ApprovalPolicy,
    FakeModel,
    Flow,
    FlowSuspended,
    InMemorySession,
    Message,
    ModelResponse,
    ProviderState,
    ToolCall,
    tool,
)
from modmex_ai.models import ModelRequest
from modmex_ai.providers.anthropic.mapper import to_messages_payload
from modmex_ai.providers.openai.mapper import to_responses_payload
from modmex_ai.schemas import dumps
from modmex_ai.sessions import SessionItem


def _ctx(open_quotes):
    return [Message(role="developer", content=f"CONTEXTO: cliente Acme, cotizaciones abiertas={open_quotes}")]


def _run_turns(model, turns):
    flow = Flow(name="wa", entrypoint=Agent(name="sales", instructions="Atiende cotizaciones.", model=model), agents=[])
    session = InMemorySession()
    for text, quotes in turns:
        flow.run(text, session=session, turn_context=_ctx(quotes))
    return flow, session


def test_turn_context_is_sent_once_per_turn_and_never_stored():
    model = FakeModel(["a", "b", "c"])
    _, session = _run_turns(model, [("Cotízame", 3), ("Sí", 2), ("¿Vigencia?", 1)])

    last = model.requests[-1].messages
    assert sum("CONTEXTO" in str(m.content) for m in last) == 1
    assert last[-2].content == "¿Vigencia?"  # this turn's input...
    assert "cotizaciones abiertas=1" in last[-1].content  # ...then the current context
    assert not any("CONTEXTO" in dumps(item) for item in session.get_items())
    assert [item.role for item in session.get_items()] == ["user", "assistant"] * 3


def test_history_prefix_is_identical_between_turns_for_openai_caching():
    model = FakeModel(["a", "b"])
    _run_turns(model, [("Cotízame", 3), ("Sí", 2)])
    first, second = (to_responses_payload(request, "gpt-test") for request in model.requests)

    assert first["instructions"] == second["instructions"] == "Atiende cotizaciones."
    # turn 2 starts with exactly what turn 1 sent before its context
    assert second["input"][: len(first["input"]) - 1] == first["input"][:-1]
    assert second["input"][-1]["role"] == "developer"
    assert "cotizaciones abiertas=2" in second["input"][-1]["content"][0]["text"]


def test_turn_context_produces_a_valid_anthropic_payload():
    model = FakeModel(["a", "b"])
    _run_turns(model, [("Cotízame", 3), ("Sí", 2)])
    payload = to_messages_payload(model.requests[-1], "claude-opus-5-5")
    assert payload["system"] == "Atiende cotizaciones."
    roles = [message["role"] for message in payload["messages"]]
    assert roles == ["user", "assistant", "user", "system"]  # native mid-conversation system message
    assert "cotizaciones abiertas=2" in payload["messages"][-1]["content"]


def test_string_shortcut_and_handoff_agent_receives_context():
    support = Agent(name="support", instructions="Soporte.", model=FakeModel(["listo"]))
    triage = Agent(name="triage", instructions="Enruta.", handoffs=["support"], model=FakeModel([
        ModelResponse(tool_calls=[ToolCall(tool_call_id="h1", name="transfer_to_support", arguments={})]),
    ]))
    flow = Flow(name="wa", entrypoint=triage, agents=[support])
    result = flow.run("ayuda", turn_context="CONTEXTO: cliente VIP")
    assert result.output == "listo"
    assert any("cliente VIP" in dumps(m) for m in support.model.requests[0].messages)


def test_turn_context_rejects_server_side_conversation_state():
    flow = Flow(name="wa", entrypoint=Agent(name="a", instructions="x", model=FakeModel(["a"])), agents=[])
    with pytest.raises(ValueError, match="stateless"):
        flow.run("hola", turn_context="ctx", provider_state=ProviderState(previous_response_id="resp_1"))


def test_turn_context_strips_provider_data_from_stored_items():
    @tool
    def lookup() -> dict:
        """Lookup."""
        return {"ok": True}

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(tool_call_id="c1", name="lookup", arguments={},
                                           provider_data={"anthropic": {"content": [{"type": "thinking", "thinking": "", "signature": "s"}]}})]),
        "hecho",
    ])
    flow = Flow(name="wa", entrypoint=Agent(name="a", instructions="x", tools=[lookup], model=model), agents=[])
    session = InMemorySession()
    flow.run("hola", session=session, turn_context="ctx")
    calls = [item for item in session.get_items() if item.type == "function_call"]
    assert calls and all(item.provider_data is None for item in calls)


def test_suspended_state_excludes_turn_context_and_resume_reinjects_it():
    class Policy(ApprovalPolicy):
        def verify(self, request, decision):
            return True

    @tool(requires_approval=True)
    def send(value: str):
        return {"sent": value}

    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(tool_call_id="call-1", name="send", arguments={"value": "ok"})]),
        "Enviado.",
    ])
    flow = Flow(name="wa", entrypoint=Agent(name="a", instructions="x", tools=[send], model=model), agents=[])
    with pytest.raises(FlowSuspended) as error:
        flow.run("envía", flow_instance_id="f1", turn_context="CONTEXTO v1")
    state = error.value.state
    assert not any("CONTEXTO" in dumps(item) for item in state.session_snapshot.items)

    result = flow.resume(
        state,
        ApprovalDecision(request_id="call-1", decision=ApprovalDecisionType.APPROVED, signature="x"),
        approval_policy=Policy(),
        turn_context="CONTEXTO v2",
    )
    assert result.output == "Enviado."
    sent = [str(m.content) for m in model.requests[-1].messages]
    assert sum("CONTEXTO" in m for m in sent) == 1 and any("CONTEXTO v2" in m for m in sent)
    final = flow.completed_state(state, result).session_snapshot.items
    assert not any("CONTEXTO" in dumps(item) for item in final)
