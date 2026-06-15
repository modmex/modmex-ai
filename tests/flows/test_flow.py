from modmex import BaseModel

from modmex_ai import Agent, FakeModel, Handoff, InMemorySession, ModelResponse, SessionItem, ToolCall, Flow, Usage, tool


class TriageOutput(BaseModel):
    intent: str
    confidence: float


class SupportOutput(BaseModel):
    events: list[dict]
    reply: str | None = None


def test_flow_runs_entrypoint_and_controlled_handoff():
    triage = Agent(
        name="triage",
        instructions="Classify the message.",
        output_type=TriageOutput,
        handoffs=[Handoff("support")],
    )
    support = Agent(
        name="support",
        instructions="Prepare support events.",
        output_type=SupportOutput,
    )
    model = FakeModel([
        ModelResponse(
            tool_calls=[ToolCall(tool_call_id="handoff-1", name="transfer_to_support", arguments={})],
            usage=Usage(input_tokens=10, output_tokens=4, total_tokens=14),
        ),
        ModelResponse(
            output_text='{"events":[{"type":"human.review.required"}],"reply":null}',
            usage=Usage(input_tokens=8, output_tokens=5, total_tokens=13),
        ),
    ])

    flow = Flow(
        name="analyze-message",
        entrypoint=triage,
        agents=[support],
        model=model,
        emit=lambda output, _results: output.model_dump()["events"],
    )

    result = flow.run({"type": "message.created", "text": "I need help"})

    assert [item.agent for item in result.agent_results] == ["triage", "support"]
    assert result.last_agent is support
    assert result.last_agent_name == "support"
    assert result.continuation.agent_name == "support"
    assert result.events == [{"type": "human.review.required"}]
    assert result.usage.input_tokens == 18
    assert result.usage.output_tokens == 9
    assert result.usage.total_tokens == 27
    assert len(result.trace.steps) >= 3
    assert [item.type for item in model.requests[1].messages[-2:]] == [
        "handoff_call",
        "handoff_call_output",
    ]


def test_flow_uses_explicit_starting_agent_without_changing_its_entrypoint():
    triage = Agent(name="triage", instructions="Route.")
    support = Agent(name="support", instructions="Help.")
    model = FakeModel(["Handled by support"])
    flow = Flow(
        name="routing-flow",
        entrypoint=triage,
        agents=[support],
        model=model,
    )

    result = flow.run("I still need help", starting_agent="support")

    assert flow.entrypoint == "triage"
    assert result.last_agent is support
    assert result.last_agent_name == "support"
    assert [item.agent for item in result.agent_results] == ["support"]
    assert "Help." in model.requests[0].messages[0].content


def test_flow_result_can_continue_from_the_last_agent_without_storing_routing_in_session():
    triage = Agent(name="triage", instructions="Route.", handoffs=[Handoff("support")])
    support = Agent(name="support", instructions="Help.")
    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(tool_call_id="handoff-1", name="transfer_to_support", arguments={})]),
        "First support reply",
        "Second support reply",
    ])
    flow = Flow(name="sticky-flow", entrypoint=triage, agents=[support], model=model)
    session = InMemorySession(session_id="conversation-1")

    first = flow.run("I need help", session=session)
    second = flow.run(
        "I still need help",
        session=session,
        starting_agent=first.last_agent,
        provider_state=first.continuation.provider_state,
    )

    assert [item.agent for item in first.agent_results] == ["triage", "support"]
    assert [item.agent for item in second.agent_results] == ["support"]
    assert second.last_agent_name == "support"


def test_flow_passes_initial_input_directly_and_context_separately():
    @tool
    def read_context(ctx=None):
        return ctx.context["customer_id"]

    agent = Agent(
        name="agent",
        instructions="Use context.",
        tools=[read_context],
        output_type=TriageOutput,
    )
    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(tool_call_id="call-1", name="read_context", arguments={})]),
        '{"intent":"support","confidence":0.9}',
    ])
    flow = Flow(name="context-flow", entrypoint=agent, agents=[], model=model)

    result = flow.run("hello", context={"customer_id": "cust-1"})

    assert result.output.intent == "support"
    assert model.requests[0].messages[-1].content == "hello"


def test_flow_result_returns_replay_ready_input_list():
    agent = Agent(
        name="agent",
        instructions="Reply.",
    )
    flow = Flow(
        name="chat",
        entrypoint=agent,
        agents=[],
        model=FakeModel(["Hello Alex"]),
    )

    result = flow.run("Hello")

    assert result.to_input_list() == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello Alex"},
    ]


def test_flow_uses_session_history_and_persists_new_turn():
    agent = Agent(
        name="agent",
        instructions="Reply.",
    )
    model = FakeModel(["I remember"])
    flow = Flow(name="chat", entrypoint=agent, agents=[], model=model)
    session = InMemorySession(session_id="conversation-1")
    session.add_items([SessionItem(role="assistant", content="Hi Alex")])

    result = flow.run("What did I say?", session=session)

    assert [message.role for message in model.requests[-1].messages] == [
        "system",
        "assistant",
        "user",
    ]
    assert model.requests[-1].messages[-2].content == "Hi Alex"
    assert model.requests[-1].messages[-1].content == "What did I say?"
    assert result.to_input_list() == [
        {"role": "assistant", "content": "Hi Alex"},
        {"role": "user", "content": "What did I say?"},
        {"role": "assistant", "content": "I remember"},
    ]
    assert [item.to_input() for item in session.get_items()] == [
        {"role": "assistant", "content": "Hi Alex"},
        {"role": "user", "content": "What did I say?"},
        {"role": "assistant", "content": "I remember"},
    ]


def test_flow_passes_session_history_to_handoff_agent():
    triage = Agent(
        name="triage",
        instructions="Route.",
        output_type=TriageOutput,
        handoffs=[Handoff("support")],
    )
    support = Agent(
        name="support",
        instructions="Use conversation history.",
        output_type=SupportOutput,
    )
    model = FakeModel([
        ModelResponse(
            tool_calls=[ToolCall(tool_call_id="handoff-1", name="transfer_to_support", arguments={})],
        ),
        '{"events":[{"type":"human.review.required"}],"reply":null}',
    ])
    flow = Flow(name="history-flow", entrypoint=triage, agents=[support], model=model)
    session = InMemorySession(session_id="conversation-1")
    session.add_items([
        SessionItem(role="user", content="first message"),
        SessionItem(role="assistant", content="first reply"),
    ])

    flow.run("second message", session=session)

    support_messages = model.requests[1].messages
    assert [message.role for message in support_messages[:4]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert support_messages[1].content == "first message"
    assert support_messages[2].content == "first reply"
    assert support_messages[3].content == "second message"
    assert [item.type for item in support_messages[4:]] == [
        "handoff_call",
        "handoff_call_output",
    ]


def test_agent_executes_tool_calls_before_final_output():
    @tool
    def get_customer(customer_id: str) -> dict:
        return {"id": customer_id, "tier": "gold"}

    agent = Agent(
        name="triage",
        instructions="Classify the message.",
        tools=[get_customer],
        output_type=TriageOutput,
    )
    model = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call-1",
                    name="get_customer",
                    arguments={"customer_id": "cust-1"},
                )
            ]
        ),
        '{"intent":"support","confidence":0.88}',
    ])

    result = agent.run({"customer_id": "cust-1"}, model=model)

    assert result.output.intent == "support"
    assert model.requests[0].tools[0]["name"] == "get_customer"
    assert model.requests[1].messages[-1].to_input() == {
        "type": "function_call_output",
        "tool_call_id": "call-1",
        "output": '{"id":"cust-1","tier":"gold"}',
    }
