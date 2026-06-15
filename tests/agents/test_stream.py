import asyncio

from modmex_ai import (
    Agent,
    AgentStreamEventType,
    FakeModel,
    Flow,
    FlowStreamEventType,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventType,
    ToolCall,
    InMemorySession,
    ProviderState,
)


def test_agent_stream_yields_text_deltas_then_the_completed_result():
    class StreamingModel(FakeModel):
        def stream(self, request):
            self.requests.append(request)
            yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta="Hel")
            yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta="lo")
            yield ModelStreamEvent.completed(ModelResponse(output_text="Hello"))

    events = list(Agent(
        name="assistant",
        instructions="Help.",
        model=StreamingModel([]),
    ).run_stream("hi"))

    assert [event.text_delta for event in events[:-1]] == ["Hel", "lo"]
    assert events[-1].type == AgentStreamEventType.COMPLETED
    assert events[-1].result.output == "Hello"


def test_flow_stream_continues_after_a_handoff_and_completes():
    support = Agent(
        name="support",
        instructions="Help.",
        model=FakeModel(["Resolved."]),
    )
    triage = Agent(
        name="triage",
        instructions="Route.",
        handoffs=["support"],
        model=FakeModel([ModelResponse(tool_calls=[ToolCall(
            tool_call_id="handoff-1",
            name="transfer_to_support",
            arguments={},
        )])]),
    )

    events = list(Flow(name="support-flow", entrypoint=triage, agents=[support]).run_stream("help"))

    assert events[0].type == FlowStreamEventType.AGENT
    assert events[0].agent_event.type == AgentStreamEventType.HANDOFF
    assert events[-1].type == FlowStreamEventType.COMPLETED
    assert events[-1].result.output == "Resolved."
    assert events[-1].result.last_agent is support


def test_agent_arun_stream_uses_native_async_events():
    class AsyncStreamingModel(FakeModel):
        async def acomplete(self, request):
            raise AssertionError("astream should be used")

        async def _events(self):
            yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta="Hi")
            yield ModelStreamEvent.completed(ModelResponse(output_text="Hi"))

        def astream(self, request):
            self.requests.append(request)
            return self._events()

    async def run():
        agent = Agent(name="async", instructions="Help.", model=AsyncStreamingModel([]))
        return [event async for event in agent.arun_stream("hello")]

    events = asyncio.run(run())
    assert [event.type for event in events] == [
        AgentStreamEventType.TEXT_DELTA,
        AgentStreamEventType.COMPLETED,
    ]


def test_flow_arun_stream_uses_native_async_agent_stream():
    class AsyncStreamingModel(FakeModel):
        async def _events(self):
            yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta="Hi")
            yield ModelStreamEvent.completed(ModelResponse(output_text="Hi"))

        def astream(self, request):
            self.requests.append(request)
            return self._events()

    async def run():
        agent = Agent(name="async", instructions="Help.", model=AsyncStreamingModel([]))
        flow = Flow(name="async-flow", entrypoint=agent, agents=[])
        return [event async for event in flow.arun_stream("hello")]

    events = asyncio.run(run())
    assert events[-1].type == FlowStreamEventType.COMPLETED
    assert events[-1].result.output == "Hi"


def test_flow_arun_stream_handles_handoff_session_and_provider_state():
    class AsyncStreamingModel(FakeModel):
        async def _events(self, response):
            yield ModelStreamEvent.completed(response)

        def astream(self, request):
            self.requests.append(request)
            return self._events(self._responses.pop(0))

    async def run():
        support = Agent(name="support", instructions="Resolve.", model=AsyncStreamingModel([
            ModelResponse(output_text="Resolved.", provider_state=ProviderState(previous_response_id="r2")),
        ]))
        triage = Agent(name="triage", instructions="Route.", handoffs=["support"], model=AsyncStreamingModel([
            ModelResponse(tool_calls=[ToolCall(tool_call_id="h1", name="transfer_to_support", arguments={})]),
        ]))
        session = InMemorySession(session_id="s")
        flow = Flow(name="f", entrypoint=triage, agents=[support], emit=lambda output, results: [{"output": output, "count": len(results)}])
        return [event async for event in flow.arun_stream("help", session=session)] , session

    events, session = asyncio.run(run())
    assert events[-1].result.output == "Resolved."
    assert events[-1].result.events == [{"output": "Resolved.", "count": 2}]
    assert len(session.get_items()) >= 3


def test_agent_stream_executes_a_tool_and_emits_tool_events():
    class StreamingModel(FakeModel):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def stream(self, request):
            self.requests.append(request)
            self.calls += 1
            if self.calls == 1:
                yield ModelResponse(tool_calls=[ToolCall(tool_call_id="t", name="lookup", arguments={})])
            else:
                yield ModelResponse(output_text="done")

    def lookup():
        return {"ok": True}

    events = list(Agent(name="a", instructions="x", tools=[lookup], model=StreamingModel()).run_stream("go"))
    assert [event.type for event in events][-3:] == [
        AgentStreamEventType.TOOL_CALL_DELTA,
        AgentStreamEventType.TOOL_FINISHED,
        AgentStreamEventType.COMPLETED,
    ]
