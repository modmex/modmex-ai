import asyncio
import json

import pytest

from modmex_ai import (
    Agent,
    FakeModel,
    Flow,
    Handoff,
    ModelResponse,
    ToolCall,
    ProviderState,
    ModelSettings,
    OpenAISettings,
    Message,
    ModelRequest,
    ApprovalPolicy,
    ApprovalDecision,
    ApprovalDecisionType,
    tool,
    FlowSuspended,
)
from modmex_ai.errors import UnknownToolError
from modmex_ai.flows.state import PersistedFlowState
from modmex_ai.guardrails import GuardrailResult
from modmex_ai.http import HttpResponse
from modmex_ai.providers.openai import OpenAIResponsesModel
from modmex_ai.providers.openai.mapper import to_responses_payload, to_chat_payload


def response(n, *, call=None, text="done", conversation=None):
    output = (
        [{"type": "function_call", "call_id": str(n), "name": call, "arguments": "{}"}]
        if call
        else [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
    )
    return {
        "id": f"r{n}",
        "output": output,
        **({"conversation": {"id": conversation}} if conversation else {}),
    }


class Transport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.payloads = []

    def post_json(self, url, *, data, **kwargs):
        # Snapshot the wire payload, not a mutable ModelRequest reference.
        self.payloads.append(json.loads(json.dumps(data)))
        return HttpResponse(status_code=200, headers={}, body=self.responses.pop(0))

    def post_json_stream(self, url, *, data, **kwargs):
        result = self.post_json(url, data=data)
        yield (
            "data: "
            + json.dumps({"type": "response.completed", "response": result.body})
            + "\n\n"
        ).encode()


def execute(agent, mode, text="hello", **kwargs):
    async def run():
        if mode == "async":
            return await agent.run_async(text, **kwargs)
        events = [event async for event in agent.arun_stream(text, **kwargs)]
        return events[-1].result

    if mode == "sync":
        return agent.run(text, **kwargs)
    if mode == "stream":
        return list(agent.run_stream(text, **kwargs))[-1].result
    return asyncio.run(run())


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async_stream"])
@pytest.mark.parametrize("state_mode", ["response_chain", "conversation", "stateless"])
def test_real_adapter_multistep_cursor_and_persisted_next_turn(mode, state_mode):
    transport = Transport(
        [
            response(1, call="lookup"),
            response(2, call="lookup"),
            response(3),
            response(4),
        ]
    )
    model = OpenAIResponsesModel("test", http_client=transport)
    agent = Agent(name="driver", instructions="Driver instructions", model=model)

    @agent.tool
    def lookup():
        return {"found": True}

    state = ProviderState(
        mode=state_mode, conversation_id="c1" if state_mode == "conversation" else None
    )
    result = execute(agent, mode, provider_state=state)
    assert result.output == "done"
    second, third = transport.payloads[1:3]
    if state_mode == "stateless":
        assert all(
            "previous_response_id" not in body and "conversation" not in body
            for body in transport.payloads
        )
        assert len(third["input"]) > len(second["input"])
    else:
        assert [item["call_id"] for item in second["input"]] == ["1"]
        assert [item["call_id"] for item in third["input"]] == ["2"]
        if state_mode == "response_chain":
            assert second["previous_response_id"] == "r1"
            assert third["previous_response_id"] == "r2"
        else:
            assert all(
                body["conversation"] == "c1" and "previous_response_id" not in body
                for body in transport.payloads
            )
    restored = ProviderState(**result.provider_state.model_dump())
    execute(agent, mode, text="next turn", provider_state=restored)
    assert len(transport.payloads[-1]["input"]) == 1
    assert transport.payloads[-1]["input"][0]["role"] == "user"


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async_stream"])
def test_remote_guardrail_correction_is_not_lost_after_tools(mode):
    class RejectBad:
        name = "reject_bad"

        def check(self, value, context=None):
            return GuardrailResult(passed=value != "bad", reason="Do not say bad")

    transport = Transport(
        [response(1, call="lookup"), response(2, text="bad"), response(3, text="good")]
    )

    def lookup():
        return "ok"

    agent = Agent(
        name="a",
        instructions="Help",
        tools=[lookup],
        model=OpenAIResponsesModel("test", http_client=transport),
        output_guardrails=[RejectBad()],
        max_output_guardrail_retries=1,
    )
    assert execute(agent, mode).output == "good"
    correction = transport.payloads[-1]
    assert correction["previous_response_id"] == "r2"
    assert correction["input"] == []
    assert "Do not say bad" in correction["instructions"]
    assert (
        "Previous candidate" in correction["instructions"]
        or "previous candidate" in correction["instructions"]
    )


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async_stream"])
def test_remote_handoff_only_submits_new_acknowledgement(mode):
    transport = Transport([response(1, call="transfer_to_support"), response(2)])
    model = OpenAIResponsesModel("test", http_client=transport)
    first = Agent(name="triage", instructions="Route", handoffs=["support"])
    second = Agent(name="support", instructions="Support")
    flow = Flow(name="f", entrypoint=first, agents=[second], model=model)
    if mode == "sync":
        result = flow.run("help")
    elif mode == "stream":
        result = list(flow.run_stream("help"))[-1].result
    else:

        async def run():
            if mode == "async":
                return await flow.run_async("help")
            events = [event async for event in flow.arun_stream("help")]
            return events[-1].result

        result = asyncio.run(run())
    assert result.output == "done"
    payload = transport.payloads[-1]
    assert payload["instructions"] == "Support"
    assert payload["previous_response_id"] == "r1"
    assert len(payload["input"]) == 1
    assert payload["input"][0]["type"] == "function_call_output"


@pytest.mark.parametrize("mode", ["sync", "async", "stream", "async_stream"])
def test_approval_checkpoint_resumes_latest_remote_response_without_repeating_reads(
    mode,
):
    calls = []

    def lookup():
        calls.append("read")
        return "found"

    @tool(requires_approval=True)
    def send():
        calls.append("write")
        return "sent"

    class Policy(ApprovalPolicy):
        def verify(self, request, decision):
            return decision.signature == "ok"

    transport = Transport(
        [response(1, call="lookup"), response(2, call="send"), response(3)]
    )
    agent = Agent(
        name="a",
        instructions="help",
        tools=[lookup, send],
        model=OpenAIResponsesModel("test", http_client=transport),
    )
    flow = Flow(name="f", entrypoint=agent, agents=[])
    try:
        if mode == "sync":
            flow.run("go")
        elif mode == "stream":
            list(flow.run_stream("go"))
        elif mode == "async":
            asyncio.run(flow.run_async("go"))
        else:

            async def consume():
                return [event async for event in flow.arun_stream("go")]

            asyncio.run(consume())
        raise AssertionError("Expected suspension")
    except FlowSuspended as error:
        state = PersistedFlowState(**error.state.model_dump())
    assert state.suspension.provider_state.previous_response_id == "r2"
    decision = ApprovalDecision(
        request_id="2", decision=ApprovalDecisionType.APPROVED, signature="ok"
    )
    if mode in {"async", "async_stream"}:
        result = asyncio.run(
            flow.resume_async(state, decision, approval_policy=Policy())
        )
    else:
        result = flow.resume(state, decision, approval_policy=Policy())
    assert result.output == "done"
    assert calls == ["read", "write"]
    last = transport.payloads[-1]
    assert last["previous_response_id"] == "r2"
    assert [item["call_id"] for item in last["input"]] == ["2"]


def test_mixed_handoff_rejected_before_any_side_effect_and_selector_enforced():
    writes = []

    def write():
        writes.append(True)

    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(tool_call_id="1", name="write", arguments={}),
                    ToolCall(tool_call_id="2", name="transfer_to_b", arguments={}),
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="sole tool"):
        Agent(
            name="a", instructions="", tools=[write], handoffs=["b"], model=model
        ).run("go")
    assert not writes
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(tool_call_id="1", name="write", arguments={})]
            )
        ]
    )
    with pytest.raises(UnknownToolError):
        Agent(
            name="a",
            instructions="",
            tools=[write],
            model=model,
            tool_selector=lambda context, names: [],
        ).run("go")
    assert not writes


def test_selector_rechecks_availability_keeps_registered_order_and_model_parallel_is_independent():
    def first(ctx=None):
        ctx.state["phase"] = "second"
        return "ok"

    def second():
        return "ok"

    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(tool_call_id="1", name="first", arguments={})]
            ),
            "done",
        ]
    )
    agent = Agent(
        name="a",
        instructions="",
        tools=[first, second],
        model=model,
        settings=ModelSettings(
            parallel_tool_calls=True,
            openai=OpenAISettings(
                prompt_cache_key="workspace:profile:v1", prompt_cache_retention="24h"
            ),
        ),
        tool_selector=lambda ctx, names: [ctx.state.get("phase", "first")],
    )
    assert agent.parallel_tool_calls is False
    agent.run("go")
    assert [schema["name"] for schema in model.requests[0].tools] == ["first"]
    assert [schema["name"] for schema in model.requests[1].tools] == ["second"]
    for mapper in (to_responses_payload, to_chat_payload):
        payload = mapper(model.requests[1], "test")
        assert payload["parallel_tool_calls"] is True
        assert payload["prompt_cache_key"] == "workspace:profile:v1"
        assert payload["prompt_cache_retention"] == "24h"


def test_dual_ids_and_unsafe_mode_changes_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        to_responses_payload(
            ModelRequest(
                messages=[],
                provider_state=ProviderState(
                    previous_response_id="r", conversation_id="c"
                ),
            ),
            "test",
        )
    for state in (
        ProviderState(mode="conversation"),
        ProviderState(mode="stateless", previous_response_id="r"),
        ProviderState(mode="response_chain", conversation_id="c"),
    ):
        with pytest.raises(ValueError):
            state.resolved_mode


@pytest.mark.parametrize("mode", ["async", "async_stream"])
def test_parallel_remote_results_preserve_order_and_do_not_replay_prior_outputs(mode):
    initial = response(1, call="lookup")
    initial["output"].append(
        {
            "type": "function_call",
            "call_id": "other",
            "name": "lookup",
            "arguments": "{}",
        }
    )
    transport = Transport([initial, response(2, call="lookup"), response(3)])

    async def lookup():
        await asyncio.sleep(0)
        return "found"

    agent = Agent(
        name="a",
        instructions="",
        tools=[lookup],
        parallel_tool_calls=True,
        model=OpenAIResponsesModel("test", http_client=transport),
    )
    assert execute(agent, mode).output == "done"
    assert [item["call_id"] for item in transport.payloads[1]["input"]] == [
        "1",
        "other",
    ]
    assert [item["call_id"] for item in transport.payloads[2]["input"]] == ["2"]


def test_mixed_approval_batch_rejected_before_other_writes():
    writes = []

    def first():
        writes.append("first")

    @tool(requires_approval=True)
    def approved():
        writes.append("approved")

    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(tool_call_id="1", name="first", arguments={}),
                    ToolCall(tool_call_id="2", name="approved", arguments={}),
                ]
            )
        ]
    )
    with pytest.raises(ValueError, match="one at a time"):
        Agent(name="a", instructions="", tools=[first, approved], model=model).run("go")
    assert writes == []


def test_remote_history_filter_rejected_and_missing_state_not_silently_reused():
    agent = Agent(
        name="a",
        instructions="",
        model=FakeModel(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall(tool_call_id="1", name="transfer_to_b", arguments={})
                    ],
                    provider_state=ProviderState(previous_response_id="r1"),
                )
            ]
        ),
        handoffs=[Handoff("b", input_filter=lambda data: [])],
    )
    with pytest.raises(ValueError, match="history filters"):
        agent.run("go")
    with pytest.raises(ValueError, match="continuation state"):
        Agent(name="a", instructions="", model=FakeModel(["done"])).run(
            "go", provider_state=ProviderState(previous_response_id="r1")
        )


def test_response_with_both_ids_selects_conversation_without_emitting_both():
    from modmex_ai.providers.openai.mapper import from_responses_payload

    parsed = from_responses_payload(
        response(1, conversation="c1"), headers={}, status_code=200, model="test"
    )
    assert parsed.provider_state.mode == "conversation"
    assert parsed.provider_state.previous_response_id is None
    wire = to_responses_payload(
        ModelRequest(messages=[], provider_state=parsed.provider_state), "test"
    )
    assert wire["conversation"] == "c1"
    assert "previous_response_id" not in wire


def test_retry_reuses_pending_input_and_cache_fields_without_advancing_cursor():
    httpx = pytest.importorskip("httpx")
    from modmex_ai.http import AsyncHttpClient
    from modmex_ai.http.retries import RetryConfig

    bodies, retries = [], []

    def respond(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(429, json={"error": {"code": "rate_limit_exceeded"}})
        return httpx.Response(
            200, json=response(1, call="lookup") if len(bodies) == 2 else response(2)
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            transport = AsyncHttpClient(
                client=client,
                retry=RetryConfig(backoff_seconds=0, on_retry=retries.append),
            )

            def lookup():
                return "found"

            agent = Agent(
                name="a",
                instructions="stable",
                tools=[lookup],
                model=OpenAIResponsesModel("test", async_http_client=transport),
                settings=ModelSettings(
                    openai=OpenAISettings(prompt_cache_key="stable-key")
                ),
            )
            return await agent.run_async("go")

    assert asyncio.run(run()).output == "done"
    assert bodies[0] == bodies[1]
    assert bodies[2]["previous_response_id"] == "r1"
    assert [item["call_id"] for item in bodies[2]["input"]] == ["1"]
    assert len(retries) == 1


def test_reordered_selector_does_not_reorder_schema_prefix():
    def first():
        return "first"

    def second():
        return "second"

    model = FakeModel(["done"])
    Agent(
        name="a",
        instructions="",
        model=model,
        tools=[first, second],
        tool_selector=lambda ctx, names: list(reversed(names)),
    ).run("go")
    assert [schema["name"] for schema in model.requests[0].tools] == ["first", "second"]


def test_stream_metrics_distinguish_first_event_from_first_text():
    from modmex_ai import ModelStreamEvent, ModelStreamEventType

    class StreamModel(FakeModel):
        def stream(self, request):
            yield ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL_DELTA,
                tool_call=ToolCall(tool_call_id="1", name="lookup", arguments={}),
            )
            yield ModelStreamEvent(
                type=ModelStreamEventType.TEXT_DELTA, text_delta="Hi"
            )
            yield ModelStreamEvent.completed(ModelResponse(output_text="Hi"))

    result = list(
        Agent(name="a", instructions="", model=StreamModel([])).run_stream("hi")
    )[-1].result
    steps = result.trace.steps
    event = next(step for step in steps if step.type == "model_first_fragment")
    text = next(step for step in steps if step.type == "model_first_token")
    assert steps.index(event) < steps.index(text)
    assert text.data["duration_ms"] >= event.data["duration_ms"]
