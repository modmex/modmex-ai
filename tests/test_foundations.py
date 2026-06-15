import pytest

from modmex_ai import (
    Agent,
    ApprovalDecision,
    ApprovalDecisionType,
    ApprovalPolicy,
    ApprovalRequest,
    EvalCase,
    EvalRunner,
    FakeModel,
    Flow,
    FlowStateStatus,
    FlowSuspended,
    InMemoryFlowStateStore,
    PersistedFlowState,
    FlowStateConflictError,
    InMemoryDurableSessionStore,
    InMemorySession,
    ModelResponse,
    ObservabilityObserver,
    SessionConflictError,
    SessionSnapshot,
    ToolCall,
)
from modmex_ai.errors import ApprovalRequired
from modmex_ai.agents import RunContext
from modmex_ai.tools import ToolExecutor, tool


def test_durable_snapshot_compacts_and_detects_conflicts():
    store = InMemoryDurableSessionStore()
    snapshot = SessionSnapshot.from_session(InMemorySession(session_id="s"))
    stored = store.save(snapshot, expected_revision=0)

    assert stored.revision == 1
    assert store.load("s").compact(max_items=0, summary="short").summary == "short"
    with pytest.raises(SessionConflictError):
        store.save(snapshot, expected_revision=0)


def test_trace_observer_receives_model_and_tool_events():
    received = []

    class Observer(ObservabilityObserver):
        def on_event(self, event):
            received.append(event.type)

    context = RunContext(input="hello")
    context.trace.add_observer(Observer())
    Agent(name="a", instructions="Help.", model=FakeModel(["ok"])).run("hello", context=context)

    assert received == ["model_request", "model_response"]


def test_trace_observer_failures_do_not_break_an_agent_run():
    class BrokenObserver(ObservabilityObserver):
        def on_event(self, event):
            raise RuntimeError("metrics unavailable")

    context = RunContext(input="hello")
    context.trace.add_observer(BrokenObserver())
    assert Agent(name="a", instructions="Help.", model=FakeModel(["ok"])).run("hello", context=context).output == "ok"
    assert context.trace.observer_errors == ["metrics unavailable", "metrics unavailable"]


def test_approval_required_tool_resumes_after_a_verified_decision():
    class Policy(ApprovalPolicy):
        def verify(self, request: ApprovalRequest, decision: ApprovalDecision) -> bool:
            return decision.signature == "verified"

    @tool(requires_approval=True, approval_reason="external side effect")
    def send(value: str):
        return value

    context = RunContext(input=None)
    executor = ToolExecutor([send])
    call = ToolCall(tool_call_id="call-1", name="send", arguments={"value": "ok"})
    with pytest.raises(ApprovalRequired) as error:
        executor.execute(call, context=context)

    assert error.value.request.tool_name == "send"
    context.state["approval_policy"] = Policy()
    context.state["approval_decisions"] = {
        "call-1": ApprovalDecision(request_id="call-1", decision=ApprovalDecisionType.APPROVED, signature="verified")
    }
    assert executor.execute(call, context=context).output == "ok"


def test_flow_suspends_and_resumes_the_same_approved_tool_call():
    class Policy(ApprovalPolicy):
        def verify(self, request: ApprovalRequest, decision: ApprovalDecision) -> bool:
            return decision.signature == "verified"

    calls = []

    @tool(requires_approval=True)
    def send(value: str):
        calls.append(value)
        return {"sent": value}

    agent = Agent(
        name="assistant",
        instructions="Help.",
        tools=[send],
        model=FakeModel([
            ModelResponse(tool_calls=[ToolCall(tool_call_id="call-1", name="send", arguments={"value": "ok"})]),
            "Delivered.",
        ]),
    )
    flow = Flow(name="approval-flow", entrypoint=agent, agents=[])

    with pytest.raises(FlowSuspended) as error:
        flow.run("send it", flow_instance_id="flow-1", idempotency_key="message-1")

    suspended = error.value.state
    assert suspended.status == FlowStateStatus.SUSPENDED
    assert suspended.suspension.pending_tool_call.tool_call_id == "call-1"
    result = flow.resume(
        suspended,
        ApprovalDecision(request_id="call-1", decision=ApprovalDecisionType.APPROVED, signature="verified"),
        approval_policy=Policy(),
    )

    assert calls == ["ok"]
    assert result.output == "Delivered."
    assert flow.completed_state(suspended, result).status == FlowStateStatus.COMPLETED


def test_eval_runner_checks_output_agent_and_tools():
    agent = Agent(name="a", instructions="Help.", model=FakeModel(["ok"]))
    flow = Flow(name="f", entrypoint=agent, agents=[])

    result = EvalRunner(flow).run_case(EvalCase(name="happy", input="hello", expected_output="ok", expected_agent="a"))

    assert result.passed


def test_eval_runner_enforces_cost_and_latency_gates_and_can_replay():
    agent = Agent(name="a", instructions="Help.", model=FakeModel(["ok", "ok"]))
    runner = EvalRunner(Flow(name="f", entrypoint=agent, agents=[]))
    failed = runner.run_case(EvalCase(name="gate", input="hello", max_latency_ms=-1, max_total_tokens=-1))

    assert not failed.passed
    assert "latency exceeded max_latency_ms" in failed.failures
    assert "total tokens exceeded max_total_tokens" in failed.failures
    assert runner.replay(failed).passed


def test_flow_state_store_and_eval_failure_paths_are_durable_and_descriptive():
    store = InMemoryFlowStateStore()
    state = store.save(PersistedFlowState(flow_instance_id="f"), expected_revision=0)
    assert store.load("f").revision == 1
    with pytest.raises(FlowStateConflictError):
        store.save(state, expected_revision=0)
    runner = EvalRunner(Flow(name="f", entrypoint=Agent(name="a", instructions="x", model=FakeModel(["ok"])), agents=[]))
    result = runner.run_case(EvalCase(name="bad", input="x", expected_output="no", expected_agent="other", expected_tools=["missing"]))
    assert len(result.failures) == 3
    with pytest.raises(ValueError):
        runner.replay(type("Result", (), {"flow_result": None})())
