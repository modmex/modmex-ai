from __future__ import annotations

from collections.abc import Callable
from typing import Any

from modmex_ai.agents import Agent, AgentResult, Handoff, RunContext
from modmex_ai.approvals import ApprovalDecision, ApprovalPolicy
from modmex_ai.errors import ApprovalRequired, MaxHandoffsExceeded, UnknownAgentError
from modmex_ai.flows.continuation import FlowContinuation
from modmex_ai.flows.state import (
    FlowStateStatus,
    FlowSuspended,
    FlowSuspension,
    PersistedFlowState,
)
from modmex_ai.messages import Message
from modmex_ai.models import ModelClient, ProviderState, Usage
from modmex_ai.schemas import dumps, serialize
from modmex_ai.sessions import Session, SessionItem, SessionSnapshot
from modmex_ai.tools import ToolExecutor
from modmex_ai.flows.result import FlowResult
from modmex_ai.flows.stream import FlowStreamEvent, FlowStreamEventType
from modmex_ai.tracing import Trace


EmitFn = Callable[[Any, list[AgentResult]], list[dict[str, Any]]]


class Flow:
    def __init__(
        self,
        *,
        name: str,
        entrypoint: str | Agent,
        agents: list[Agent],
        model: ModelClient | None = None,
        emit: EmitFn | None = None,
        max_handoffs: int = 4,
    ) -> None:
        self.name = name
        self.entrypoint = (
            entrypoint.name if isinstance(entrypoint, Agent) else entrypoint
        )
        self.agents = {agent.name: agent for agent in agents}
        if isinstance(entrypoint, Agent):
            self.agents[entrypoint.name] = entrypoint
        self.model = model
        self.emit = emit
        self.max_handoffs = max_handoffs

    def run(
        self,
        input: Any,
        *,
        starting_agent: str | Agent | None = None,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
        provider_state: ProviderState | None = None,
        flow_instance_id: str | None = None,
        idempotency_key: str | None = None,
        state_metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> FlowResult:
        state = _FlowRun(self, input, **_run_options(locals()))
        for agent in state.agents():
            try:
                result = agent.run(state.agent_input, **state.agent_kwargs())
            except ApprovalRequired as error:
                raise state.suspended(error, agent) from error
            if state.accept(agent, result):
                break
        return state.finish()

    async def run_async(
        self,
        input: Any,
        *,
        starting_agent: str | Agent | None = None,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
        provider_state: ProviderState | None = None,
        flow_instance_id: str | None = None,
        idempotency_key: str | None = None,
        state_metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> FlowResult:
        """Run natively when the active model exposes ``acomplete``."""
        options = _run_options(locals())
        first_agent = self._agent(
            self._agent_name(starting_agent) if starting_agent is not None else self.entrypoint
        )
        if not callable(getattr(self.model or first_agent.model, "acomplete", None)):
            import asyncio

            return await asyncio.to_thread(lambda: self.run(input, **options))
        state = _FlowRun(self, input, **options)
        for agent in state.agents():
            try:
                result = await agent.run_async(state.agent_input, **state.agent_kwargs())
            except ApprovalRequired as error:
                raise state.suspended(error, agent) from error
            if state.accept(agent, result):
                break
        return state.finish()

    async def arun(self, input: Any, **kwargs: Any) -> FlowResult:
        """Async run; same as :meth:`run_async`, named like ``arun_stream``."""
        return await self.run_async(input, **kwargs)

    def run_stream(
        self,
        input: Any,
        *,
        starting_agent: str | Agent | None = None,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
        provider_state: ProviderState | None = None,
        flow_instance_id: str | None = None,
        idempotency_key: str | None = None,
        state_metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ):
        """Stream active-agent events and continue transparently through handoffs."""
        state = _FlowRun(self, input, **_run_options(locals()))
        for agent in state.agents():
            result: AgentResult | None = None
            try:
                for event in agent.run_stream(state.agent_input, **state.agent_kwargs()):
                    yield FlowStreamEvent(
                        type=FlowStreamEventType.AGENT,
                        agent_name=agent.name,
                        agent_event=event,
                    )
                    if event.result is not None:
                        result = event.result
            except ApprovalRequired as error:
                raise state.suspended(error, agent) from error
            if result is None:
                raise RuntimeError("Agent stream ended without an AgentResult")
            if state.accept(agent, result):
                break
        yield FlowStreamEvent(type=FlowStreamEventType.COMPLETED, result=state.finish())

    async def arun_stream(
        self,
        input: Any,
        *,
        starting_agent: str | Agent | None = None,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
        provider_state: ProviderState | None = None,
        flow_instance_id: str | None = None,
        idempotency_key: str | None = None,
        state_metadata: dict[str, Any] | None = None,
        ttl: int | None = None,
    ):
        """Native async streaming across agents and handoffs."""
        state = _FlowRun(self, input, **_run_options(locals()))
        for agent in state.agents():
            result: AgentResult | None = None
            try:
                async for event in agent.arun_stream(
                    state.agent_input, **state.agent_kwargs()
                ):
                    yield FlowStreamEvent(
                        type=FlowStreamEventType.AGENT,
                        agent_name=agent.name,
                        agent_event=event,
                    )
                    if event.result is not None:
                        result = event.result
            except ApprovalRequired as error:
                raise state.suspended(error, agent) from error
            if result is None:
                raise RuntimeError("Agent stream ended without an AgentResult")
            if state.accept(agent, result):
                break
        yield FlowStreamEvent(type=FlowStreamEventType.COMPLETED, result=state.finish())

    def resume(
        self,
        state: PersistedFlowState,
        decision: ApprovalDecision,
        *,
        approval_policy: ApprovalPolicy,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Execute the suspended tool once, then continue from the saved checkpoint."""
        agent, suspension, run_context = self._prepare_resume(
            state, decision, approval_policy=approval_policy, context=context, metadata=metadata
        )
        result = ToolExecutor(agent.tools, error_policy=agent.tool_error_policy).execute(
            suspension.pending_tool_call, context=run_context
        )
        return self.run(
            self._resume_input(suspension, result),
            **self._resume_options(state, agent, context=context, metadata=metadata),
        )

    async def resume_async(
        self,
        state: PersistedFlowState,
        decision: ApprovalDecision,
        *,
        approval_policy: ApprovalPolicy,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> FlowResult:
        """Async counterpart of :meth:`resume` for async tools and model clients."""
        agent, suspension, run_context = self._prepare_resume(
            state, decision, approval_policy=approval_policy, context=context, metadata=metadata
        )
        result = await ToolExecutor(
            agent.tools, error_policy=agent.tool_error_policy
        ).execute_async(suspension.pending_tool_call, context=run_context)
        return await self.run_async(
            self._resume_input(suspension, result),
            **self._resume_options(state, agent, context=context, metadata=metadata),
        )

    def _prepare_resume(
        self,
        state: PersistedFlowState,
        decision: ApprovalDecision,
        *,
        approval_policy: ApprovalPolicy,
        context: Any,
        metadata: dict[str, Any] | None,
    ) -> tuple[Agent, FlowSuspension, RunContext]:
        suspension = self._require_suspension(state)
        if decision.request_id != suspension.approval_request.request_id:
            raise ValueError("Approval decision does not match the suspended tool call.")
        agent = self._agent(suspension.active_agent_name)
        run_context = RunContext(
            input=None, context=context, metadata=metadata or {}, trace=Trace()
        )
        if agent.tool_selector is not None:
            names = [tool.name for tool in agent.tools] + [
                handoff.name for handoff in agent.handoffs
            ]
            selected = agent.tool_selector(run_context, list(names))
            if not isinstance(selected, (list, tuple)) or any(
                not isinstance(name, str) for name in selected
            ):
                raise ValueError("tool_selector must return registered tool names")
            if len(set(selected)) != len(selected) or not set(selected).issubset(names):
                raise ValueError("tool_selector returned duplicate or unregistered names")
            if suspension.pending_tool_call.name not in selected:
                raise ValueError("Suspended tool is no longer available")
        run_context.state["approval_policy"] = approval_policy
        run_context.state["approval_decisions"] = {decision.request_id: decision}
        return agent, suspension, run_context

    @staticmethod
    def _resume_input(suspension: FlowSuspension, result: Any) -> list[SessionItem]:
        output_item = SessionItem(
            type="function_call_output",
            tool_call_id=result.tool_call_id,
            name=result.name,
            output=result.output,
        )
        if suspension.provider_state and suspension.provider_state.has_remote_state:
            return [*suspension.pending_items, output_item]
        return [output_item]

    def _resume_options(
        self,
        state: PersistedFlowState,
        agent: Agent,
        *,
        context: Any,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        suspension = state.suspension
        assert suspension is not None
        return {
            "starting_agent": agent.name,
            "context": context,
            "metadata": metadata,
            "session": self._session_for_state(state),
            "provider_state": suspension.provider_state,
            "flow_instance_id": state.flow_instance_id,
            "idempotency_key": state.idempotency_key,
            "state_metadata": state.metadata,
            "ttl": state.ttl,
        }

    @staticmethod
    def completed_state(
        state: PersistedFlowState,
        result: FlowResult,
    ) -> PersistedFlowState:
        """Build the durable terminal checkpoint a host should save after ``resume``."""
        snapshot = state.session_snapshot
        if snapshot is not None:
            # The resumed turn's inputs (the approved tool's output) are not output
            # items; without them the saved history has a call with no result.
            new_inputs = [
                item
                for item in result.input_items[len(snapshot.items) :]
                if item not in snapshot.items
            ]
            snapshot = SessionSnapshot(
                **{
                    **snapshot.model_dump(),
                    "items": [*snapshot.items, *new_inputs, *result.output_items],
                }
            )
        return PersistedFlowState(
            **{
                **state.model_dump(),
                "status": FlowStateStatus.COMPLETED,
                "session_snapshot": snapshot,
                "continuation": result.continuation,
                "provider_state": result.provider_state,
                "suspension": None,
            }
        )

    def _agent(self, name: str) -> Agent:
        agent = self.agents.get(name)
        if agent is None:
            raise UnknownAgentError(f"Unknown agent: {name}")
        return agent

    def _agent_name(self, value: str | Agent) -> str:
        name = value.name if isinstance(value, Agent) else value
        self._agent(name)
        return name

    def _handoff_for_result(self, agent: Agent, result: AgentResult) -> Handoff:
        if result.handoff_name is None:
            raise UnknownAgentError(
                f"Agent {agent.name!r} returned an unnamed handoff."
            )
        handoff = next(
            (
                candidate
                for candidate in agent.handoffs
                if candidate.name == result.handoff_name
            ),
            None,
        )
        if handoff is None:
            raise UnknownAgentError(
                f"Agent {agent.name!r} returned an unknown handoff."
            )
        return handoff

    @staticmethod
    def _handoff_input(handoff, conversation_items, result):
        if result.provider_state and result.provider_state.has_remote_state:
            if handoff.input_filter is not None:
                raise ValueError(
                    "Remote handoffs cannot apply local history filters; use stateless mode"
                )
            return [item for item in result.items if item.type == "handoff_call_output"]
        return handoff.filter_history(conversation_items, result.handoff_input)

    @staticmethod
    def _initial_agent_input(
        *,
        input: Any,
        current_items: list[SessionItem],
        input_items: list[SessionItem],
        session: Session | None,
        provider_state: ProviderState | None,
    ) -> Any:
        if provider_state and provider_state.has_remote_state:
            return current_items
        if session:
            return input_items
        return input

    def _items_for_input(self, input: Any) -> list[SessionItem]:
        if isinstance(input, list) and all(
            isinstance(item, (Message, SessionItem, dict)) for item in input
        ):
            return [
                item
                if isinstance(item, SessionItem)
                else SessionItem.from_message(item)
                if isinstance(item, Message)
                else SessionItem(**item)
                if item.get("type")
                else SessionItem.from_message(Message(**item))
                for item in input
            ]
        content = input if isinstance(input, str) else dumps(input)
        return [SessionItem(role="user", content=content)]

    def _item_for_output(self, output: Any) -> SessionItem:
        content = output if isinstance(output, str) else dumps(output)
        return SessionItem(role="assistant", content=content)

    def _suspended_error(
        self,
        *,
        error: ApprovalRequired,
        checkpoint: dict | None = None,
        agent_name: str,
        input_items: list[SessionItem],
        session: Session | None,
        provider_state: ProviderState | None,
        flow_instance_id: str | None,
        idempotency_key: str | None,
        metadata: dict[str, Any] | None,
        ttl: int | None,
    ) -> FlowSuspended:
        request = error.request
        if not hasattr(request, "request_id") or not hasattr(request, "tool_name"):
            raise TypeError("ApprovalRequired must contain an ApprovalRequest.")
        pending_call = SessionItem(
            type="function_call",
            tool_call_id=request.request_id,
            name=request.tool_name,
            arguments=request.arguments,
        )
        if flow_instance_id is None and session is None:
            # Never key a durable state by the flow name: concurrent suspensions of
            # the same flow would overwrite each other.
            from uuid import uuid4

            flow_instance_id = f"{self.name}-{uuid4().hex}"
        snapshot = SessionSnapshot(
            session_id=session.id if session else flow_instance_id,
            items=self._items_for_input(checkpoint["items"][1:])
            if checkpoint
            else [*input_items, pending_call],
        )
        continuation = FlowContinuation(
            agent_name=agent_name, provider_state=provider_state
        )
        state = PersistedFlowState(
            flow_instance_id=flow_instance_id or snapshot.session_id,
            status=FlowStateStatus.SUSPENDED,
            idempotency_key=idempotency_key,
            session_snapshot=snapshot,
            continuation=continuation,
            provider_state=provider_state,
            suspension=FlowSuspension(
                approval_request=request,
                pending_tool_call=self._tool_call_from_request(request),
                active_agent_name=agent_name,
                provider_state=provider_state,
                continuation=continuation,
                pending_items=checkpoint.get("pending", []) if checkpoint else [],
            ),
            metadata=metadata or {},
            ttl=ttl,
        )
        return FlowSuspended(state)

    @staticmethod
    def _tool_call_from_request(request: Any):
        from modmex_ai.models import ToolCall

        return ToolCall(
            tool_call_id=request.request_id,
            name=request.tool_name,
            arguments=request.arguments,
        )

    @staticmethod
    def _require_suspension(state: PersistedFlowState) -> FlowSuspension:
        if state.status != FlowStateStatus.SUSPENDED or state.suspension is None:
            raise ValueError("Flow state is not suspended for approval.")
        return state.suspension

    @staticmethod
    def _session_for_state(state: PersistedFlowState) -> Session:
        if state.session_snapshot is None:
            raise ValueError(
                "Suspended Flow state does not contain a session snapshot."
            )
        return state.session_snapshot.to_memory_session()


_RUN_OPTIONS = (
    "starting_agent",
    "context",
    "metadata",
    "session",
    "provider_state",
    "flow_instance_id",
    "idempotency_key",
    "state_metadata",
    "ttl",
)


def _run_options(scope: dict[str, Any]) -> dict[str, Any]:
    return {name: scope[name] for name in _RUN_OPTIONS}


class _FlowRun:
    """State of one Flow execution, shared by every execution mode.

    The four public entry points (run, run_async, run_stream, arun_stream) only
    differ in how they call the active agent; setup, handoff bookkeeping,
    suspension and the final result live here so they cannot drift apart.
    """

    def __init__(
        self,
        flow: Flow,
        input: Any,
        *,
        starting_agent: str | Agent | None,
        context: Any,
        metadata: dict[str, Any] | None,
        session: Session | None,
        provider_state: ProviderState | None,
        flow_instance_id: str | None,
        idempotency_key: str | None,
        state_metadata: dict[str, Any] | None,
        ttl: int | None,
    ) -> None:
        self.flow = flow
        self.trace = Trace()
        self.run_context = RunContext(
            input=input, context=context, metadata=metadata or {}, trace=self.trace
        )
        self.agent_name = (
            flow._agent_name(starting_agent)
            if starting_agent is not None
            else flow.entrypoint
        )
        self.session = session
        self.flow_instance_id = flow_instance_id
        self.idempotency_key = idempotency_key
        self.state_metadata = state_metadata
        self.ttl = ttl
        self.agent_results: list[AgentResult] = []
        self.usage = Usage()
        self.output: Any = None
        self.provider_state = provider_state
        history_items = session.get_items() if session else []
        self.current_items = flow._items_for_input(input)
        self.input_items = [*history_items, *self.current_items]
        self.conversation_items = self.input_items.copy()
        self.agent_input = flow._initial_agent_input(
            input=input,
            current_items=self.current_items,
            input_items=self.input_items,
            session=session,
            provider_state=provider_state,
        )
        self.handoff_index = 0
        self.agent: Agent | None = None

    def agents(self):
        """Yield the active agent for each turn until one finishes without a handoff."""
        for index in range(self.flow.max_handoffs + 1):
            self.handoff_index = index
            self.agent = self.flow._agent(self.agent_name)
            yield self.agent
        raise MaxHandoffsExceeded(f"Flow {self.flow.name!r} exceeded max_handoffs")

    def agent_kwargs(self) -> dict[str, Any]:
        return {
            "model": self.flow.model,
            "context": self.run_context,
            "provider_state": self.provider_state,
            "run_input_guardrails": self.handoff_index == 0,
        }

    def suspended(self, error: ApprovalRequired, agent: Agent) -> FlowSuspended:
        checkpoint = self.run_context.state.get("__modmex_ai_approval_checkpoint")
        return self.flow._suspended_error(
            error=error,
            agent_name=agent.name,
            input_items=self.input_items,
            session=self.session,
            provider_state=(checkpoint or {}).get("provider_state", self.provider_state),
            checkpoint=checkpoint or None,
            flow_instance_id=self.flow_instance_id,
            idempotency_key=self.idempotency_key,
            metadata=self.state_metadata,
            ttl=self.ttl,
        )

    def accept(self, agent: Agent, result: AgentResult) -> bool:
        """Record a finished agent run; return True when the flow is done."""
        self.agent_results.append(result)
        self.usage.add(result.usage)
        if result.provider_state is not None:
            self.provider_state = result.provider_state
        self.output = result.output
        if not result.handoff_target:
            return True
        handoff = self.flow._handoff_for_result(agent, result)
        self.conversation_items.extend(result.items)
        self.run_context.state["handoff"] = {
            "from": agent.name,
            "to": result.handoff_target,
            "input": serialize(result.handoff_input),
        }
        self.trace.add(
            "handoff", agent.name, to=result.handoff_target, index=self.handoff_index
        )
        self.agent_name = result.handoff_target
        self.agent_input = self.flow._handoff_input(
            handoff, self.conversation_items, result
        )
        return False

    def finish(self) -> FlowResult:
        assert self.agent is not None
        output = self.output
        events = self.flow.emit(output, self.agent_results) if self.flow.emit else []
        generated_items = [
            item for agent_result in self.agent_results for item in agent_result.items
        ]
        output_items = [
            *generated_items,
            *([self.flow._item_for_output(output)] if output is not None else []),
        ]
        if self.session:
            self.session.add_items([*self.current_items, *output_items])
        return FlowResult(
            output=output,
            last_agent=self.agent,
            last_agent_name=self.agent.name,
            continuation=FlowContinuation(
                agent_name=self.agent.name, provider_state=self.provider_state
            ),
            agent_results=self.agent_results,
            events=events,
            usage=self.usage,
            provider_state=self.provider_state,
            input_items=self.input_items,
            output_items=output_items,
            trace=self.trace,
        )
