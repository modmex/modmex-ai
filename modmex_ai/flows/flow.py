from __future__ import annotations

import asyncio
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
        self.entrypoint = entrypoint.name if isinstance(entrypoint, Agent) else entrypoint
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
        trace = Trace()
        run_context = RunContext(input=input, context=context, metadata=metadata or {}, trace=trace)
        agent_name = self._agent_name(starting_agent) if starting_agent is not None else self.entrypoint
        agent_results: list[AgentResult] = []
        output: Any = None
        usage = Usage()
        active_provider_state = provider_state
        history_items = session.get_items() if session else []
        current_items = self._items_for_input(input)
        input_items = [*history_items, *current_items]
        conversation_items = input_items.copy()
        agent_input = self._initial_agent_input(
            input=input,
            current_items=current_items,
            input_items=input_items,
            session=session,
            provider_state=provider_state,
        )

        for handoff_index in range(self.max_handoffs + 1):
            agent = self._agent(agent_name)
            try:
                result = agent.run(
                    agent_input,
                    model=self.model,
                    context=run_context,
                    provider_state=active_provider_state,
                    run_input_guardrails=handoff_index == 0,
                )
            except ApprovalRequired as error:
                raise self._suspended_error(
                    error=error,
                    agent_name=agent.name,
                    input_items=input_items,
                    session=session,
                    provider_state=active_provider_state,
                    flow_instance_id=flow_instance_id,
                    idempotency_key=idempotency_key,
                    metadata=state_metadata,
                    ttl=ttl,
                ) from error
            agent_results.append(result)
            usage.add(result.usage)
            if result.provider_state is not None:
                active_provider_state = result.provider_state
            output = result.output

            if not result.handoff_target:
                break
            handoff = self._handoff_for_result(agent, result)
            conversation_items.extend(result.items)
            run_context.state["handoff"] = {
                "from": agent.name,
                "to": result.handoff_target,
                "input": serialize(result.handoff_input),
            }
            trace.add("handoff", agent.name, to=result.handoff_target, index=handoff_index)
            agent_name = result.handoff_target
            agent_input = handoff.filter_history(conversation_items, result.handoff_input)
        else:
            raise MaxHandoffsExceeded(f"Flow {self.name!r} exceeded max_handoffs")

        events = self.emit(output, agent_results) if self.emit else []
        generated_items = [
            item
            for agent_result in agent_results
            for item in agent_result.items
        ]
        output_items = [
            *generated_items,
            *([self._item_for_output(output)] if output is not None else []),
        ]
        if session:
            session.add_items([*current_items, *output_items])
        return FlowResult(
            output=output,
            last_agent=agent,
            last_agent_name=agent.name,
            continuation=FlowContinuation(
                agent_name=agent.name,
                provider_state=active_provider_state,
            ),
            agent_results=agent_results,
            events=events,
            usage=usage,
            provider_state=active_provider_state,
            input_items=input_items,
            output_items=output_items,
            trace=trace,
        )

    async def run_async(self, input: Any, **kwargs: Any) -> FlowResult:
        """Run natively when an active agent exposes ``acomplete``."""
        starting_agent = kwargs.get("starting_agent")
        agent_name = self._agent_name(starting_agent) if starting_agent is not None else self.entrypoint
        first_agent = self._agent(agent_name)
        if not callable(getattr(self.model or first_agent.model, "acomplete", None)):
            return await asyncio.to_thread(self.run, input, **kwargs)
        context_value = kwargs.get("context")
        metadata = kwargs.get("metadata")
        session = kwargs.get("session")
        provider_state = kwargs.get("provider_state")
        trace = Trace()
        run_context = RunContext(input=input, context=context_value, metadata=metadata or {}, trace=trace)
        agent_results: list[AgentResult] = []
        usage = Usage()
        active_provider_state = provider_state
        history_items = session.get_items() if session else []
        current_items = self._items_for_input(input)
        input_items = [*history_items, *current_items]
        conversation_items = input_items.copy()
        agent_input = self._initial_agent_input(input=input, current_items=current_items, input_items=input_items, session=session, provider_state=provider_state)
        output: Any = None
        for handoff_index in range(self.max_handoffs + 1):
            agent = self._agent(agent_name)
            try:
                result = await agent.run_async(
                    agent_input,
                    model=self.model,
                    context=run_context,
                    provider_state=active_provider_state,
                    run_input_guardrails=handoff_index == 0,
                )
            except ApprovalRequired as error:
                raise self._suspended_error(
                    error=error,
                    agent_name=agent.name,
                    input_items=input_items,
                    session=session,
                    provider_state=active_provider_state,
                    flow_instance_id=kwargs.get("flow_instance_id"),
                    idempotency_key=kwargs.get("idempotency_key"),
                    metadata=kwargs.get("state_metadata"),
                    ttl=kwargs.get("ttl"),
                ) from error
            agent_results.append(result)
            usage.add(result.usage)
            if result.provider_state is not None:
                active_provider_state = result.provider_state
            output = result.output
            if not result.handoff_target:
                break
            handoff = self._handoff_for_result(agent, result)
            conversation_items.extend(result.items)
            run_context.state["handoff"] = {"from": agent.name, "to": result.handoff_target, "input": serialize(result.handoff_input)}
            trace.add("handoff", agent.name, to=result.handoff_target, index=handoff_index)
            agent_name = result.handoff_target
            agent_input = handoff.filter_history(conversation_items, result.handoff_input)
        else:
            raise MaxHandoffsExceeded(f"Flow {self.name!r} exceeded max_handoffs")
        events = self.emit(output, agent_results) if self.emit else []
        generated_items = [item for agent_result in agent_results for item in agent_result.items]
        output_items = [*generated_items, *([self._item_for_output(output)] if output is not None else [])]
        if session:
            session.add_items([*current_items, *output_items])
        return FlowResult(
            output=output,
            last_agent=agent,
            last_agent_name=agent.name,
            continuation=FlowContinuation(agent_name=agent.name, provider_state=active_provider_state),
            agent_results=agent_results,
            events=events,
            usage=usage,
            provider_state=active_provider_state,
            input_items=input_items,
            output_items=output_items,
            trace=trace,
        )

    def run_stream(
        self,
        input: Any,
        *,
        starting_agent: str | Agent | None = None,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
        provider_state: ProviderState | None = None,
    ):
        """Stream active-agent events and continue transparently through handoffs."""
        trace = Trace()
        run_context = RunContext(input=input, context=context, metadata=metadata or {}, trace=trace)
        agent_name = self._agent_name(starting_agent) if starting_agent is not None else self.entrypoint
        agent_results: list[AgentResult] = []
        usage = Usage()
        active_provider_state = provider_state
        history_items = session.get_items() if session else []
        current_items = self._items_for_input(input)
        input_items = [*history_items, *current_items]
        conversation_items = input_items.copy()
        agent_input = self._initial_agent_input(
            input=input,
            current_items=current_items,
            input_items=input_items,
            session=session,
            provider_state=provider_state,
        )
        output: Any = None
        for handoff_index in range(self.max_handoffs + 1):
            agent = self._agent(agent_name)
            result: AgentResult | None = None
            for event in agent.run_stream(
                agent_input,
                model=self.model,
                context=run_context,
                provider_state=active_provider_state,
                run_input_guardrails=handoff_index == 0,
            ):
                yield FlowStreamEvent(
                    type=FlowStreamEventType.AGENT,
                    agent_name=agent.name,
                    agent_event=event,
                )
                if event.result is not None:
                    result = event.result
            if result is None:
                raise RuntimeError("Agent stream ended without an AgentResult")
            agent_results.append(result)
            usage.add(result.usage)
            active_provider_state = result.provider_state or active_provider_state
            output = result.output
            if not result.handoff_target:
                break
            handoff = self._handoff_for_result(agent, result)
            conversation_items.extend(result.items)
            run_context.state["handoff"] = {
                "from": agent.name,
                "to": result.handoff_target,
                "input": serialize(result.handoff_input),
            }
            trace.add("handoff", agent.name, to=result.handoff_target, index=handoff_index)
            agent_name = result.handoff_target
            agent_input = handoff.filter_history(conversation_items, result.handoff_input)
        else:
            raise MaxHandoffsExceeded(f"Flow {self.name!r} exceeded max_handoffs")
        events = self.emit(output, agent_results) if self.emit else []
        generated_items = [item for agent_result in agent_results for item in agent_result.items]
        output_items = [*generated_items, *([self._item_for_output(output)] if output is not None else [])]
        if session:
            session.add_items([*current_items, *output_items])
        yield FlowStreamEvent(
            type=FlowStreamEventType.COMPLETED,
            result=FlowResult(
                output=output,
                last_agent=agent,
                last_agent_name=agent.name,
                continuation=FlowContinuation(agent_name=agent.name, provider_state=active_provider_state),
                agent_results=agent_results,
                events=events,
                usage=usage,
                provider_state=active_provider_state,
                input_items=input_items,
                output_items=output_items,
                trace=trace,
            ),
        )

    async def arun_stream(
        self,
        input: Any,
        *,
        starting_agent: str | Agent | None = None,
        context: Any = None,
        metadata: dict[str, Any] | None = None,
        session: Session | None = None,
        provider_state: ProviderState | None = None,
    ):
        """Native async streaming across agents and handoffs."""
        trace = Trace()
        run_context = RunContext(input=input, context=context, metadata=metadata or {}, trace=trace)
        agent_name = self._agent_name(starting_agent) if starting_agent is not None else self.entrypoint
        agent_results: list[AgentResult] = []
        usage = Usage()
        active_provider_state = provider_state
        history_items = session.get_items() if session else []
        current_items = self._items_for_input(input)
        input_items = [*history_items, *current_items]
        conversation_items = input_items.copy()
        agent_input = self._initial_agent_input(input=input, current_items=current_items, input_items=input_items, session=session, provider_state=provider_state)
        output: Any = None
        for handoff_index in range(self.max_handoffs + 1):
            agent = self._agent(agent_name)
            result: AgentResult | None = None
            async for event in agent.arun_stream(
                agent_input,
                model=self.model,
                context=run_context,
                provider_state=active_provider_state,
                run_input_guardrails=handoff_index == 0,
            ):
                yield FlowStreamEvent(type=FlowStreamEventType.AGENT, agent_name=agent.name, agent_event=event)
                if event.result is not None:
                    result = event.result
            if result is None:
                raise RuntimeError("Agent stream ended without an AgentResult")
            agent_results.append(result)
            usage.add(result.usage)
            active_provider_state = result.provider_state or active_provider_state
            output = result.output
            if not result.handoff_target:
                break
            handoff = self._handoff_for_result(agent, result)
            conversation_items.extend(result.items)
            run_context.state["handoff"] = {"from": agent.name, "to": result.handoff_target, "input": serialize(result.handoff_input)}
            trace.add("handoff", agent.name, to=result.handoff_target, index=handoff_index)
            agent_name = result.handoff_target
            agent_input = handoff.filter_history(conversation_items, result.handoff_input)
        else:
            raise MaxHandoffsExceeded(f"Flow {self.name!r} exceeded max_handoffs")
        events = self.emit(output, agent_results) if self.emit else []
        generated_items = [item for agent_result in agent_results for item in agent_result.items]
        output_items = [*generated_items, *([self._item_for_output(output)] if output is not None else [])]
        if session:
            session.add_items([*current_items, *output_items])
        yield FlowStreamEvent(type=FlowStreamEventType.COMPLETED, result=FlowResult(
            output=output, last_agent=agent, last_agent_name=agent.name,
            continuation=FlowContinuation(agent_name=agent.name, provider_state=active_provider_state),
            agent_results=agent_results, events=events, usage=usage,
            provider_state=active_provider_state, input_items=input_items,
            output_items=output_items, trace=trace,
        ))

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
        suspension = self._require_suspension(state)
        if decision.request_id != suspension.approval_request.request_id:
            raise ValueError("Approval decision does not match the suspended tool call.")
        agent = self._agent(suspension.active_agent_name)
        run_context = RunContext(input=None, context=context, metadata=metadata or {}, trace=Trace())
        run_context.state["approval_policy"] = approval_policy
        run_context.state["approval_decisions"] = {decision.request_id: decision}
        result = ToolExecutor(agent.tools).execute(suspension.pending_tool_call, context=run_context)
        output_item = SessionItem(
            type="function_call_output",
            tool_call_id=result.tool_call_id,
            name=result.name,
            output=result.output,
        )
        session = self._session_for_state(state)
        return self.run(
            [output_item],
            starting_agent=agent.name,
            context=context,
            metadata=metadata,
            session=session,
            provider_state=suspension.provider_state,
            flow_instance_id=state.flow_instance_id,
            idempotency_key=state.idempotency_key,
            state_metadata=state.metadata,
            ttl=state.ttl,
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
        suspension = self._require_suspension(state)
        if decision.request_id != suspension.approval_request.request_id:
            raise ValueError("Approval decision does not match the suspended tool call.")
        agent = self._agent(suspension.active_agent_name)
        run_context = RunContext(input=None, context=context, metadata=metadata or {}, trace=Trace())
        run_context.state["approval_policy"] = approval_policy
        run_context.state["approval_decisions"] = {decision.request_id: decision}
        result = await ToolExecutor(agent.tools).execute_async(suspension.pending_tool_call, context=run_context)
        output_item = SessionItem(
            type="function_call_output",
            tool_call_id=result.tool_call_id,
            name=result.name,
            output=result.output,
        )
        session = self._session_for_state(state)
        return await self.run_async(
            [output_item],
            starting_agent=agent.name,
            context=context,
            metadata=metadata,
            session=session,
            provider_state=suspension.provider_state,
            flow_instance_id=state.flow_instance_id,
            idempotency_key=state.idempotency_key,
            state_metadata=state.metadata,
            ttl=state.ttl,
        )

    @staticmethod
    def completed_state(
        state: PersistedFlowState,
        result: FlowResult,
    ) -> PersistedFlowState:
        """Build the durable terminal checkpoint a host should save after ``resume``."""
        snapshot = state.session_snapshot
        if snapshot is not None:
            snapshot = SessionSnapshot(
                **{
                    **snapshot.model_dump(),
                    "items": [*snapshot.items, *result.output_items],
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
            raise UnknownAgentError(f"Agent {agent.name!r} returned an unnamed handoff.")
        handoff = next(
            (candidate for candidate in agent.handoffs if candidate.name == result.handoff_name),
            None,
        )
        if handoff is None:
            raise UnknownAgentError(f"Agent {agent.name!r} returned an unknown handoff.")
        return handoff

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
        if isinstance(input, list) and all(isinstance(item, (Message, SessionItem, dict)) for item in input):
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
        snapshot = SessionSnapshot(
            session_id=session.id if session else flow_instance_id or self.name,
            items=[*input_items, pending_call],
        )
        continuation = FlowContinuation(agent_name=agent_name, provider_state=provider_state)
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
            raise ValueError("Suspended Flow state does not contain a session snapshot.")
        return state.session_snapshot.to_memory_session()
