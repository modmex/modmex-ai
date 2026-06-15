from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modmex_ai.agents.context import RunContext
from modmex_ai.agents.handoff import find_handoff
from modmex_ai.agents.result import AgentResult
from modmex_ai.errors import MaxToolCallsExceeded
from modmex_ai.messages import Message
from modmex_ai.models import ModelClient, ModelRequest, ModelResponse, ProviderState, ToolCall, Usage
from modmex_ai.sessions import SessionItem
from modmex_ai.tools import ToolExecutor, ToolResult

if TYPE_CHECKING:
    from modmex_ai.agents.agent import Agent


@dataclass
class AgentExecutionStep:
    """The next action requested by the provider-neutral agent state machine."""

    result: AgentResult | None = None
    tool_calls: list[ToolCall] | None = None


class AgentExecution:
    """Shared Agent state machine; sync and async drivers supply I/O only."""

    def __init__(
        self,
        *,
        agent: Agent,
        model: ModelClient,
        input: Any,
        context: RunContext,
        provider_state: ProviderState | None,
        run_output_guardrails: bool,
    ) -> None:
        self.agent = agent
        self.model = model
        self.context = context
        self.run_output_guardrails = run_output_guardrails
        self.messages = agent._initial_messages(
            input,
            include_output_schema_prompt=not self.model.profile.supports_structured_output,
        )
        self.executor = ToolExecutor(agent.tools)
        self.tool_schemas = [*self.executor.schemas(), *agent._handoff_schemas()]
        self.output_schema = agent._output_schema()
        self.tool_call_count = 0
        self.output_guardrail_retry_count = 0
        self.usage = Usage()
        self.generated_items: list[SessionItem] = []
        self.provider_state = provider_state

    def next_request(self) -> ModelRequest:
        self.context.trace.add(
            "model_request",
            self.agent.name,
            tools=[tool["name"] for tool in self.tool_schemas],
        )
        return ModelRequest(
            messages=self.messages,
            tools=self.tool_schemas,
            output_schema=self.output_schema,
            output_strict=self.agent.output_strict,
            settings=self.agent.settings,
            provider_state=self.provider_state,
        )

    def accept_response(self, response: ModelResponse) -> AgentExecutionStep:
        if response.provider_state is not None:
            self.provider_state = response.provider_state
        self.usage.add(response.usage)
        self.context.trace.add(
            "model_response",
            self.agent.name,
            request_id=response.request_id,
            tool_calls=[call.name for call in response.tool_calls],
            usage=response.usage,
        )
        if response.tool_calls:
            return self._tool_step(response.tool_calls)

        output = self.agent._parse_output(response.output_text)
        if self.run_output_guardrails and self.agent._should_retry_output(
            output=output,
            response_text=response.output_text,
            messages=self.messages,
            context=self.context,
            retry_count=self.output_guardrail_retry_count,
        ):
            self.output_guardrail_retry_count += 1
            return AgentExecutionStep()
        return AgentExecutionStep(result=self._result(output=output))

    def begin_tool_call(self, tool_call: ToolCall) -> None:
        item = SessionItem(
            type="function_call",
            tool_call_id=tool_call.tool_call_id,
            name=tool_call.name,
            arguments=tool_call.arguments,
        )
        self.generated_items.append(item)
        self.messages.append(item)

    def accept_tool_result(self, result: ToolResult) -> None:
        item = SessionItem(
            type="function_call_output",
            tool_call_id=result.tool_call_id,
            name=result.name,
            output=result.output,
        )
        self.generated_items.append(item)
        nested_usage = self.context.state.pop("__modmex_ai_nested_usage", None)
        if nested_usage:
            self.usage.add(nested_usage)
        self.context.trace.add("tool_call", result.name, tool_call_id=result.tool_call_id)
        self.messages.append(item)

    def _tool_step(self, tool_calls: list[ToolCall]) -> AgentExecutionStep:
        self.tool_call_count += len(tool_calls)
        if self.tool_call_count > self.agent.max_tool_calls:
            raise MaxToolCallsExceeded(f"Agent {self.agent.name!r} exceeded max_tool_calls")
        for tool_call in tool_calls:
            handoff = find_handoff(tool_call, self.agent.handoffs)
            if handoff is None:
                continue
            handoff_input = handoff.invoke(self.context, tool_call.arguments)
            self.generated_items.extend([
                SessionItem(
                    type="handoff_call",
                    tool_call_id=tool_call.tool_call_id,
                    name=handoff.name,
                    arguments=tool_call.arguments,
                ),
                SessionItem(
                    type="handoff_call_output",
                    tool_call_id=tool_call.tool_call_id,
                    name=handoff.name,
                    output={"transferred": True},
                ),
            ])
            self.context.trace.add(
                "handoff_call",
                handoff.name,
                tool_call_id=tool_call.tool_call_id,
                to=handoff.agent,
            )
            return AgentExecutionStep(result=AgentResult(
                output=None,
                agent=self.agent.name,
                handoff_target=handoff.agent,
                handoff_name=handoff.name,
                handoff_input=handoff_input,
                usage=self.usage,
                items=self.generated_items,
                provider_state=self.provider_state,
                trace=self.context.trace,
            ))
        return AgentExecutionStep(tool_calls=tool_calls)

    def _result(self, *, output: Any) -> AgentResult:
        return AgentResult(
            output=output,
            agent=self.agent.name,
            usage=self.usage,
            items=self.generated_items,
            trace=self.context.trace,
            provider_state=self.provider_state,
        )
