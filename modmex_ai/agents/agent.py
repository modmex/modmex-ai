from __future__ import annotations

import asyncio
import json
from typing import Any

from modmex_ai.agents.context import RunContext
from modmex_ai.agents.execution import AgentExecution
from modmex_ai.agents.handoff import Handoff, normalize_handoffs
from modmex_ai.agents.result import AgentResult
from modmex_ai.agents.stream import AgentStreamEvent, AgentStreamEventType
from modmex_ai.errors import OutputGuardrailTriggered, OutputValidationError
from modmex_ai.guardrails import (
    InputGuardrail,
    OutputGuardrail,
    ToolInputGuardrail,
    ToolOutputGuardrail,
    enforce_input_guardrails,
    enforce_output_guardrails,
)
from modmex_ai.messages import Message
from modmex_ai.models import (
    ModelClient,
    ModelResponse,
    ModelSettings,
    ModelStreamEvent,
    ModelStreamEventType,
    ProviderState,
    Usage,
)
from modmex_ai.schemas import dumps, schema_for_model, validate_model
from modmex_ai.sessions import SessionItem
from modmex_ai.tools import Tool, as_tool, tool as create_tool


_NESTED_USAGE_STATE_KEY = "__modmex_ai_nested_usage"


class Agent:
    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        output_type: type[Any] | None = None,
        output_strict: bool = True,
        tools: list[Tool | Any] | None = None,
        handoffs: list[str | Handoff] | None = None,
        model: ModelClient | None = None,
        settings: ModelSettings | None = None,
        max_tool_calls: int = 8,
        input_guardrails: list[InputGuardrail] | None = None,
        output_guardrails: list[OutputGuardrail] | None = None,
        max_output_guardrail_retries: int = 0,
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.output_type = output_type
        self.output_strict = output_strict
        self.tools = [as_tool(value) for value in (tools or [])]
        self.handoffs = normalize_handoffs(handoffs)
        self.model = model
        self.settings = settings
        self.max_tool_calls = max_tool_calls
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        if max_output_guardrail_retries < 0:
            raise ValueError("max_output_guardrail_retries must be non-negative")
        self.max_output_guardrail_retries = max_output_guardrail_retries

    def add_tool(self, value: Tool | Any) -> Tool:
        registered = as_tool(value)
        self.tools.append(registered)
        return registered

    def tool(
        self,
        func: Any = None,
        *,
        name: str | None = None,
        description: str | None = None,
        input_guardrails: list[ToolInputGuardrail] | None = None,
        output_guardrails: list[ToolOutputGuardrail] | None = None,
        requires_approval: bool = False,
        approval_reason: str | None = None,
    ):
        def decorator(inner):
            return self.add_tool(create_tool(
                inner,
                name=name,
                description=description,
                input_guardrails=input_guardrails,
                output_guardrails=output_guardrails,
                requires_approval=requires_approval,
                approval_reason=approval_reason,
            ))
        return decorator if func is None else decorator(func)

    def as_tool(self, *, tool_name: str | None = None, tool_description: str | None = None) -> Tool:
        def run_agent(input: str, ctx=None) -> Any:
            result = self.run(input, context=ctx)
            if isinstance(ctx, RunContext):
                nested_usage = ctx.state.setdefault(_NESTED_USAGE_STATE_KEY, Usage())
                nested_usage.add(result.usage)
            return result.output
        return create_tool(run_agent, name=tool_name or self.name, description=tool_description or self.instructions)

    def run(self, input: Any, **kwargs: Any) -> AgentResult:
        return self._run_execution(self._execution(input, **kwargs))

    async def run_async(self, input: Any, **kwargs: Any) -> AgentResult:
        execution = self._execution(input, **kwargs)
        acomplete = getattr(execution.model, "acomplete", None)
        if not callable(acomplete):
            return await asyncio.to_thread(self._run_execution, execution)
        while True:
            step = execution.accept_response(await acomplete(execution.next_request()))
            if step.result is not None:
                return step.result
            for tool_call in step.tool_calls or []:
                execution.begin_tool_call(tool_call)
                execution.accept_tool_result(await execution.executor.execute_async(tool_call, context=execution.context))

    def run_stream(self, input: Any, **kwargs: Any):
        execution = self._execution(input, **kwargs)
        while True:
            completed: ModelResponse | None = None
            for raw_event in execution.model.stream(execution.next_request()):
                event = self._model_stream_event(raw_event)
                if event.type == ModelStreamEventType.TEXT_DELTA:
                    yield AgentStreamEvent(type=AgentStreamEventType.TEXT_DELTA, text_delta=event.text_delta or "")
                elif event.type == ModelStreamEventType.TOOL_CALL_DELTA:
                    yield AgentStreamEvent(type=AgentStreamEventType.TOOL_CALL_DELTA, tool_call=event.tool_call)
                elif event.type == ModelStreamEventType.COMPLETED:
                    completed = event.response
            if completed is None:
                raise RuntimeError("Model stream ended without a completed response")
            step = execution.accept_response(completed)
            if step.result is not None:
                yield AgentStreamEvent(type=AgentStreamEventType.HANDOFF if step.result.handoff_target else AgentStreamEventType.COMPLETED, result=step.result)
                return
            for tool_call in step.tool_calls or []:
                execution.begin_tool_call(tool_call)
                yield AgentStreamEvent(type=AgentStreamEventType.TOOL_CALL_DELTA, tool_call=tool_call)
                tool_result = execution.executor.execute(tool_call, context=execution.context)
                execution.accept_tool_result(tool_result)
                yield AgentStreamEvent(type=AgentStreamEventType.TOOL_FINISHED, tool_call=tool_call, data={"output": tool_result.output})

    async def arun_stream(self, input: Any, **kwargs: Any):
        """Async counterpart of ``run_stream`` for models with native streaming."""
        execution = self._execution(input, **kwargs)
        astream = getattr(execution.model, "astream", None)
        if not callable(astream):
            raise NotImplementedError("Model does not implement native async streaming")
        while True:
            completed: ModelResponse | None = None
            async for raw_event in astream(execution.next_request()):
                event = self._model_stream_event(raw_event)
                if event.type == ModelStreamEventType.TEXT_DELTA:
                    yield AgentStreamEvent(type=AgentStreamEventType.TEXT_DELTA, text_delta=event.text_delta or "")
                elif event.type == ModelStreamEventType.TOOL_CALL_DELTA:
                    yield AgentStreamEvent(type=AgentStreamEventType.TOOL_CALL_DELTA, tool_call=event.tool_call)
                elif event.type == ModelStreamEventType.COMPLETED:
                    completed = event.response
            if completed is None:
                raise RuntimeError("Model stream ended without a completed response")
            step = execution.accept_response(completed)
            if step.result is not None:
                yield AgentStreamEvent(type=AgentStreamEventType.HANDOFF if step.result.handoff_target else AgentStreamEventType.COMPLETED, result=step.result)
                return
            for tool_call in step.tool_calls or []:
                execution.begin_tool_call(tool_call)
                yield AgentStreamEvent(type=AgentStreamEventType.TOOL_CALL_DELTA, tool_call=tool_call)
                tool_result = await execution.executor.execute_async(tool_call, context=execution.context)
                execution.accept_tool_result(tool_result)
                yield AgentStreamEvent(type=AgentStreamEventType.TOOL_FINISHED, tool_call=tool_call, data={"output": tool_result.output})

    def _run_execution(self, execution: AgentExecution) -> AgentResult:
        while True:
            step = execution.accept_response(execution.model.complete(execution.next_request()))
            if step.result is not None:
                return step.result
            for tool_call in step.tool_calls or []:
                execution.begin_tool_call(tool_call)
                execution.accept_tool_result(execution.executor.execute(tool_call, context=execution.context))

    def _execution(
        self,
        input: Any,
        *,
        model: ModelClient | None = None,
        context: RunContext | Any = None,
        provider_state: ProviderState | None = None,
        run_input_guardrails: bool = True,
        run_output_guardrails: bool = True,
    ) -> AgentExecution:
        active_model = model or self.model
        if active_model is None:
            raise ValueError(f"Agent {self.name!r} has no model")
        run_context = context if isinstance(context, RunContext) else RunContext(input=input, context=context)
        if run_input_guardrails:
            enforce_input_guardrails(input, self.input_guardrails, context=run_context)
        return AgentExecution(agent=self, model=active_model, input=input, context=run_context, provider_state=provider_state, run_output_guardrails=run_output_guardrails)

    @staticmethod
    def _model_stream_event(value: ModelStreamEvent | ModelResponse) -> ModelStreamEvent:
        if isinstance(value, ModelResponse):
            return ModelStreamEvent.completed(value)
        if isinstance(value, ModelStreamEvent):
            return value
        raise TypeError("Model stream must yield ModelStreamEvent or ModelResponse")

    def _output_schema(self) -> dict[str, Any] | None:
        return schema_for_model(self.output_type) if self.output_type else None

    def _should_retry_output(self, *, output: Any, response_text: str | None, messages: list[Message | SessionItem], context: RunContext, retry_count: int) -> bool:
        try:
            enforce_output_guardrails(output, self.output_guardrails, context=context)
        except OutputGuardrailTriggered as error:
            if retry_count >= self.max_output_guardrail_retries:
                raise
            context.trace.add("guardrail", f"{self.name}_output_retry", attempt=retry_count + 1, reason=str(error))
            messages.extend([
                Message(role="assistant", content=response_text or ""),
                Message(role="developer", content="The previous candidate output was rejected by an output guardrail: " f"{error}. Produce a corrected replacement that fully satisfies the output schema and the guardrail rule."),
            ])
            return True
        return False

    def _initial_messages(self, input: Any, *, include_output_schema_prompt: bool = True) -> list[Message | SessionItem]:
        instructions = self.instructions
        if self.output_type and include_output_schema_prompt:
            instructions += "\n\nReturn only JSON that matches this schema:\n" + json.dumps(schema_for_model(self.output_type), separators=(",", ":"))
        return [Message(role="system", content=instructions), *self._input_messages(input)]

    def _input_messages(self, input: Any) -> list[Message | SessionItem]:
        if isinstance(input, list) and all(isinstance(item, (Message, SessionItem, dict)) for item in input):
            return [item if isinstance(item, (Message, SessionItem)) else SessionItem(**item) if item.get("type") else Message(**item) for item in input]
        return [Message(role="user", content=input if isinstance(input, str) else dumps(input))]

    def _parse_output(self, text: str | None) -> Any:
        if self.output_type is None:
            return text or ""
        if text is None:
            raise OutputValidationError(f"Agent {self.name!r} returned no output")
        return validate_model(json.loads(text), self.output_type)

    def _handoff_schemas(self) -> list[dict[str, Any]]:
        return [handoff.schema() for handoff in self.handoffs]
