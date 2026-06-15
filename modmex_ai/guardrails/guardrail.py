from __future__ import annotations

from typing import Any, Protocol, TypeVar

from modmex_ai.errors import (
    GuardrailTriggered,
    InputGuardrailTriggered,
    OutputGuardrailTriggered,
    ToolInputGuardrailTriggered,
    ToolOutputGuardrailTriggered,
)
from modmex_ai.guardrails.result import GuardrailResult


GuardrailValue = TypeVar("GuardrailValue")


class Guardrail(Protocol):
    name: str

    def check(self, value: Any, context: Any = None) -> GuardrailResult:
        ...


InputGuardrail = Guardrail
OutputGuardrail = Guardrail
ToolInputGuardrail = Guardrail
ToolOutputGuardrail = Guardrail


def enforce_guardrails(
    value: GuardrailValue,
    guardrails: list[Guardrail],
    *,
    context: Any = None,
    error_type: type[GuardrailTriggered] = GuardrailTriggered,
) -> None:
    for guardrail in guardrails:
        result = guardrail.check(value, context=context)
        _trace_guardrail(context, guardrail.name, result)
        if not result.passed:
            raise error_type(result.reason or f"Guardrail {guardrail.name!r} triggered")


def enforce_input_guardrails(
    value: Any,
    guardrails: list[InputGuardrail],
    *,
    context: Any = None,
) -> None:
    enforce_guardrails(value, guardrails, context=context, error_type=InputGuardrailTriggered)


def enforce_output_guardrails(
    value: Any,
    guardrails: list[OutputGuardrail],
    *,
    context: Any = None,
) -> None:
    enforce_guardrails(value, guardrails, context=context, error_type=OutputGuardrailTriggered)


def enforce_tool_input_guardrails(
    value: Any,
    guardrails: list[ToolInputGuardrail],
    *,
    context: Any = None,
) -> None:
    enforce_guardrails(value, guardrails, context=context, error_type=ToolInputGuardrailTriggered)


def enforce_tool_output_guardrails(
    value: Any,
    guardrails: list[ToolOutputGuardrail],
    *,
    context: Any = None,
) -> None:
    enforce_guardrails(value, guardrails, context=context, error_type=ToolOutputGuardrailTriggered)


def _trace_guardrail(context: Any, name: str, result: GuardrailResult) -> None:
    trace = getattr(context, "trace", None)
    if trace is not None:
        trace.add(
            "guardrail",
            name,
            passed=result.passed,
            reason=result.reason,
        )
