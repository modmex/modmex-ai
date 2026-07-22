from __future__ import annotations

import asyncio
import inspect
import json
import types
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

from modmex import BaseModel

from modmex_ai.approvals import ApprovalDecision, ApprovalDecisionType, ApprovalPolicy, ApprovalRequest
from modmex_ai.errors import ApprovalRejected, ApprovalRequired, GuardrailTriggered, ToolExecutionError, ToolValidationError
from modmex_ai.guardrails import (
    ToolInputGuardrail,
    ToolOutputGuardrail,
    enforce_tool_input_guardrails,
    enforce_tool_output_guardrails,
)
from modmex_ai.schemas import dumps, function_schema, schema_for_type, validate_tool_args


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    output: Any

    def to_message_content(self) -> str:
        return dumps(self.output)


class Tool:
    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        input_guardrails: list[ToolInputGuardrail] | None = None,
        output_guardrails: list[ToolOutputGuardrail] | None = None,
        requires_approval: bool = False,
        approval_reason: str | None = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.requires_approval = requires_approval
        self.approval_reason = approval_reason

    def schema(self) -> dict[str, Any]:
        parameters = {}
        required = []
        for name, parameter in self.signature.parameters.items():
            if name in ("self", "context", "ctx"):
                continue
            annotation = self.type_hints.get(name, parameter.annotation)
            parameters[name] = schema_for_type(annotation)
            if parameter.default is inspect.Signature.empty:
                required.append(name)
        return function_schema(self.name, self.description, parameters, required)

    def run(self, arguments: dict[str, Any] | str, *, context: Any = None) -> Any:
        try:
            kwargs = self._prepare_arguments(arguments, context=context)
            self._require_approval(kwargs, context=context)
            output = self.func(**kwargs)
            if inspect.isawaitable(output):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    output = asyncio.run(output)
                else:
                    if inspect.iscoroutine(output):
                        output.close()
                    raise ToolExecutionError(
                        f"Tool {self.name!r} is async; use Tool.arun from an async runtime"
                    )
            enforce_tool_output_guardrails(output, self.output_guardrails, context=context)
            return output
        except GuardrailTriggered:
            raise
        except (ApprovalRequired, ApprovalRejected):
            raise
        except ToolValidationError:
            raise
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool {self.name!r} failed: {exc}") from exc

    async def arun(self, arguments: dict[str, Any] | str, *, context: Any = None) -> Any:
        try:
            kwargs = self._prepare_arguments(arguments, context=context)
            self._require_approval(kwargs, context=context)
            output = self.func(**kwargs)
            if inspect.isawaitable(output):
                output = await output
            enforce_tool_output_guardrails(output, self.output_guardrails, context=context)
            return output
        except GuardrailTriggered:
            raise
        except (ApprovalRequired, ApprovalRejected):
            raise
        except ToolValidationError:
            raise
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool {self.name!r} failed: {exc}") from exc

    def _require_approval(self, arguments: dict[str, Any], *, context: Any) -> None:
        if not self.requires_approval:
            return
        tool_call_id = getattr(context, "state", {}).get("current_tool_call_id", self.name)
        request = ApprovalRequest(
            request_id=str(tool_call_id),
            tool_name=self.name,
            arguments=arguments,
            reason=self.approval_reason,
        )
        decision = getattr(context, "state", {}).get("approval_decisions", {}).get(request.request_id)
        policy = getattr(context, "state", {}).get("approval_policy")
        if not isinstance(decision, ApprovalDecision):
            raise ApprovalRequired(request)
        if decision.decision != ApprovalDecisionType.APPROVED:
            raise ApprovalRejected(f"Approval rejected for tool {self.name!r}")
        if not isinstance(policy, ApprovalPolicy) or not policy.verify(request, decision):
            raise ApprovalRejected(f"Approval could not be verified for tool {self.name!r}")

    def _prepare_arguments(
        self,
        arguments: dict[str, Any] | str,
        *,
        context: Any,
    ) -> dict[str, Any]:
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        kwargs = self._coerce_arguments(arguments)
        enforce_tool_input_guardrails(kwargs, self.input_guardrails, context=context)
        if "context" in self.signature.parameters:
            kwargs["context"] = context
        if "ctx" in self.signature.parameters:
            kwargs["ctx"] = context
        return kwargs

    def _coerce_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        expected = {
            name
            for name in self.signature.parameters
            if name not in ("self", "context", "ctx")
        }
        unexpected = set(arguments) - expected
        if unexpected:
            raise ToolValidationError(
                f"Unexpected tool arguments: {', '.join(sorted(unexpected))}"
            )
        kwargs: dict[str, Any] = {}
        for name, parameter in self.signature.parameters.items():
            if name in ("self", "context", "ctx"):
                continue
            if name not in arguments:
                if parameter.default is inspect.Signature.empty:
                    raise ToolValidationError(f"Missing tool argument: {name}")
                kwargs[name] = parameter.default
                continue
            annotation = self.type_hints.get(name, parameter.annotation)
            if (
                arguments[name] is None
                and parameter.default is not inspect.Signature.empty
                and not _accepts_none(annotation)
            ):
                kwargs[name] = parameter.default
                continue
            kwargs[name] = _coerce(arguments[name], annotation)
        return kwargs


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    input_guardrails: list[ToolInputGuardrail] | None = None,
    output_guardrails: list[ToolOutputGuardrail] | None = None,
    requires_approval: bool = False,
    approval_reason: str | None = None,
):
    def decorator(inner: Callable[..., Any]) -> Tool:
        return Tool(
            inner,
            name=name,
            description=description,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            requires_approval=requires_approval,
            approval_reason=approval_reason,
        )

    if func is None:
        return decorator
    return decorator(func)


def as_tool(value: Tool | Callable[..., Any]) -> Tool:
    if isinstance(value, Tool):
        return value
    return Tool(value)


def _coerce(value: Any, annotation: Any) -> Any:
    if annotation is inspect.Signature.empty or annotation is Any or value is None:
        return value
    try:
        if (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
        ):
            return validate_tool_args(value, annotation)
        if annotation in (str, int, float, bool):
            return annotation(value)
        return value
    except Exception as exc:
        raise ToolValidationError(str(exc)) from exc


def _accepts_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in (Union, types.UnionType) and type(None) in get_args(annotation)
