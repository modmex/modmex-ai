from __future__ import annotations

import json
from base64 import b64encode
from copy import deepcopy
from collections.abc import AsyncIterable, Iterable
from typing import Any

from modmex_ai.messages import FileInput, ImageInput, Message, TextInput
from modmex_ai.models import (
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventType,
    ProviderState,
    ToolCall,
    Usage,
)
from modmex_ai.schemas import dumps
from modmex_ai.sessions import SessionItem


def to_responses_payload(request: ModelRequest, model: str) -> dict[str, Any]:
    instructions, input_items = _split_instructions(request.messages)
    input_items = _input_items_for_provider_state(input_items, request.provider_state)
    payload: dict[str, Any] = {
        "model": request.model or model,
        "input": input_items,
    }
    _apply_provider_state(payload, request.provider_state)
    if instructions:
        payload["instructions"] = instructions
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                **tool,
                "parameters": _tool_parameters(request, tool),
                "strict": request.tool_strict,
            }
            for tool in request.tools
        ]
    if request.output_schema:
        schema = (
            to_openai_strict_schema(request.output_schema)
            if request.output_strict
            else deepcopy(request.output_schema)
        )
        payload["text"] = {
            "format": {
                "type": "json_schema",
                "name": request.output_schema.get("title", "output"),
                "strict": request.output_strict,
                "schema": schema,
            }
        }
    _apply_responses_settings(payload, request)
    return payload


def from_responses_payload(payload: dict[str, Any], *, headers: dict[str, str], status_code: int, model: str) -> ModelResponse:
    tool_calls: list[ToolCall] = []
    output_text_parts: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") == "function_call":
            tool_calls.append(
                ToolCall(
                    tool_call_id=item.get("call_id") or item.get("id") or "",
                    name=item.get("name", ""),
                    arguments=item.get("arguments", "{}"),
                )
            )
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    output_text_parts.append(content.get("text", ""))
    return ModelResponse(
        output_text="".join(output_text_parts) or payload.get("output_text"),
        tool_calls=tool_calls,
        raw=payload,
        usage=_usage_from_payload(payload),
        request_id=headers.get("x-request-id") or payload.get("id"),
        status_code=status_code,
        headers=headers,
        provider="openai",
        model=model,
        provider_state=_provider_state_from_response(payload),
    )


def to_chat_payload(request: ModelRequest, model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model or model,
        "messages": [_message_payload(message) for message in request.messages],
    }
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    **tool,
                    "parameters": _tool_parameters(request, tool),
                    "strict": request.tool_strict,
                },
            }
            for tool in request.tools
        ]
    if request.output_schema:
        schema = (
            to_openai_strict_schema(request.output_schema)
            if request.output_strict
            else deepcopy(request.output_schema)
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": request.output_schema.get("title", "output"),
                "strict": request.output_strict,
                "schema": schema,
            },
        }
    _apply_chat_settings(payload, request)
    return payload


def from_chat_payload(payload: dict[str, Any], *, headers: dict[str, str], status_code: int, model: str) -> ModelResponse:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = [
        ToolCall(
            tool_call_id=call.get("id", ""),
            name=call.get("function", {}).get("name", ""),
            arguments=call.get("function", {}).get("arguments", "{}"),
        )
        for call in message.get("tool_calls", []) or []
    ]
    return ModelResponse(
        output_text=message.get("content"),
        tool_calls=tool_calls,
        raw=payload,
        usage=_usage_from_payload(payload),
        request_id=headers.get("x-request-id") or payload.get("id"),
        status_code=status_code,
        headers=headers,
        provider="openai",
        model=model,
    )


def chat_stream_events(
    payloads: Iterable[dict[str, Any]],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
):
    """Normalize Chat Completions chunks and finish with one complete response."""
    text_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    raw_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        raw_payloads.append(payload)
        choice = (payload.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
            yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta=content, raw=payload)
        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            state = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            state["id"] = call.get("id") or state["id"]
            function = call.get("function") or {}
            state["name"] = function.get("name") or state["name"]
            arguments = function.get("arguments") or ""
            state["arguments"] += arguments
            yield ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL_DELTA,
                tool_call=ToolCall(
                    tool_call_id=state["id"],
                    name=state["name"],
                    arguments=arguments,
                ),
                raw=payload,
            )
    response = ModelResponse(
        output_text="".join(text_parts) or None,
        tool_calls=[ToolCall(tool_call_id=value["id"], name=value["name"], arguments=value["arguments"] or "{}") for value in calls.values()],
        raw=raw_payloads,
        headers=headers,
        status_code=status_code,
        provider="openai",
        model=model,
    )
    yield ModelStreamEvent.completed(response)


def responses_stream_events(
    payloads: Iterable[dict[str, Any]],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
):
    """Normalize typed Responses SSE events and preserve the terminal response."""
    text_parts: list[str] = []
    calls: dict[str, dict[str, str]] = {}
    completed: ModelResponse | None = None
    raw_payloads: list[dict[str, Any]] = []
    for event in payloads:
        raw_payloads.append(event)
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                text_parts.append(delta)
                yield ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta=delta, raw=event)
        elif event_type == "response.function_call_arguments.delta":
            key = event.get("call_id") or event.get("item_id") or ""
            state = calls.setdefault(key, {"id": key, "name": event.get("name", ""), "arguments": ""})
            state["name"] = event.get("name") or state["name"]
            delta = event.get("delta", "")
            state["arguments"] += delta
            yield ModelStreamEvent(type=ModelStreamEventType.TOOL_CALL_DELTA, tool_call=ToolCall(tool_call_id=state["id"], name=state["name"], arguments=delta), raw=event)
        elif event_type == "response.function_call_arguments.done":
            key = event.get("call_id") or event.get("item_id") or ""
            state = calls.setdefault(key, {"id": key, "name": "", "arguments": ""})
            state["name"] = event.get("name") or state["name"]
            state["arguments"] = event.get("arguments") or state["arguments"]
        elif event_type == "response.completed" and isinstance(event.get("response"), dict):
            completed = from_responses_payload(event["response"], headers=headers, status_code=status_code, model=model)
    if completed is None:
        completed = ModelResponse(output_text="".join(text_parts) or None, tool_calls=[ToolCall(tool_call_id=value["id"], name=value["name"], arguments=value["arguments"] or "{}") for value in calls.values()], raw=raw_payloads, headers=headers, status_code=status_code, provider="openai", model=model)
    yield ModelStreamEvent.completed(completed)


async def chat_stream_events_async(
    payloads: AsyncIterable[dict[str, Any]],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
):
    """Async counterpart of ``chat_stream_events`` for native SSE clients."""
    text_parts: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    raw_payloads: list[dict[str, Any]] = []
    async for payload in payloads:
        raw_payloads.append(payload)
        choice = (payload.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
            yield ModelStreamEvent(
                type=ModelStreamEventType.TEXT_DELTA,
                text_delta=content,
                raw=payload,
            )
        for call in delta.get("tool_calls") or []:
            index = call.get("index", 0)
            state = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            state["id"] = call.get("id") or state["id"]
            function = call.get("function") or {}
            state["name"] = function.get("name") or state["name"]
            arguments = function.get("arguments") or ""
            state["arguments"] += arguments
            yield ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL_DELTA,
                tool_call=ToolCall(
                    tool_call_id=state["id"],
                    name=state["name"],
                    arguments=arguments,
                ),
                raw=payload,
            )
    response = ModelResponse(
        output_text="".join(text_parts) or None,
        tool_calls=[
            ToolCall(
                tool_call_id=value["id"],
                name=value["name"],
                arguments=value["arguments"] or "{}",
            )
            for value in calls.values()
        ],
        raw=raw_payloads,
        headers=headers,
        status_code=status_code,
        provider="openai",
        model=model,
    )
    yield ModelStreamEvent.completed(response)


async def responses_stream_events_async(
    payloads: AsyncIterable[dict[str, Any]],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
):
    """Async counterpart of ``responses_stream_events`` for native SSE clients."""
    text_parts: list[str] = []
    calls: dict[str, dict[str, str]] = {}
    completed: ModelResponse | None = None
    raw_payloads: list[dict[str, Any]] = []
    async for event in payloads:
        raw_payloads.append(event)
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = event.get("delta", "")
            if delta:
                text_parts.append(delta)
                yield ModelStreamEvent(
                    type=ModelStreamEventType.TEXT_DELTA,
                    text_delta=delta,
                    raw=event,
                )
        elif event_type == "response.function_call_arguments.delta":
            key = event.get("call_id") or event.get("item_id") or ""
            state = calls.setdefault(key, {"id": key, "name": event.get("name", ""), "arguments": ""})
            state["name"] = event.get("name") or state["name"]
            delta = event.get("delta", "")
            state["arguments"] += delta
            yield ModelStreamEvent(
                type=ModelStreamEventType.TOOL_CALL_DELTA,
                tool_call=ToolCall(
                    tool_call_id=state["id"],
                    name=state["name"],
                    arguments=delta,
                ),
                raw=event,
            )
        elif event_type == "response.function_call_arguments.done":
            key = event.get("call_id") or event.get("item_id") or ""
            state = calls.setdefault(key, {"id": key, "name": "", "arguments": ""})
            state["name"] = event.get("name") or state["name"]
            state["arguments"] = event.get("arguments") or state["arguments"]
        elif event_type == "response.completed" and isinstance(event.get("response"), dict):
            completed = from_responses_payload(
                event["response"],
                headers=headers,
                status_code=status_code,
                model=model,
            )
    if completed is None:
        completed = ModelResponse(
            output_text="".join(text_parts) or None,
            tool_calls=[
                ToolCall(
                    tool_call_id=value["id"],
                    name=value["name"],
                    arguments=value["arguments"] or "{}",
                )
                for value in calls.values()
            ],
            raw=raw_payloads,
            headers=headers,
            status_code=status_code,
            provider="openai",
            model=model,
        )
    yield ModelStreamEvent.completed(completed)


def tool_results_to_responses_input(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function_call_output",
            "call_id": result["tool_call_id"],
            "output": json.dumps(result["output"], separators=(",", ":")),
        }
        for result in tool_results
    ]


def _tool_parameters(request: ModelRequest, tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool["parameters"]
    return (
        to_openai_strict_schema(parameters)
        if request.tool_strict
        else deepcopy(parameters)
    )


def to_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Lower a generic JSON Schema into OpenAI strict structured-output shape."""
    return _normalize_schema(deepcopy(schema))


def _normalize_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    schema.pop("default", None)
    nullable = schema.pop("nullable", False)
    if nullable and "type" in schema:
        schema["type"] = _nullable_type(schema["type"])

    if schema.get("type") == "object":
        properties = schema.get("properties") or {}
        natural_required = set(schema.get("required") or [])
        schema["properties"] = {
            name: _strict_property_schema(
                _normalize_schema(value),
                required=name in natural_required,
            )
            for name, value in properties.items()
        }
        schema["required"] = list(schema["properties"].keys())
        schema["additionalProperties"] = False

    if schema.get("type") == "array" and "items" in schema:
        schema["items"] = _normalize_schema(schema["items"])

    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = [_normalize_schema(item) for item in schema[keyword]]

    if "$defs" in schema:
        schema["$defs"] = {
            name: _normalize_schema(definition)
            for name, definition in schema["$defs"].items()
        }

    return schema


def _nullable_type(value: Any) -> Any:
    if isinstance(value, list):
        return value if "null" in value else [*value, "null"]
    return [value, "null"]


def _strict_property_schema(schema: dict[str, Any], *, required: bool) -> dict[str, Any]:
    if required:
        return schema
    if "type" in schema:
        schema["type"] = _nullable_type(schema["type"])
        return schema
    if "anyOf" in schema:
        if not any(item == {"type": "null"} for item in schema["anyOf"]):
            schema["anyOf"].append({"type": "null"})
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def _apply_provider_state(payload: dict[str, Any], state: ProviderState | None) -> None:
    if state is None:
        return
    if state.previous_response_id:
        payload["previous_response_id"] = state.previous_response_id
    if state.conversation_id:
        payload["conversation"] = state.conversation_id


def _input_items_for_provider_state(
    input_items: list[dict[str, Any]],
    state: ProviderState | None,
) -> list[dict[str, Any]]:
    if state is None or not state.previous_response_id:
        return input_items
    tool_outputs = [
        item for item in input_items
        if item.get("type") == "function_call_output"
    ]
    return tool_outputs or input_items


def _provider_state_from_response(payload: dict[str, Any]) -> ProviderState | None:
    response_id = payload.get("id")
    conversation = payload.get("conversation") or payload.get("conversation_id")
    if isinstance(conversation, dict):
        conversation_id = conversation.get("id")
    else:
        conversation_id = conversation
    if not response_id and not conversation_id:
        return None
    return ProviderState(
        provider="openai",
        previous_response_id=response_id,
        conversation_id=conversation_id,
    )


def _split_instructions(messages: list[Message | SessionItem]) -> tuple[str | None, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for item in messages:
        if isinstance(item, SessionItem):
            operation = _operation_input_item(item)
            if operation is not None:
                input_items.append(operation)
            continue
        message = item
        if message.role in ("system", "developer"):
            instructions.append(_instructions_payload(message.content))
            continue
        if message.role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": _string_payload(message.content),
                }
            )
            continue
        input_items.append(
            {
                "role": message.role,
                "content": _responses_content(message.content),
            }
        )
    return "\n\n".join(instructions) if instructions else None, input_items


def _message_payload(item: Message | SessionItem) -> dict[str, Any]:
    if isinstance(item, SessionItem):
        return _operation_chat_message(item)
    message = item
    if message.role == "tool":
        return {
            key: value
            for key, value in {
                "role": message.role,
                "content": message.content,
                "tool_call_id": message.tool_call_id,
                "name": message.name,
            }.items()
            if value is not None
        }
    return {"role": message.role, "content": _chat_content(message.content)}


def _operation_input_item(item: SessionItem) -> dict[str, Any] | None:
    if item.type in ("function_call", "handoff_call"):
        return {
            "type": "function_call",
            "call_id": item.tool_call_id,
            "name": item.name,
            "arguments": _arguments_payload(item.arguments),
        }
    if item.type in ("function_call_output", "handoff_call_output"):
        return {
            "type": "function_call_output",
            "call_id": item.tool_call_id,
            "output": _string_payload(item.output),
        }
    if item.type == "message":
        return _message_input_item(item.to_message())
    return None


def _operation_chat_message(item: SessionItem) -> dict[str, Any]:
    if item.type in ("function_call", "handoff_call"):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": item.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": item.name,
                        "arguments": _arguments_payload(item.arguments),
                    },
                }
            ],
        }
    if item.type in ("function_call_output", "handoff_call_output"):
        return {
            "role": "tool",
            "content": _string_payload(item.output),
            "name": item.name,
            "tool_call_id": item.tool_call_id,
        }
    if item.type == "message":
        return _message_payload(item.to_message())
    return {"role": "assistant", "content": dumps(item.to_input())}


def _arguments_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value or {}, separators=(",", ":"))


def _string_payload(value: Any) -> str:
    if isinstance(value, str):
        return value
    return dumps(value)


def _message_input_item(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {
            "type": "function_call_output",
            "call_id": message.tool_call_id,
            "output": _string_payload(message.content),
        }
    return {
        "role": message.role,
        "content": _responses_content(message.content),
    }


def _instructions_payload(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = _content_inputs(content)
    if any(not isinstance(part, TextInput) for part in parts):
        raise ValueError("OpenAI Responses instructions support text input only.")
    return "\n".join(part.text for part in parts)


def _responses_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    return [_responses_content_part(part) for part in _content_inputs(content)]


def _responses_content_part(part: TextInput | FileInput | ImageInput) -> dict[str, Any]:
    if isinstance(part, TextInput):
        return {"type": "input_text", "text": part.text}
    if isinstance(part, FileInput):
        payload: dict[str, Any] = {"type": "input_file"}
        if part.file_id is not None:
            payload["file_id"] = part.file_id
        elif part.url is not None:
            payload["file_url"] = part.url
        else:
            payload["file_data"] = _data_url(part.data or "", part.media_type)
        if part.filename is not None:
            payload["filename"] = part.filename
        if part.detail is not None:
            payload["detail"] = part.detail.value
        return payload
    return {
        "type": "input_image",
        "image_url": part.url or _data_url(part.data or "", part.media_type),
        "detail": part.detail.value,
    }


def _chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = _content_inputs(content)
    if any(not isinstance(part, TextInput) for part in parts):
        raise ValueError(
            "OpenAI Chat Completions does not support FileInput or ImageInput; use OpenAI Responses."
        )
    return "\n".join(part.text for part in parts)


def _content_inputs(content: Any) -> list[TextInput | FileInput | ImageInput]:
    if not isinstance(content, list):
        raise ValueError("Multimodal message content must be a list of typed input parts.")
    return [_content_input(part) for part in content]


def _content_input(value: Any) -> TextInput | FileInput | ImageInput:
    if isinstance(value, (TextInput, FileInput, ImageInput)):
        return value
    if isinstance(value, dict):
        part_type = value.get("type")
        if part_type == "text":
            return TextInput(**value)
        if part_type == "file":
            return FileInput(**value)
        if part_type == "image":
            return ImageInput(**value)
    raise ValueError("Message content parts must be TextInput, FileInput, or ImageInput.")


def _data_url(data: str | bytes, media_type: str | None) -> str:
    if isinstance(data, bytes):
        data = b64encode(data).decode("ascii")
    if data.startswith("data:"):
        return data
    return f"data:{media_type or 'application/octet-stream'};base64,{data}"


def _apply_responses_settings(payload: dict[str, Any], request: ModelRequest) -> None:
    settings = request.settings
    if not settings:
        return
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature
    if settings.top_p is not None:
        payload["top_p"] = settings.top_p
    if settings.max_tokens is not None:
        payload["max_output_tokens"] = settings.max_tokens


def _apply_chat_settings(payload: dict[str, Any], request: ModelRequest) -> None:
    settings = request.settings
    if not settings:
        return
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature
    if settings.top_p is not None:
        payload["top_p"] = settings.top_p
    if settings.max_tokens is not None:
        payload["max_completion_tokens"] = settings.max_tokens


def _usage_from_payload(payload: dict[str, Any]) -> Usage:
    raw = payload.get("usage") or {}
    input_tokens = _int(raw, "input_tokens") or _int(raw, "prompt_tokens")
    output_tokens = _int(raw, "output_tokens") or _int(raw, "completion_tokens")
    total_tokens = _int(raw, "total_tokens") or input_tokens + output_tokens
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=(
            _nested_int(raw, "input_tokens_details", "cached_tokens")
            or _nested_int(raw, "prompt_tokens_details", "cached_tokens")
        ),
        reasoning_output_tokens=(
            _nested_int(raw, "output_tokens_details", "reasoning_tokens")
            or _nested_int(raw, "completion_tokens_details", "reasoning_tokens")
        ),
        details={"raw": raw} if raw else {},
    )


def _int(value: dict[str, Any], key: str) -> int:
    found = value.get(key)
    return found if isinstance(found, int) else 0


def _nested_int(value: dict[str, Any], parent: str, key: str) -> int:
    found = value.get(parent)
    if isinstance(found, dict) and isinstance(found.get(key), int):
        return found[key]
    return 0
