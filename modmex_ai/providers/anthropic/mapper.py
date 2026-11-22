"""Translate provider-neutral requests to the Anthropic Messages API and back.

Wire format reference: ``POST /v1/messages`` with ``anthropic-version: 2026-10-21``.

Thinking blocks
---------------
Current Claude models always think. When a turn ends in ``tool_use`` the API
requires the assistant content of that turn (including signed ``thinking``
blocks) to be sent back unchanged, and editing earlier turns invalidates later
thinking blocks. The adapter therefore stores the complete assistant content in
``ToolCall.provider_data`` of the first tool call of the turn; the runner copies
it to the ``SessionItem`` and this mapper replays it verbatim.
"""

from __future__ import annotations

import hashlib
import json
from base64 import b64decode, b64encode
from collections.abc import AsyncIterable, Iterable
from copy import deepcopy
from typing import Any

from modmex_ai.errors import ModelRefusalError, ProviderError
from modmex_ai.messages import FileInput, ImageInput, Message, TextInput
from modmex_ai.models import (
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelStreamEventType,
    ToolCall,
    Usage,
)
from modmex_ai.schemas import dumps
from modmex_ai.sessions import SessionItem


PROVIDER = "anthropic"
DEFAULT_MAX_TOKENS = 16_000
DEFAULT_STREAM_MAX_TOKENS = 64_000
EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# JSON Schema keywords the structured-output compiler rejects.
_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "maxItems",
    "uniqueItems",
}


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def to_messages_payload(
    request: ModelRequest, model: str, *, stream: bool = False
) -> dict[str, Any]:
    _check_provider_state(request)
    replays: list[tuple[str | None, list[dict[str, Any]]]] = []
    system, messages = _split_messages(request.messages, replays)
    settings = request.settings
    max_tokens = settings.max_tokens if settings and settings.max_tokens else None
    payload: dict[str, Any] = {
        "model": request.model or model,
        "max_tokens": max_tokens
        or (DEFAULT_STREAM_MAX_TOKENS if stream else DEFAULT_MAX_TOKENS),
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if request.tools:
        payload["tools"] = [_tool_payload(request, tool) for tool in request.tools]
    _drop_stale_thinking(messages, replays, binding_digest(payload))
    output_config: dict[str, Any] = {}
    if request.output_schema:
        output_config["format"] = {
            "type": "json_schema",
            "schema": to_anthropic_schema(request.output_schema),
        }
    _apply_settings(payload, output_config, request)
    if output_config:
        payload["output_config"] = output_config
    if stream:
        payload["stream"] = True
    return payload


def beta_headers(request: ModelRequest) -> list[str]:
    """Return the ``anthropic-beta`` values a request needs."""
    settings = request.settings.anthropic if request.settings else None
    if settings is None:
        return []
    betas = list(settings.betas or [])
    if settings.fallbacks is not None and FALLBACK_BETA not in betas:
        betas.append(FALLBACK_BETA)
    return betas


def binding_digest(payload: dict[str, Any]) -> str:
    """Fingerprint of what a thinking signature is bound to besides the messages.

    Anthropic binds each thinking block to the ``system`` prompt and the *set* of
    tools (order-insensitive) it was produced with. A tool_selector that changes
    the tools, or a handoff to an agent with other instructions, changes it.
    """
    tools = sorted(payload.get("tools") or [], key=lambda tool: tool.get("name", ""))
    material = json.dumps(
        {"system": payload.get("system"), "tools": tools},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _drop_stale_thinking(
    turns: list[dict[str, Any]],
    replays: list[tuple[str | None, list[dict[str, Any]]]],
    current: str,
) -> None:
    """Remove thinking blocks whose binding no longer matches this request.

    The API accepts removing a *leading* run of thinking blocks but not a gap in
    the middle, so everything up to the last stale block goes. Blocks recorded
    without a binding (older data) are treated as current.
    """
    last_stale = max(
        (index for index, (binding, _) in enumerate(replays) if binding not in (None, current)),
        default=-1,
    )
    if last_stale < 0:
        return
    stale = {id(block) for _, blocks in replays[: last_stale + 1] for block in blocks}
    for turn in turns:
        if isinstance(turn["content"], list):
            turn["content"] = [b for b in turn["content"] if id(b) not in stale]


def _check_provider_state(request: ModelRequest) -> None:
    state = request.provider_state
    if state is None:
        return
    if state.provider not in {None, PROVIDER}:
        raise ValueError("Provider state does not belong to Anthropic")
    if state.has_remote_state:
        raise ValueError(
            "The Anthropic Messages API is stateless; use stateless provider state"
        )


def _tool_payload(request: ModelRequest, tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters") or {"type": "object", "properties": {}}
    payload: dict[str, Any] = {"name": tool["name"]}
    if tool.get("description"):
        payload["description"] = tool["description"]
    if request.tool_strict:
        payload["input_schema"] = to_anthropic_schema(parameters)
        payload["strict"] = True
    else:
        payload["input_schema"] = deepcopy(parameters)
    return payload


def _apply_settings(
    payload: dict[str, Any], output_config: dict[str, Any], request: ModelRequest
) -> None:
    settings = request.settings
    if not settings:
        return
    if settings.openai is not None:
        raise ValueError("OpenAI settings cannot be sent to Anthropic")
    if settings.temperature is not None or settings.top_p is not None:
        # API version 2026-10-21 rejects sampling parameters on every model.
        raise ValueError(
            "Anthropic does not accept temperature or top_p; remove them from ModelSettings"
        )
    if settings.reasoning_effort is not None:
        if settings.reasoning_effort not in EFFORT_LEVELS:
            raise ValueError(
                "Invalid reasoning_effort for Anthropic; use low, medium, high, xhigh, or max"
            )
        output_config["effort"] = settings.reasoning_effort
    if settings.parallel_tool_calls is False and request.tools:
        payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    anthropic = settings.anthropic
    if anthropic is None:
        return
    if anthropic.adaptive_thinking or anthropic.thinking_display is not None:
        thinking: dict[str, Any] = {"type": "adaptive"}
        if anthropic.thinking_display is not None:
            thinking["display"] = anthropic.thinking_display
        payload["thinking"] = thinking
    if anthropic.cache_ttl is not None:
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        if anthropic.cache_ttl == "1h":
            cache_control["ttl"] = "1h"
        payload["cache_control"] = cache_control
    if anthropic.fallbacks is not None:
        payload["fallbacks"] = (
            anthropic.fallbacks
            if anthropic.fallbacks == "default"
            else [{"model": name} for name in anthropic.fallbacks]
        )


def to_anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Lower a generic JSON Schema into the subset structured outputs accept.

    Objects get ``additionalProperties: false``; numeric/length constraints the
    compiler rejects are dropped (validate them in code if they matter). Unlike
    OpenAI strict mode, optional properties stay optional.
    """
    return _normalize_schema(deepcopy(schema))


def _normalize_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_normalize_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    for key in _UNSUPPORTED_SCHEMA_KEYS:
        schema.pop(key, None)
    if isinstance(schema.get("minItems"), int) and schema["minItems"] > 1:
        schema.pop("minItems")
    if schema.pop("nullable", False) and "type" in schema:
        current = schema["type"]
        if isinstance(current, list):
            schema["type"] = current if "null" in current else [*current, "null"]
        else:
            schema["type"] = [current, "null"]
    if schema.get("type") == "object" or "properties" in schema:
        schema["properties"] = {
            name: _normalize_schema(value)
            for name, value in (schema.get("properties") or {}).items()
        }
        schema["additionalProperties"] = False
    if "items" in schema:
        schema["items"] = _normalize_schema(schema["items"])
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            schema[keyword] = _normalize_schema(schema[keyword])
    for keyword in ("$defs", "definitions"):
        if keyword in schema:
            schema[keyword] = {
                name: _normalize_schema(value)
                for name, value in schema[keyword].items()
            }
    return schema


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


def _split_messages(
    items: list[Message | SessionItem],
    replays: list[tuple[str | None, list[dict[str, Any]]]] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Leading system/developer text becomes ``system``; the rest becomes turns.

    A system/developer message after the conversation started is sent as a
    mid-conversation ``role: "system"`` message, so the top-level system prompt
    (and with it the prompt cache and thinking-block binding) stays frozen.
    """
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    replayed_tool_ids: set[str] = set()
    for item in items:
        role = getattr(item, "role", None)
        is_message = isinstance(item, Message) or (
            isinstance(item, SessionItem) and item.type == "message"
        )
        if is_message and role in {"system", "developer"}:
            text = _text_only(_message_of(item).content, what="System messages")
            if not turns:
                system_parts.append(text)
            elif text:
                turns.append({"role": "system", "content": text})
            continue
        if isinstance(item, SessionItem) and item.type != "message":
            _append_operation(turns, item, replayed_tool_ids, replays)
            continue
        message = _message_of(item)
        if message.role == "tool":
            _append_blocks(
                turns,
                "user",
                [_tool_result_block(message.tool_call_id, message.content)],
            )
            continue
        blocks = _content_blocks(message.content, role=message.role)
        if blocks:
            _append_blocks(turns, message.role, blocks)
    return ("\n\n".join(system_parts) if system_parts else None), _place_system_turns(
        turns
    )


def _place_system_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep mid-conversation system turns only where the API accepts them.

    A ``role: "system"`` message must follow a user turn and be last or be
    followed by an assistant turn. Elsewhere (e.g. a guardrail retry right after
    an assistant answer) it is sent as user text instead.
    """
    placed: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if turn["role"] == "system":
            previous = placed[-1]["role"] if placed else None
            following = turns[index + 1]["role"] if index + 1 < len(turns) else None
            if previous != "user" or following not in {None, "assistant"}:
                text = f"<system>\n{turn['content']}\n</system>"
                _append_blocks(placed, "user", [{"type": "text", "text": text}])
                continue
            placed.append(turn)
            continue
        if placed and placed[-1]["role"] == turn["role"]:
            placed[-1]["content"].extend(turn["content"])
        else:
            placed.append(turn)
    return placed


def _message_of(item: Message | SessionItem) -> Message:
    return item if isinstance(item, Message) else item.to_message()


def _append_operation(
    turns: list[dict[str, Any]],
    item: SessionItem,
    replayed_tool_ids: set[str],
    replays: list[tuple[str | None, list[dict[str, Any]]]] | None = None,
) -> None:
    if item.type in ("function_call", "handoff_call"):
        replay = _replay_content(item)
        if replay is not None:
            turns.append({"role": "assistant", "content": replay})
            if replays is not None:
                replays.append(
                    (
                        _replay_binding(item),
                        [
                            block
                            for block in replay
                            if isinstance(block, dict)
                            and block.get("type") in {"thinking", "redacted_thinking"}
                        ],
                    )
                )
            replayed_tool_ids.update(
                block.get("id")
                for block in replay
                if isinstance(block, dict) and block.get("type") == "tool_use"
            )
            return
        if item.tool_call_id in replayed_tool_ids:
            return
        _append_blocks(
            turns,
            "assistant",
            [
                {
                    "type": "tool_use",
                    "id": item.tool_call_id,
                    "name": item.name,
                    "input": _arguments_object(item.arguments),
                }
            ],
        )
        return
    if item.type in ("function_call_output", "handoff_call_output"):
        _append_blocks(
            turns, "user", [_tool_result_block(item.tool_call_id, item.output)]
        )


def _replay_content(item: SessionItem) -> list[dict[str, Any]] | None:
    data = item.provider_data or {}
    anthropic = data.get(PROVIDER) if isinstance(data, dict) else None
    content = anthropic.get("content") if isinstance(anthropic, dict) else None
    return deepcopy(content) if isinstance(content, list) and content else None


def _replay_binding(item: SessionItem) -> str | None:
    data = item.provider_data or {}
    anthropic = data.get(PROVIDER) if isinstance(data, dict) else None
    binding = anthropic.get("binding") if isinstance(anthropic, dict) else None
    return binding if isinstance(binding, str) else None


def _append_blocks(
    turns: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]
) -> None:
    # Consecutive same-role turns are one turn to the API; merging keeps parallel
    # tool results in a single user message, as the API expects.
    if turns and turns[-1]["role"] == role and isinstance(turns[-1]["content"], list):
        turns[-1]["content"].extend(blocks)
    else:
        turns.append({"role": role, "content": list(blocks)})


def _tool_result_block(tool_use_id: str | None, output: Any) -> dict[str, Any]:
    if not tool_use_id:
        raise ValueError("Tool results require the originating tool_call_id")
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": output if isinstance(output, str) else dumps(output),
    }
    if _is_tool_failure(output):
        block["is_error"] = True
    return block


def _is_tool_failure(output: Any) -> bool:
    """The envelope ToolExecutor returns for a failed call under "return_error"."""
    return (
        isinstance(output, dict)
        and output.get("ok") is False
        and isinstance(output.get("error"), dict)
    )


def _arguments_object(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Tool call arguments must be a JSON object")
        return parsed
    return dict(value)


def _content_blocks(content: Any, *, role: str) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in _content_inputs(content):
        if isinstance(part, TextInput):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
            continue
        if role == "assistant":
            raise ValueError("Anthropic assistant messages support text content only.")
        blocks.append(
            _image_block(part) if isinstance(part, ImageInput) else _document_block(part)
        )
    return blocks


def _text_only(content: Any, *, what: str) -> str:
    if isinstance(content, str):
        return content
    parts = _content_inputs(content)
    if any(not isinstance(part, TextInput) for part in parts):
        raise ValueError(f"{what} support text input only.")
    return "\n".join(part.text for part in parts)


def _image_block(part: ImageInput) -> dict[str, Any]:
    if part.url is not None:
        return {"type": "image", "source": {"type": "url", "url": part.url}}
    media_type, data = _base64_source(part.data or "", part.media_type)
    if not media_type:
        raise ValueError("ImageInput with inline data requires media_type")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def _document_block(part: FileInput) -> dict[str, Any]:
    if part.file_id is not None:
        source: dict[str, Any] = {"type": "file", "file_id": part.file_id}
    elif part.url is not None:
        source = {"type": "url", "url": part.url}
    else:
        media_type, data = _base64_source(part.data or "", part.media_type)
        if media_type == "application/pdf":
            source = {"type": "base64", "media_type": media_type, "data": data}
        elif media_type == "text/plain":
            source = {
                "type": "text",
                "media_type": media_type,
                "data": b64decode(data).decode("utf-8"),
            }
        else:
            raise ValueError(
                "Anthropic inline documents support application/pdf or text/plain"
            )
    block: dict[str, Any] = {"type": "document", "source": source}
    if part.filename:
        block["title"] = part.filename
    return block


def _base64_source(data: str | bytes, media_type: str | None) -> tuple[str | None, str]:
    if isinstance(data, bytes):
        return media_type, b64encode(data).decode("ascii")
    if data.startswith("data:"):
        header, _, encoded = data.partition(",")
        declared = header[5:].split(";", 1)[0] or None
        return media_type or declared, encoded
    return media_type, data


def _content_inputs(content: Any) -> list[TextInput | FileInput | ImageInput]:
    if not isinstance(content, list):
        raise ValueError("Multimodal message content must be a list of typed input parts.")
    parts: list[TextInput | FileInput | ImageInput] = []
    for value in content:
        if isinstance(value, (TextInput, FileInput, ImageInput)):
            parts.append(value)
        elif isinstance(value, dict) and value.get("type") == "text":
            parts.append(TextInput(**value))
        elif isinstance(value, dict) and value.get("type") == "file":
            parts.append(FileInput(**value))
        elif isinstance(value, dict) and value.get("type") == "image":
            parts.append(ImageInput(**value))
        else:
            raise ValueError(
                "Message content parts must be TextInput, FileInput, or ImageInput."
            )
    return parts


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #


def from_messages_payload(
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
    binding: str | None = None,
) -> ModelResponse:
    content = payload.get("content") or []
    stop_reason = payload.get("stop_reason")
    request_id = _header(headers, "request-id") or payload.get("id")
    if stop_reason == "refusal":
        raise ModelRefusalError(
            "Anthropic declined the request (stop_reason=refusal)",
            stop_details=payload.get("stop_details"),
            status_code=status_code,
            request_id=request_id,
            response_body=payload,
            headers=headers,
        )
    tool_calls = [
        ToolCall(
            tool_call_id=block.get("id", ""),
            name=block.get("name", ""),
            arguments=block.get("input") or {},
        )
        for block in content
        if block.get("type") == "tool_use"
    ]
    if tool_calls and stop_reason == "max_tokens":
        raise ProviderError(
            "Anthropic response hit max_tokens during a tool call; raise max_tokens",
            status_code=status_code,
            request_id=request_id,
            response_body=payload,
            headers=headers,
        )
    if tool_calls:
        replay: dict[str, Any] = {"content": deepcopy(content)}
        if binding is not None:
            replay["binding"] = binding
        tool_calls[0].provider_data = {PROVIDER: replay}
    text = "".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )
    return ModelResponse(
        output_text=text or None,
        tool_calls=tool_calls,
        raw=payload,
        usage=_usage_from_payload(payload.get("usage") or {}),
        request_id=request_id,
        status_code=status_code,
        headers=headers,
        provider=PROVIDER,
        model=payload.get("model") or model,
        stop_reason=stop_reason,
    )


def _usage_from_payload(raw: dict[str, Any]) -> Usage:
    uncached = _int(raw, "input_tokens")
    cache_read = _int(raw, "cache_read_input_tokens")
    cache_write = _int(raw, "cache_creation_input_tokens")
    # Normalize to OpenAI semantics: input_tokens includes cached tokens.
    input_tokens = uncached + cache_read + cache_write
    output_tokens = _int(raw, "output_tokens")
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=cache_read,
        details={"raw": raw} if raw else {},
    )


def _int(value: dict[str, Any], key: str) -> int:
    found = value.get(key)
    return found if isinstance(found, int) and not isinstance(found, bool) else 0


def _header(headers: dict[str, str], name: str) -> str | None:
    return next((v for k, v in headers.items() if k.lower() == name), None)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _StreamAccumulator:
    """Rebuild the final Message from SSE events while emitting deltas."""

    def __init__(
        self,
        *,
        headers: dict[str, str],
        status_code: int,
        model: str,
        binding: str | None = None,
    ) -> None:
        self.binding = binding
        self.headers = headers
        self.status_code = status_code
        self.model = model
        self.message: dict[str, Any] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.partial_json: dict[int, str] = {}

    def feed(self, event: dict[str, Any]) -> list[ModelStreamEvent]:
        event_type = event.get("type")
        if event_type == "message_start":
            self.message = dict(event.get("message") or {})
        elif event_type == "content_block_start":
            index = event.get("index", len(self.blocks))
            block = self.blocks[index] = dict(event.get("content_block") or {})
            if block.get("type") == "tool_use":
                self.partial_json[index] = ""
                # Announce the call as soon as it starts (tools without
                # arguments get no input_json_delta), like the OpenAI adapters.
                return [
                    ModelStreamEvent(
                        type=ModelStreamEventType.TOOL_CALL_DELTA,
                        tool_call=ToolCall(
                            tool_call_id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments="",
                        ),
                        raw=event,
                    )
                ]
        elif event_type == "content_block_delta":
            return self._delta(event)
        elif event_type == "content_block_stop":
            self._finish_block(event.get("index", 0))
        elif event_type == "message_delta":
            self.message.update(event.get("delta") or {})
            usage = dict(self.message.get("usage") or {})
            usage.update(event.get("usage") or {})
            self.message["usage"] = usage
        elif event_type == "error":
            error = event.get("error") or {}
            raise ProviderError(
                f"Anthropic stream error: {error.get('message') or error.get('type')}",
                status_code=529 if error.get("type") == "overloaded_error" else None,
                request_id=self.message.get("id"),
                response_body=event,
            )
        return []

    def _delta(self, event: dict[str, Any]) -> list[ModelStreamEvent]:
        index = event.get("index", 0)
        block = self.blocks.setdefault(index, {})
        delta = event.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text", "")
            block["text"] = block.get("text", "") + text
            if text:
                return [
                    ModelStreamEvent(
                        type=ModelStreamEventType.TEXT_DELTA, text_delta=text, raw=event
                    )
                ]
        elif delta_type == "input_json_delta":
            fragment = delta.get("partial_json", "")
            self.partial_json[index] = self.partial_json.get(index, "") + fragment
            if fragment:
                return [
                    ModelStreamEvent(
                        type=ModelStreamEventType.TOOL_CALL_DELTA,
                        tool_call=ToolCall(
                            tool_call_id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=fragment,
                        ),
                        raw=event,
                    )
                ]
        elif delta_type == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
        elif delta_type == "signature_delta":
            block["signature"] = block.get("signature", "") + delta.get("signature", "")
        elif delta_type == "citations_delta":
            block.setdefault("citations", []).append(delta.get("citation"))
        return []

    def _finish_block(self, index: int) -> None:
        if index not in self.partial_json:
            return
        text = self.partial_json.pop(index)
        try:
            self.blocks[index]["input"] = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            raise ProviderError(
                "Anthropic streamed invalid tool input JSON",
                request_id=self.message.get("id"),
                response_body={"partial_json": text},
            ) from exc

    def finish(self) -> ModelResponse:
        for index in list(self.partial_json):
            # A block that never received content_block_stop (truncated stream).
            self._finish_block(index)
        message = dict(self.message)
        message["content"] = [self.blocks[index] for index in sorted(self.blocks)]
        return from_messages_payload(
            message,
            headers=self.headers,
            status_code=self.status_code,
            model=self.model,
            binding=self.binding,
        )


def messages_stream_events(
    payloads: Iterable[dict[str, Any]],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
    binding: str | None = None,
):
    state = _StreamAccumulator(
        headers=headers, status_code=status_code, model=model, binding=binding
    )
    for payload in payloads:
        yield from state.feed(payload)
    yield ModelStreamEvent.completed(state.finish())


async def messages_stream_events_async(
    payloads: AsyncIterable[dict[str, Any]],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
    binding: str | None = None,
):
    state = _StreamAccumulator(
        headers=headers, status_code=status_code, model=model, binding=binding
    )
    async for payload in payloads:
        for event in state.feed(payload):
            yield event
    yield ModelStreamEvent.completed(state.finish())
