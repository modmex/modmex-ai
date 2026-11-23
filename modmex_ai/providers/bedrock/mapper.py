"""Translate provider-neutral requests to Amazon Bedrock Converse and back.

Converse is one request shape for most text models in the Bedrock catalog. What a
given model supports (documents, prompt caching, structured output, reasoning)
still varies; check its model card. Notable rules this mapper enforces:

- File inputs travel as bytes. Only Amazon Nova accepts ``s3://`` URIs directly,
  so other models get ``s3://`` objects resolved to bytes; ``https://`` URLs and
  provider file ids are rejected with a clear error.
- OpenAI models on Bedrock reject Converse document blocks (use the OpenAI
  Responses adapter pointed at Bedrock for PDFs).
- A document must be accompanied by a text block in the same message.
- Converse has no mid-conversation system role: later system/developer messages
  (e.g. ``turn_context``) are sent as user text.
- Reasoning blocks carry a signature that must be replayed unchanged; the whole
  assistant content of a tool-use turn is kept in ``ToolCall.provider_data``.
"""

from __future__ import annotations

import hashlib
import json
import re
from base64 import b64decode, b64encode
from collections.abc import AsyncIterable, Callable, Iterable
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
from modmex_ai.providers.anthropic.mapper import to_anthropic_schema
from modmex_ai.schemas import dumps
from modmex_ai.sessions import SessionItem


PROVIDER = "bedrock"

S3Loader = Callable[[str], bytes]

_IMAGE_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}
_DOCUMENT_FORMATS = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}
_DOCUMENT_EXTENSIONS = {fmt: fmt for fmt in _DOCUMENT_FORMATS.values()}
_DOCUMENT_EXTENSIONS["htm"] = "html"
_DOCUMENT_EXTENSIONS["markdown"] = "md"


def is_nova(model: str) -> bool:
    return "amazon.nova" in model


def is_openai(model: str) -> bool:
    return "openai." in model


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


def to_converse_payload(
    request: ModelRequest,
    model: str,
    *,
    load_s3: S3Loader | None = None,
) -> dict[str, Any]:
    _check_provider_state(request)
    target = request.model or model
    replays: list[tuple[str | None, list[dict[str, Any]]]] = []
    system, messages = _split_messages(request.messages, target, load_s3, replays)
    payload: dict[str, Any] = {"messages": messages}
    if system:
        payload["system"] = [{"text": text} for text in system]
    if request.tools:
        payload["toolConfig"] = {
            "tools": [{"toolSpec": _tool_spec(request, tool)} for tool in request.tools]
        }
    _drop_stale_reasoning(messages, replays, binding_digest(payload))
    if request.output_schema:
        schema = to_anthropic_schema(request.output_schema)
        payload["outputConfig"] = {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": json.dumps(schema, separators=(",", ":")),
                        "name": _schema_name(request.output_schema),
                    }
                },
            }
        }
    _apply_settings(payload, request)
    return payload


def _check_provider_state(request: ModelRequest) -> None:
    state = request.provider_state
    if state is None:
        return
    if state.provider not in {None, PROVIDER}:
        raise ValueError("Provider state does not belong to Bedrock Converse")
    if state.has_remote_state:
        raise ValueError("Bedrock Converse is stateless; use stateless provider state")


def _tool_spec(request: ModelRequest, tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters") or {"type": "object", "properties": {}}
    spec: dict[str, Any] = {"name": tool["name"]}
    if tool.get("description"):
        spec["description"] = tool["description"]
    if request.tool_strict:
        spec["strict"] = True
        spec["inputSchema"] = {"json": to_anthropic_schema(parameters)}
    else:
        spec["inputSchema"] = {"json": deepcopy(parameters)}
    return spec


def _schema_name(schema: dict[str, Any]) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(schema.get("title") or "output")).strip("_")
    return (name or "output")[:64]


def _apply_settings(payload: dict[str, Any], request: ModelRequest) -> None:
    settings = request.settings
    if not settings:
        return
    if settings.openai is not None or settings.anthropic is not None:
        raise ValueError(
            "OpenAI/Anthropic settings cannot be sent through Bedrock Converse; "
            "use BedrockSettings"
        )
    if settings.reasoning_effort is not None:
        # Each model family names this differently; guessing would send a field
        # the model rejects or silently ignores.
        raise ValueError(
            "reasoning_effort is model-specific on Bedrock Converse; pass the model's "
            "own field through BedrockSettings(additional_model_request_fields=...)"
        )
    inference: dict[str, Any] = {}
    if settings.max_tokens is not None:
        inference["maxTokens"] = settings.max_tokens
    if settings.temperature is not None:
        inference["temperature"] = settings.temperature
    if settings.top_p is not None:
        inference["topP"] = settings.top_p
    if inference:
        payload["inferenceConfig"] = inference
    bedrock = settings.bedrock
    if bedrock is None:
        return
    if bedrock.additional_model_request_fields:
        payload["additionalModelRequestFields"] = deepcopy(
            bedrock.additional_model_request_fields
        )
    if bedrock.guardrail_identifier is not None:
        payload["guardrailConfig"] = {
            "guardrailIdentifier": bedrock.guardrail_identifier,
            "guardrailVersion": bedrock.guardrail_version,
        }
    if bedrock.cache_ttl is not None:
        point: dict[str, Any] = {"cachePoint": {"type": "default"}}
        if bedrock.cache_ttl == "1h":
            point["cachePoint"]["ttl"] = "1h"
        if payload.get("system"):
            payload["system"].append(deepcopy(point))
        if payload["messages"]:
            payload["messages"][-1]["content"].append(deepcopy(point))


def binding_digest(payload: dict[str, Any]) -> str:
    """Fingerprint of the system prompt and tool set a reasoning block depends on."""
    tools = sorted(
        (payload.get("toolConfig") or {}).get("tools") or [],
        key=lambda tool: tool.get("toolSpec", {}).get("name", ""),
    )
    material = json.dumps(
        {"system": payload.get("system"), "tools": tools},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _drop_stale_reasoning(
    turns: list[dict[str, Any]],
    replays: list[tuple[str | None, list[dict[str, Any]]]],
    current: str,
) -> None:
    """Remove reasoning blocks up to the last one bound to another system/tool set."""
    last_stale = max(
        (index for index, (binding, _) in enumerate(replays) if binding not in (None, current)),
        default=-1,
    )
    if last_stale < 0:
        return
    stale = {id(block) for _, blocks in replays[: last_stale + 1] for block in blocks}
    for turn in turns:
        turn["content"] = [block for block in turn["content"] if id(block) not in stale]


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #


def _split_messages(
    items: list[Message | SessionItem],
    model: str,
    load_s3: S3Loader | None,
    replays: list[tuple[str | None, list[dict[str, Any]]]],
) -> tuple[list[str], list[dict[str, Any]]]:
    system: list[str] = []
    turns: list[dict[str, Any]] = []
    replayed_ids: set[str] = set()
    names: set[str] = set()
    for item in items:
        role = getattr(item, "role", None)
        is_message = isinstance(item, Message) or (
            isinstance(item, SessionItem) and item.type == "message"
        )
        if is_message and role in {"system", "developer"}:
            text = _text_only(_message_of(item).content)
            if not turns:
                if text:
                    system.append(text)
            elif text:
                _append(turns, "user", [{"text": f"<system>\n{text}\n</system>"}])
            continue
        if isinstance(item, SessionItem) and item.type != "message":
            _append_operation(turns, item, replayed_ids, replays)
            continue
        message = _message_of(item)
        if message.role == "tool":
            _append(turns, "user", [_tool_result(message.tool_call_id, message.content)])
            continue
        blocks = _content_blocks(message.content, message.role, model, load_s3, names)
        if blocks:
            _append(turns, message.role, blocks)
    return system, turns


def _message_of(item: Message | SessionItem) -> Message:
    return item if isinstance(item, Message) else item.to_message()


def _append(turns: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    # Converse expects alternating roles; consecutive same-role turns are one turn.
    if turns and turns[-1]["role"] == role:
        turns[-1]["content"].extend(blocks)
    else:
        turns.append({"role": role, "content": list(blocks)})


def _append_operation(
    turns: list[dict[str, Any]],
    item: SessionItem,
    replayed_ids: set[str],
    replays: list[tuple[str | None, list[dict[str, Any]]]],
) -> None:
    if item.type in ("function_call", "handoff_call"):
        data = (item.provider_data or {}).get(PROVIDER) if isinstance(item.provider_data, dict) else None
        content = data.get("content") if isinstance(data, dict) else None
        if isinstance(content, list) and content:
            replay = deepcopy(content)
            _append(turns, "assistant", replay)
            replayed_ids.update(
                block["toolUse"].get("toolUseId")
                for block in replay
                if isinstance(block, dict) and "toolUse" in block
            )
            binding = data.get("binding") if isinstance(data.get("binding"), str) else None
            replays.append(
                (binding, [block for block in replay if isinstance(block, dict) and "reasoningContent" in block])
            )
            return
        if item.tool_call_id in replayed_ids:
            return
        _append(
            turns,
            "assistant",
            [{"toolUse": {"toolUseId": item.tool_call_id, "name": item.name, "input": _arguments(item.arguments)}}],
        )
        return
    if item.type in ("function_call_output", "handoff_call_output"):
        _append(turns, "user", [_tool_result(item.tool_call_id, item.output)])


def _arguments(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Tool call arguments must be a JSON object")
        return parsed
    return dict(value)


def _tool_result(tool_use_id: str | None, output: Any) -> dict[str, Any]:
    if not tool_use_id:
        raise ValueError("Tool results require the originating tool_call_id")
    if isinstance(output, dict):
        content = [{"json": output}]
    elif isinstance(output, str):
        content = [{"text": output or " "}]
    else:
        content = [{"text": dumps(output)}]
    result: dict[str, Any] = {"toolUseId": tool_use_id, "content": content}
    if isinstance(output, dict) and output.get("ok") is False and isinstance(output.get("error"), dict):
        result["status"] = "error"  # ToolExecutor's failure envelope
    return {"toolResult": result}


def _text_only(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = _content_inputs(content)
    if any(not isinstance(part, TextInput) for part in parts):
        raise ValueError("System messages support text input only.")
    return "\n".join(part.text for part in parts)


def _content_blocks(
    content: Any,
    role: str,
    model: str,
    load_s3: S3Loader | None,
    names: set[str],
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts = _content_inputs(content)
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextInput):
            if part.text:
                blocks.append({"text": part.text})
            continue
        if role != "user":
            raise ValueError("Bedrock accepts images and documents only in user messages.")
        if isinstance(part, ImageInput):
            blocks.append(_image_block(part, model, load_s3))
        else:
            blocks.append(_document_block(part, model, load_s3, names))
    if any("document" in block for block in blocks) and not any("text" in block for block in blocks):
        raise ValueError(
            "Bedrock requires a text block in the same message as a document; "
            "add the question or instruction about the document."
        )
    return blocks


def _image_block(part: ImageInput, model: str, load_s3: S3Loader | None) -> dict[str, Any]:
    fmt = _IMAGE_FORMATS.get((part.media_type or "").lower())
    if fmt is None:
        raise ValueError("Bedrock images need media_type image/png, image/jpeg, image/gif or image/webp")
    return {"image": {"format": fmt, "source": _source(part.url, part.data, model, load_s3)}}


def _document_block(
    part: FileInput,
    model: str,
    load_s3: S3Loader | None,
    names: set[str],
) -> dict[str, Any]:
    if is_openai(model):
        raise ValueError(
            "OpenAI models on Bedrock reject Converse document blocks; send documents "
            "with OpenAIResponsesModel pointed at Bedrock (input_file) instead."
        )
    if part.file_id is not None:
        raise ValueError("Bedrock has no Files API; pass the document as bytes or an s3:// URI")
    fmt = _document_format(part)
    return {
        "document": {
            "format": fmt,
            "name": _document_name(part.filename, names),
            "source": _source(part.url, part.data, model, load_s3),
        }
    }


def _document_format(part: FileInput) -> str:
    fmt = _DOCUMENT_FORMATS.get((part.media_type or "").lower())
    if fmt is None and part.filename and "." in part.filename:
        fmt = _DOCUMENT_EXTENSIONS.get(part.filename.rsplit(".", 1)[1].lower())
    if fmt is None and part.url and "." in part.url.rsplit("/", 1)[-1]:
        fmt = _DOCUMENT_EXTENSIONS.get(part.url.rsplit(".", 1)[1].lower())
    if fmt is None:
        raise ValueError(
            "Bedrock needs the document format: set media_type (e.g. application/pdf) "
            "or a filename with one of pdf, csv, doc, docx, xls, xlsx, html, txt, md"
        )
    return fmt


def _document_name(filename: str | None, names: set[str]) -> str:
    # Allowed: alphanumerics, single spaces, hyphens, parentheses, brackets.
    # The name reaches the model, so keep it neutral (prompt-injection surface).
    stem = (filename or "document").rsplit(".", 1)[0]
    name = re.sub(r"[^A-Za-z0-9\s\-\(\)\[\]]+", "-", stem)
    name = re.sub(r"\s+", " ", name).strip(" -")[:180] or "document"
    candidate, counter = name, 2
    while candidate in names:
        candidate, counter = f"{name}-{counter}", counter + 1
    names.add(candidate)
    return candidate


def _source(
    url: str | None,
    data: bytes | str | None,
    model: str,
    load_s3: S3Loader | None,
) -> dict[str, Any]:
    if url is not None:
        if url.startswith("s3://"):
            if is_nova(model):
                return {"s3Location": {"uri": url}}
            if load_s3 is None:
                raise ValueError("s3:// inputs need an S3 loader on this model adapter")
            return {"bytes": b64encode(load_s3(url)).decode("ascii")}
        raise ValueError(
            "Bedrock does not fetch URLs; upload the file to S3 and pass an s3:// URI, "
            "or pass its bytes"
        )
    if isinstance(data, (bytes, bytearray)):
        return {"bytes": b64encode(bytes(data)).decode("ascii")}
    text = data or ""
    if text.startswith("data:"):
        text = text.partition(",")[2]
    b64decode(text, validate=True)  # fail here, not as an opaque Bedrock 400
    return {"bytes": text}


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
            raise ValueError("Message content parts must be TextInput, FileInput, or ImageInput.")
    return parts


# --------------------------------------------------------------------------- #
# Response
# --------------------------------------------------------------------------- #

_STOP_REASONS = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "content_filtered": "refusal",
    "guardrail_intervened": "guardrail_intervened",
}


def from_converse_payload(
    body: dict[str, Any],
    *,
    headers: dict[str, str],
    status_code: int,
    model: str,
    binding: str | None = None,
) -> ModelResponse:
    message = (body.get("output") or {}).get("message") or {}
    content = message.get("content") or []
    raw_stop = body.get("stopReason")
    request_id = _header(headers, "x-amzn-requestid")
    if raw_stop == "content_filtered":
        raise ModelRefusalError(
            "Bedrock filtered the response (stopReason=content_filtered)",
            status_code=status_code,
            request_id=request_id,
            response_body=body,
            headers=headers,
        )
    if raw_stop in {"malformed_model_output", "malformed_tool_use", "model_context_window_exceeded"}:
        raise ProviderError(
            f"Bedrock stopped with {raw_stop}",
            status_code=status_code,
            request_id=request_id,
            response_body=body,
            headers=headers,
        )
    tool_calls = [
        ToolCall(
            tool_call_id=block["toolUse"].get("toolUseId", ""),
            name=block["toolUse"].get("name", ""),
            arguments=block["toolUse"].get("input") or {},
        )
        for block in content
        if isinstance(block, dict) and "toolUse" in block
    ]
    if tool_calls and raw_stop == "max_tokens":
        raise ProviderError(
            "Bedrock response hit max_tokens during a tool call; raise max_tokens",
            status_code=status_code,
            request_id=request_id,
            response_body=body,
            headers=headers,
        )
    if tool_calls:
        replay: dict[str, Any] = {"content": deepcopy(content)}
        if binding is not None:
            replay["binding"] = binding
        tool_calls[0].provider_data = {PROVIDER: replay}
    text = "".join(block.get("text", "") for block in content if isinstance(block, dict) and "text" in block)
    return ModelResponse(
        output_text=text or None,
        tool_calls=tool_calls,
        raw=body,
        usage=_usage(body.get("usage") or {}),
        request_id=request_id,
        status_code=status_code,
        headers=headers,
        provider=PROVIDER,
        model=model,
        stop_reason=_STOP_REASONS.get(raw_stop, raw_stop),
    )


def _usage(raw: dict[str, Any]) -> Usage:
    def number(key: str) -> int:
        value = raw.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    output_tokens = number("outputTokens")
    cache_read = number("cacheReadInputTokens")
    cache_write = number("cacheWriteInputTokens")
    total = number("totalTokens")
    # Normalize to "input includes cached tokens". totalTokens tells us whether
    # inputTokens already counts them, without assuming per-model semantics.
    input_tokens = total - output_tokens if total else number("inputTokens") + cache_read + cache_write
    input_tokens = max(input_tokens, number("inputTokens"))
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total or input_tokens + output_tokens,
        cached_input_tokens=cache_read,
        details={"raw": raw} if raw else {},
    )


def _header(headers: dict[str, str], name: str) -> str | None:
    return next((value for key, value in headers.items() if key.lower() == name), None)


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


class _StreamAccumulator:
    """Rebuild a Converse response from ConverseStream events, emitting deltas."""

    def __init__(self, *, headers: dict[str, str], status_code: int, model: str, binding: str | None) -> None:
        self.headers = headers
        self.status_code = status_code
        self.model = model
        self.binding = binding
        self.blocks: dict[int, dict[str, Any]] = {}
        self.tool_json: dict[int, str] = {}
        self.stop_reason: str | None = None
        self.usage: dict[str, Any] = {}

    def feed(self, event: dict[str, Any]) -> list[ModelStreamEvent]:
        kind = event.get("type")
        if kind == "contentBlockStart":
            index = event.get("contentBlockIndex", len(self.blocks))
            tool = (event.get("start") or {}).get("toolUse")
            if tool:
                self.blocks[index] = {"toolUse": {"toolUseId": tool.get("toolUseId", ""), "name": tool.get("name", ""), "input": {}}}
                self.tool_json[index] = ""
                return [self._tool_delta(index, "", event)]
        elif kind == "contentBlockDelta":
            return self._delta(event)
        elif kind == "contentBlockStop":
            self._finish_tool(event.get("contentBlockIndex", 0))
        elif kind == "messageStop":
            self.stop_reason = event.get("stopReason")
        elif kind == "metadata":
            self.usage = event.get("usage") or {}
        return []

    def _delta(self, event: dict[str, Any]) -> list[ModelStreamEvent]:
        index = event.get("contentBlockIndex", 0)
        delta = event.get("delta") or {}
        if "text" in delta:
            block = self.blocks.setdefault(index, {"text": ""})
            block["text"] = block.get("text", "") + delta["text"]
            if delta["text"]:
                return [ModelStreamEvent(type=ModelStreamEventType.TEXT_DELTA, text_delta=delta["text"], raw=event)]
        elif "toolUse" in delta:
            fragment = (delta["toolUse"] or {}).get("input", "")
            self.tool_json[index] = self.tool_json.get(index, "") + fragment
            if fragment:
                return [self._tool_delta(index, fragment, event)]
        elif "reasoningContent" in delta:
            reasoning = delta["reasoningContent"] or {}
            block = self.blocks.setdefault(index, {"reasoningContent": {}})["reasoningContent"]
            if "redactedContent" in reasoning:
                block["redactedContent"] = block.get("redactedContent", "") + reasoning["redactedContent"]
            else:
                text_block = block.setdefault("reasoningText", {"text": ""})
                text_block["text"] = text_block.get("text", "") + reasoning.get("text", "")
                if reasoning.get("signature"):
                    text_block["signature"] = text_block.get("signature", "") + reasoning["signature"]
        return []

    def _tool_delta(self, index: int, fragment: str, event: dict[str, Any]) -> ModelStreamEvent:
        tool = self.blocks[index]["toolUse"]
        return ModelStreamEvent(
            type=ModelStreamEventType.TOOL_CALL_DELTA,
            tool_call=ToolCall(tool_call_id=tool["toolUseId"], name=tool["name"], arguments=fragment),
            raw=event,
        )

    def _finish_tool(self, index: int) -> None:
        if index not in self.tool_json:
            return
        text = self.tool_json.pop(index)
        try:
            self.blocks[index]["toolUse"]["input"] = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            raise ProviderError("Bedrock streamed invalid tool input JSON", response_body={"partial_json": text}) from exc

    def finish(self) -> ModelResponse:
        for index in list(self.tool_json):
            self._finish_tool(index)
        body = {
            "output": {"message": {"role": "assistant", "content": [self.blocks[i] for i in sorted(self.blocks)]}},
            "stopReason": self.stop_reason,
            "usage": self.usage,
        }
        return from_converse_payload(
            body, headers=self.headers, status_code=self.status_code, model=self.model, binding=self.binding
        )


def converse_stream_events(
    events: Iterable[dict[str, Any]], *, headers: dict[str, str], status_code: int, model: str, binding: str | None = None
):
    state = _StreamAccumulator(headers=headers, status_code=status_code, model=model, binding=binding)
    for event in events:
        yield from state.feed(event)
    yield ModelStreamEvent.completed(state.finish())


async def converse_stream_events_async(
    events: AsyncIterable[dict[str, Any]], *, headers: dict[str, str], status_code: int, model: str, binding: str | None = None
):
    state = _StreamAccumulator(headers=headers, status_code=status_code, model=model, binding=binding)
    async for event in events:
        for item in state.feed(event):
            yield item
    yield ModelStreamEvent.completed(state.finish())
