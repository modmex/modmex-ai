from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from modmex import BaseModel

from modmex_ai.errors import (
    ModelTimeoutError,
    ProviderConnectionError,
    ProviderError,
    RateLimitError,
)
from modmex_ai.http.retries import RetryConfig
from modmex_ai.budget import bounded_timeout, remaining_seconds


def encode_json_body(data: dict[str, Any] | bytes | None) -> bytes:
    """JSON request body. Pre-encoded bytes are sent as-is (e.g. a body that was
    signed with SigV4 must reach the wire byte-for-byte)."""
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return json.dumps(data or {}, separators=(",", ":")).encode("utf-8")


def provider_error_message(status: int, headers: dict[str, str], body: Any) -> str:
    """"Provider returned HTTP 403: AccessDeniedException: <provider message>".

    The provider's own error type and message are what explain a failure (wrong
    credentials, missing permission, invalid model id); keep them in the
    exception text instead of only the status code.
    """
    parts = [f"Provider returned HTTP {status}"]
    error_type = next((v for k, v in headers.items() if k.lower() == "x-amzn-errortype"), None)
    if error_type:
        parts.append(error_type.split(":", 1)[0])
    message = None
    if isinstance(body, dict):
        error = body.get("error")
        message = body.get("message") or body.get("Message")
        if message is None and isinstance(error, dict):
            message = error.get("message") or error.get("type")
        elif message is None and isinstance(error, str):
            message = error
    elif isinstance(body, str) and body.strip():
        message = body.strip()
    if message:
        parts.append(" ".join(str(message).split())[:300])
    return ": ".join(parts)


class HttpResponse(BaseModel):
    status_code: int
    headers: dict[str, str]
    body: Any


@dataclass(frozen=True)
class HttpFile:
    """One binary part in a multipart HTTP request."""

    data: bytes
    filename: str
    content_type: str = "application/octet-stream"


class HttpClient:
    def __init__(
        self, *, retry: RetryConfig | None = None, timeout: float = 30.0
    ) -> None:
        self.retry = retry or RetryConfig()
        self.timeout = timeout

    def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | bytes | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        payload = encode_json_body(data)
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "content-type": "application/json",
                **(headers or {}),
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retry.attempts + 1):
            try:
                return self._send(
                    request,
                    timeout=bounded_timeout(
                        timeout if timeout is not None else self.timeout
                    ),
                )
            except ProviderError as exc:
                last_error = exc
                delay = self.retry.delay(exc, attempt)
                if delay is None:
                    raise
                time.sleep(delay)
                remaining_seconds()
        assert last_error is not None
        raise last_error

    def post_json_bytes(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | bytes | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        payload = encode_json_body(data)
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"content-type": "application/json", **(headers or {})},
            method="POST",
        )
        return self._send(
            request,
            timeout=bounded_timeout(timeout if timeout is not None else self.timeout),
            decode_json=False,
        )

    def post_json_stream(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | bytes | None = None,
        chunk_size: int = 8_192,
        timeout: float | None = None,
    ) -> Iterator[bytes]:
        """Yield a chunked binary HTTP response without buffering it in memory."""
        payload = encode_json_body(data)
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"content-type": "application/json", **(headers or {})},
            method="POST",
        )
        return self._stream(
            request,
            timeout=bounded_timeout(timeout if timeout is not None else self.timeout),
            chunk_size=chunk_size,
        )

    def post_multipart(
        self,
        url: str,
        *,
        fields: dict[str, str] | None = None,
        files: dict[str, HttpFile] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        boundary = "----modmex-ai-boundary"
        body = _multipart_body(boundary, fields or {}, files or {})
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "content-type": f"multipart/form-data; boundary={boundary}",
                **(headers or {}),
            },
            method="POST",
        )
        return self._send(
            request,
            timeout=bounded_timeout(timeout if timeout is not None else self.timeout),
        )

    def _send(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        decode_json: bool = True,
    ) -> HttpResponse:
        try:
            with urllib.request.urlopen(
                request, timeout=bounded_timeout(timeout)
            ) as response:
                headers = dict(response.headers.items())
                raw_body = response.read()
                return HttpResponse(
                    status_code=response.status,
                    headers=headers,
                    body=(json.loads(raw_body.decode("utf-8")) if raw_body else {})
                    if decode_json
                    else raw_body,
                )
        except TimeoutError as exc:
            raise ModelTimeoutError("Model request timed out") from exc
        except urllib.error.HTTPError as exc:
            headers = dict(exc.headers.items())
            body_text = exc.read().decode("utf-8")
            try:
                body = json.loads(body_text)
            except json.JSONDecodeError:
                body = body_text
            request_id = headers.get("x-request-id") or headers.get("request-id")
            error_cls = RateLimitError if exc.code == 429 else ProviderError
            raise error_cls(
                provider_error_message(exc.code, headers, body),
                status_code=exc.code,
                request_id=request_id,
                response_body=body,
                headers=headers,
            ) from exc
        except urllib.error.URLError as exc:
            raise _url_error(exc) from exc
        except ConnectionError as exc:  # reset/aborted while reading the response
            raise ProviderConnectionError(str(exc)) from exc

    def _stream(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        chunk_size: int,
    ) -> Iterator[bytes]:
        last_error: ProviderError | None = None
        emitted = False
        for attempt in range(self.retry.attempts + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=bounded_timeout(timeout)
                ) as response:
                    while chunk := response.read(chunk_size):
                        emitted = True
                        yield chunk
                return
            except TimeoutError as exc:
                last_error = ModelTimeoutError("Model request timed out")
            except urllib.error.HTTPError as exc:
                last_error = _provider_error_from_http_error(exc)
            except urllib.error.URLError as exc:
                last_error = _url_error(exc)
            except ConnectionError as exc:
                last_error = ProviderConnectionError(str(exc))
            delay = None if emitted else self.retry.delay(last_error, attempt)
            if delay is None:
                raise last_error
            time.sleep(delay)
            remaining_seconds()
        assert last_error is not None
        raise last_error


def _url_error(error: urllib.error.URLError) -> ProviderError:
    if isinstance(error.reason, TimeoutError):
        return ModelTimeoutError("Model request timed out")
    # urlopen raises URLError while connecting (refused, DNS, TLS handshake),
    # before the request body reaches the server.
    return ProviderConnectionError(str(error), request_sent=False)


def _multipart_body(
    boundary: str,
    fields: dict[str, str],
    files: dict[str, HttpFile],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    for name, file in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{file.filename}"\r\n'
                ).encode(),
                f"Content-Type: {file.content_type}\r\n\r\n".encode(),
                file.data,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


def _provider_error_from_http_error(error: urllib.error.HTTPError) -> ProviderError:
    headers = dict(error.headers.items())
    body_text = error.read().decode("utf-8")
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError:
        body = body_text
    request_id = headers.get("x-request-id") or headers.get("request-id")
    error_cls = RateLimitError if error.code == 429 else ProviderError
    return error_cls(
        provider_error_message(error.code, headers, body),
        status_code=error.code,
        request_id=request_id,
        response_body=body,
        headers=headers,
    )
