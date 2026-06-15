from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any

from modmex import BaseModel

from modmex_ai.errors import ModelTimeoutError, ProviderError, RateLimitError
from modmex_ai.http.retries import RetryConfig


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
    def __init__(self, *, retry: RetryConfig | None = None, timeout: float = 30.0) -> None:
        self.retry = retry or RetryConfig()
        self.timeout = timeout

    def post_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        payload = json.dumps(data or {}, separators=(",", ":")).encode("utf-8")
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
                return self._send(request, timeout=timeout or self.timeout)
            except ProviderError as exc:
                last_error = exc
                if exc.status_code not in self.retry.status_codes or attempt >= self.retry.attempts:
                    raise
                time.sleep(self.retry.backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error

    def post_json_bytes(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> HttpResponse:
        payload = json.dumps(data or {}, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"content-type": "application/json", **(headers or {})},
            method="POST",
        )
        return self._send(request, timeout=timeout or self.timeout, decode_json=False)

    def post_json_stream(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        data: dict[str, Any] | None = None,
        chunk_size: int = 8_192,
        timeout: float | None = None,
    ) -> Iterator[bytes]:
        """Yield a chunked binary HTTP response without buffering it in memory."""
        payload = json.dumps(data or {}, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"content-type": "application/json", **(headers or {})},
            method="POST",
        )
        return self._stream(request, timeout=timeout or self.timeout, chunk_size=chunk_size)

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
        return self._send(request, timeout=timeout or self.timeout)

    def _send(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        decode_json: bool = True,
    ) -> HttpResponse:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
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
                f"Provider returned HTTP {exc.code}",
                status_code=exc.code,
                request_id=request_id,
                response_body=body,
                headers=headers,
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderError(str(exc)) from exc

    def _stream(
        self,
        request: urllib.request.Request,
        *,
        timeout: float,
        chunk_size: int,
    ) -> Iterator[bytes]:
        last_error: ProviderError | None = None
        for attempt in range(self.retry.attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    while chunk := response.read(chunk_size):
                        yield chunk
                return
            except TimeoutError as exc:
                last_error = ModelTimeoutError("Model request timed out")
            except urllib.error.HTTPError as exc:
                last_error = _provider_error_from_http_error(exc)
            except urllib.error.URLError as exc:
                last_error = ProviderError(str(exc))
            if last_error.status_code not in self.retry.status_codes or attempt >= self.retry.attempts:
                raise last_error
            time.sleep(self.retry.backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error


def _multipart_body(
    boundary: str,
    fields: dict[str, str],
    files: dict[str, HttpFile],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
            b"\r\n",
        ])
    for name, file in files.items():
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{file.filename}"\r\n'
            ).encode(),
            f"Content-Type: {file.content_type}\r\n\r\n".encode(),
            file.data,
            b"\r\n",
        ])
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
        f"Provider returned HTTP {error.code}",
        status_code=error.code,
        request_id=request_id,
        response_body=body,
        headers=headers,
    )
