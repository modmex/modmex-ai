"""Read ``s3://`` objects for model inputs (images and documents).

Bedrock models (except Amazon Nova) only accept file bytes in the request, so an
``s3://`` reference stored in a session is resolved to bytes when a request is
built. Objects are cached briefly per container: an agent's tool loop resends the
same history several times within one invocation. Keys are assumed immutable for
the cache TTL; write a new key instead of overwriting one in place.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any
from urllib.parse import unquote, urlparse


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected s3://bucket/key, got {uri!r}")
    return parsed.netloc, unquote(parsed.path.lstrip("/"))


class S3ObjectLoader:
    """Fetch S3 objects with botocore (credentials from the same chain as signing)."""

    def __init__(
        self,
        *,
        session: Any = None,
        region: str | None = None,
        client: Any = None,
        max_cache_bytes: int = 64 * 1024 * 1024,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._session = session
        self.region = region
        self._client = client
        self.max_cache_bytes = max_cache_bytes
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._cache_bytes = 0

    @property
    def client(self) -> Any:
        if self._client is None:
            session = self._session
            if session is None:
                from botocore.session import Session

                session = Session()
            self._client = session.create_client("s3", region_name=self.region)
        return self._client

    def load(self, uri: str) -> bytes:
        now = time.monotonic()
        cached = self._cache.get(uri)
        if cached is not None and now - cached[0] < self.cache_ttl_seconds:
            self._cache.move_to_end(uri)
            return cached[1]
        bucket, key = parse_s3_uri(uri)
        data = self.client.get_object(Bucket=bucket, Key=key)["Body"].read()
        self._remember(uri, now, data)
        return data

    def _remember(self, uri: str, now: float, data: bytes) -> None:
        if len(data) > self.max_cache_bytes:
            return
        previous = self._cache.pop(uri, None)
        if previous is not None:
            self._cache_bytes -= len(previous[1])
        self._cache[uri] = (now, data)
        self._cache_bytes += len(data)
        while self._cache_bytes > self.max_cache_bytes:
            _, (_, evicted) = self._cache.popitem(last=False)
            self._cache_bytes -= len(evicted)
