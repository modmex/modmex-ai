from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

from modmex_ai.errors import ProviderError
from modmex_ai.budget import remaining_seconds


class RetryConfig:
    def __init__(
        self,
        *,
        attempts: int = 2,
        backoff_seconds: float = 0.25,
        status_codes: set[int] | None = None,
        jitter_seconds: float = 0.0,
        on_retry: Callable[[dict], None] | None = None,
    ) -> None:
        if attempts < 0 or not isinstance(attempts, int):
            raise ValueError("attempts must be a non-negative integer")
        if any(
            not math.isfinite(value) or value < 0
            for value in (backoff_seconds, jitter_seconds)
        ):
            raise ValueError("Retry delays must be finite and non-negative")
        self.attempts = attempts
        self.backoff_seconds = backoff_seconds
        self.status_codes = (
            {408, 409, 429, 500, 502, 503, 504}
            if status_codes is None
            else status_codes
        )
        self.jitter_seconds = jitter_seconds
        self.on_retry = on_retry

    def delay(self, error: ProviderError, attempt: int) -> float | None:
        """Return a safe retry delay, or None when the request must stop."""
        body = error.response_body
        detail = body.get("error", {}) if isinstance(body, dict) else {}
        if isinstance(detail, dict) and any(
            detail.get(key) in {"insufficient_quota", "billing_hard_limit_reached"}
            for key in ("code", "type")
        ):
            return None
        if error.status_code not in self.status_codes or attempt >= self.attempts:
            return None
        delay = self.backoff_seconds * (2**attempt)
        value = next(
            (v for k, v in error.headers.items() if k.lower() == "retry-after"), None
        )
        if value is not None:
            try:
                requested = float(value)
            except (ValueError, TypeError):
                try:
                    requested = (
                        parsedate_to_datetime(value) - datetime.now(timezone.utc)
                    ).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    requested = 0
            if math.isfinite(requested):
                delay = max(delay, requested)
        delay += random.uniform(0, self.jitter_seconds)
        remaining = remaining_seconds()
        if remaining is not None and delay >= remaining:
            return None
        if self.on_retry is not None:
            try:
                self.on_retry(
                    {
                        "attempt": attempt + 1,
                        "delay_seconds": delay,
                        "status_code": error.status_code,
                        "request_id": error.request_id,
                    }
                )
            except Exception:
                pass  # Metrics must never change request execution.
        return delay
