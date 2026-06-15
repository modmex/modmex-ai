from __future__ import annotations


class RetryConfig:
    def __init__(
        self,
        *,
        attempts: int = 2,
        backoff_seconds: float = 0.25,
        status_codes: set[int] | None = None,
    ) -> None:
        self.attempts = attempts
        self.backoff_seconds = backoff_seconds
        self.status_codes = status_codes or {408, 409, 429, 500, 502, 503, 504}
