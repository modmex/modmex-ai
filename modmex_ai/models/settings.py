from __future__ import annotations

from modmex import BaseModel
from typing import Literal


class OpenAISettings(BaseModel):
    compact_threshold: int | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "24h"] | None = None

    def __post_init__(self) -> None:
        threshold = self.compact_threshold
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, int)
            or threshold <= 0
        ):
            raise ValueError("compact_threshold must be a positive integer token count")
        super().__post_init__()


class ModelSettings(BaseModel):
    openai: OpenAISettings | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
