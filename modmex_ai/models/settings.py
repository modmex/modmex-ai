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


class AnthropicSettings(BaseModel):
    """Anthropic Messages API options that have no provider-neutral equivalent."""

    # Top-level automatic prompt caching. "5m" (default TTL) or "1h".
    cache_ttl: Literal["5m", "1h"] | None = None
    # Send ``thinking: {type: "adaptive"}`` explicitly. Current models (Opus 5.5,
    # Sonnet 5.5, Haiku 5.5) already think adaptively when it is omitted; Opus
    # 4.7/4.8 need it set to think at all.
    adaptive_thinking: bool = False
    # Visibility of thinking text; the API default on current models is "omitted".
    thinking_display: Literal["summarized", "omitted"] | None = None
    # Server-side refusal fallback: "default" or explicit model ids.
    fallbacks: Literal["default"] | list[str] | None = None
    # Extra ``anthropic-beta`` values.
    betas: list[str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.fallbacks, list) and not self.fallbacks:
            raise ValueError("fallbacks must be 'default' or a non-empty list of model ids")
        super().__post_init__()


class ModelSettings(BaseModel):
    openai: OpenAISettings | None = None
    anthropic: AnthropicSettings | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
