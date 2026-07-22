from __future__ import annotations

from modmex import BaseModel


class ModelSettings(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
