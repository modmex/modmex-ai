from __future__ import annotations

from typing import Any

from modmex import BaseModel


class GuardrailResult(BaseModel):
    passed: bool
    output_info: Any = None
    reason: str | None = None
