from __future__ import annotations

from dataclasses import field
from typing import Any

from modmex import BaseModel

from modmex_ai.tracing import Trace


class RunContext(BaseModel):
    input: Any
    context: Any = None
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: Trace = field(default_factory=Trace)
