"""Optional persistence adapters for modmex-ai."""

from importlib import import_module
from typing import Any

__all__ = ["DynamoDbDurableSessionStore", "DynamoDbFlowStateStore"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("modmex_ai.persistence.dynamodb"), name)
