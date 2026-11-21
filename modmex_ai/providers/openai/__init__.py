"""OpenAI adapters, imported lazily so text models don't load realtime/voice."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    'OpenAIChatModel': 'modmex_ai.providers.openai.chat',
    'OpenAISpeechProvider': 'modmex_ai.providers.openai.audio',
    'OpenAITranscriptionProvider': 'modmex_ai.providers.openai.audio',
    'OpenAIRealtimeClient': 'modmex_ai.providers.openai.realtime',
    'OpenAIRealtimeSession': 'modmex_ai.providers.openai.realtime',
    'OpenAIRealtimeSessionConfig': 'modmex_ai.providers.openai.realtime',
    'OpenAIInputAudioTranscriptionConfig': 'modmex_ai.providers.openai.realtime',
    'OpenAIServerVadConfig': 'modmex_ai.providers.openai.realtime',
    'OpenAIResponsesModel': 'modmex_ai.providers.openai.responses',
    'OpenAIRealtimeTranscriptionConfig': 'modmex_ai.providers.openai.transcription',
    'OpenAIRealtimeTranscriptionProvider': 'modmex_ai.providers.openai.transcription',
    'OpenAIRealtimeTranscriptionSession': 'modmex_ai.providers.openai.transcription',
}

__all__ = [
    "OpenAIChatModel",
    "OpenAISpeechProvider",
    "OpenAITranscriptionProvider",
    "OpenAIRealtimeClient",
    "OpenAIRealtimeSession",
    "OpenAIRealtimeSessionConfig",
    "OpenAIInputAudioTranscriptionConfig",
    "OpenAIServerVadConfig",
    "OpenAIResponsesModel",
    "OpenAIRealtimeTranscriptionConfig",
    "OpenAIRealtimeTranscriptionProvider",
    "OpenAIRealtimeTranscriptionSession",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    from modmex_ai.providers.openai.chat import OpenAIChatModel
    from modmex_ai.providers.openai.audio import OpenAISpeechProvider, OpenAITranscriptionProvider
    from modmex_ai.providers.openai.realtime import (
        OpenAIRealtimeClient,
        OpenAIRealtimeSession,
        OpenAIRealtimeSessionConfig,
        OpenAIInputAudioTranscriptionConfig,
        OpenAIServerVadConfig,
    )
    from modmex_ai.providers.openai.responses import OpenAIResponsesModel
    from modmex_ai.providers.openai.transcription import (
        OpenAIRealtimeTranscriptionConfig,
        OpenAIRealtimeTranscriptionProvider,
        OpenAIRealtimeTranscriptionSession,
    )
