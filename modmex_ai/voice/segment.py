"""Split streamed model text into speakable segments for incremental TTS."""

from __future__ import annotations

import re

# End of a sentence: terminal punctuation (optionally followed by closing quotes or
# brackets) and then whitespace. "3.5" or "e.g.x" do not end a sentence; a line
# break always does.
_SENTENCE_END = re.compile(r"""[.!?\u2026]+["'\u201d\u2019)\]]*\s+|\n+""")


class SentenceBuffer:
    """Accumulate text deltas and release complete sentences.

    ``max_chars`` bounds how long a run-on sentence may delay speech: past it,
    the buffer is cut at the last comma, semicolon or space. ``min_chars``
    keeps very short fragments ("Ok.") together with the next sentence so the
    TTS provider is not called for a single word.
    """

    def __init__(self, *, min_chars: int = 12, max_chars: int = 240) -> None:
        if min_chars < 0 or max_chars <= min_chars:
            raise ValueError("Require 0 <= min_chars < max_chars")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._text = ""

    def feed(self, delta: str) -> list[str]:
        self._text += delta
        ready: list[str] = []
        while True:
            cut = self._next_cut()
            if cut is None:
                return ready
            segment, self._text = self._text[:cut].strip(), self._text[cut:]
            if segment:
                ready.append(segment)

    def flush(self) -> str | None:
        segment, self._text = self._text.strip(), ""
        return segment or None

    def _next_cut(self) -> int | None:
        for match in _SENTENCE_END.finditer(self._text):
            if match.end() >= self.min_chars:
                return match.end()
        if len(self._text) > self.max_chars:
            window = self._text[: self.max_chars]
            for separator in (", ", "; ", ": ", " "):
                index = window.rfind(separator)
                if index > self.min_chars:
                    return index + len(separator)
            return self.max_chars
        return None
