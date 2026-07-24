"""Provider-neutral input parts for multimodal model messages.

The parts intentionally describe *what* is being sent, rather than the wire
format of a particular provider. Provider adapters remain responsible for
turning them into their supported request payloads.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, TypeAlias

from modmex import BaseModel, model_validator


class InputDetail(str, Enum):
    """Requested visual-analysis fidelity for an image or document."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class TextInput(BaseModel):
    """A textual part of a multimodal message."""

    type: Literal["text"] = "text"
    text: str


class FileInput(BaseModel):
    """A document supplied by URL, provider file id, or inline base64 data.

    Exactly one source must be supplied. ``data`` may be raw bytes or base64
    content; adapters use ``media_type`` to build their wire representation.
    """

    type: Literal["file"] = "file"
    url: str | None = None
    file_id: str | None = None
    data: BinaryData | None = None
    filename: str | None = None
    media_type: str | None = None
    detail: InputDetail | None = None

    @model_validator(mode="after")
    def _validate_source(self, values: dict[str, object]) -> dict[str, object]:
        source_count = sum(
            values.get(name) is not None
            for name in ("url", "file_id", "data")
        )
        if source_count != 1:
            raise ValueError("FileInput requires exactly one of url, file_id, or data.")
        return values


class ImageInput(BaseModel):
    """An image supplied by URL or inline base64 data."""

    type: Literal["image"] = "image"
    url: str | None = None
    data: BinaryData | None = None
    media_type: str | None = None
    detail: InputDetail = InputDetail.AUTO

    @model_validator(mode="after")
    def _validate_source(self, values: dict[str, object]) -> dict[str, object]:
        source_count = sum(values.get(name) is not None for name in ("url", "data"))
        if source_count != 1:
            raise ValueError("ImageInput requires exactly one of url or data.")
        return values


BinaryData: TypeAlias = bytes | str
ContentInput: TypeAlias = TextInput | FileInput | ImageInput
