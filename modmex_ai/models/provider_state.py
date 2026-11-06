from __future__ import annotations

from typing import Literal
from modmex import BaseModel


class ProviderState(BaseModel):
    provider: str | None = None
    conversation_id: str | None = None
    previous_response_id: str | None = None
    mode: Literal["stateless", "response_chain", "conversation"] | None = None

    @property
    def resolved_mode(self) -> str:
        if self.conversation_id and self.previous_response_id:
            raise ValueError(
                "conversation_id and previous_response_id are mutually exclusive"
            )
        mode = self.mode or (
            "conversation"
            if self.conversation_id
            else "response_chain"
            if self.previous_response_id
            else "stateless"
        )
        if mode == "stateless" and (self.conversation_id or self.previous_response_id):
            raise ValueError("Stateless mode cannot contain remote identifiers")
        if mode == "conversation" and (
            not self.conversation_id or self.previous_response_id
        ):
            raise ValueError("Conversation mode requires only conversation_id")
        if mode == "response_chain" and self.conversation_id:
            raise ValueError("Response chain cannot contain conversation_id")
        return mode

    @property
    def has_remote_state(self) -> bool:
        return self.resolved_mode in {"conversation", "response_chain"}
