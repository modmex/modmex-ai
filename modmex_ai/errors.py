class ModmexAIError(Exception):
    """Base exception for modmex-ai."""


class ProviderError(ModmexAIError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        response_body: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body
        self.headers = headers or {}


class RateLimitError(ProviderError):
    pass


class ModelTimeoutError(ProviderError):
    pass


class ToolValidationError(ModmexAIError):
    pass


class ToolExecutionError(ModmexAIError):
    pass


class OutputValidationError(ModmexAIError):
    pass


class GuardrailTriggered(ModmexAIError):
    pass


class InputGuardrailTriggered(GuardrailTriggered):
    pass


class OutputGuardrailTriggered(GuardrailTriggered):
    pass


class ToolInputGuardrailTriggered(GuardrailTriggered):
    pass


class ToolOutputGuardrailTriggered(GuardrailTriggered):
    pass


class MaxToolCallsExceeded(ModmexAIError):
    pass


class MaxHandoffsExceeded(ModmexAIError):
    pass


class UnknownAgentError(ModmexAIError):
    pass


class UnknownToolError(ModmexAIError):
    pass


class RealtimeError(ModmexAIError):
    pass


class RealtimeConnectionError(RealtimeError):
    pass


class RealtimeProtocolError(RealtimeError):
    pass


class RealtimeProviderError(RealtimeError):
    pass


class VoiceTurnCancelled(ModmexAIError):
    """The host requested cancellation of the active chained voice turn."""


class ApprovalRequired(ModmexAIError):
    """A tool requires an external approval before it can be executed."""

    def __init__(self, request: object) -> None:
        super().__init__("Tool execution requires approval")
        self.request = request


class ApprovalRejected(ModmexAIError):
    """The approval policy rejected a requested tool execution."""
