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


class ProviderConnectionError(ProviderError):
    """The request never got an HTTP response (connection reset, DNS, TLS...).

    ``request_sent`` is False only when the failure happened before the request
    could reach the server (e.g. connection refused, DNS), so retrying cannot
    repeat a side effect. Model calls retry both cases; non-idempotent calls
    such as MCP ``tools/call`` retry only unsent requests.
    """

    def __init__(self, message: str, *, request_sent: bool = True, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.request_sent = request_sent


class ModelRefusalError(ProviderError):
    """The model declined the request (e.g. Anthropic ``stop_reason: refusal``)."""

    def __init__(self, message: str, *, stop_details: object | None = None, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.stop_details = stop_details


class ModelTimeoutError(ProviderError):
    pass


class RunTimeoutError(ModmexAIError):
    """The execution deadline expired; writes may have completed before timeout."""


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
