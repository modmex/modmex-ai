import asyncio

import pytest
from modmex import BaseModel

from modmex_ai import ApprovalDecision, ApprovalDecisionType, ApprovalPolicy, ToolCall, tool
from modmex_ai.agents import RunContext
from modmex_ai.errors import ApprovalRejected, ToolExecutionError, ToolValidationError, UnknownToolError
from modmex_ai.tools import Tool, ToolExecutor, as_tool


def test_tool_schema_is_inferred_from_signature():
    @tool
    def search(term: str, limit: int = 10) -> list[str]:
        """Search things."""
        return [term] * limit

    schema = search.schema()

    assert schema["name"] == "search"
    assert schema["parameters"]["properties"]["term"]["type"] == "string"
    assert schema["parameters"]["properties"]["limit"]["type"] == "integer"
    assert schema["parameters"]["required"] == ["term"]
    assert "strict" not in schema


def test_tool_executor_rejects_unknown_tool():
    executor = ToolExecutor([])

    with pytest.raises(UnknownToolError):
        executor.execute(ToolCall(tool_call_id="call-1", name="missing", arguments={}))


def test_tool_rejects_missing_required_argument():
    @tool
    def search(term: str) -> str:
        return term

    with pytest.raises(ToolValidationError):
        search.run({})


def test_tool_accepts_string_arguments_and_defaults():
    @tool
    def search(term: str, limit: int = 2) -> list[str]:
        return [term] * limit

    assert search.run('{"term":"x"}') == ["x", "x"]
    assert search.run({"term": "x", "limit": None}) == ["x", "x"]


def test_tool_injects_context_and_ctx_parameters():
    @tool
    def with_context(value: str, context=None):
        return f"{context}:{value}"

    @tool
    def with_ctx(value: str, ctx=None):
        return f"{ctx}:{value}"

    assert with_context.run({"value": "a"}, context="context") == "context:a"
    assert with_ctx.run({"value": "b"}, context="ctx") == "ctx:b"


def test_tool_schema_skips_context_parameters():
    @tool
    def with_context(value: str, context=None):
        return value

    assert "context" not in with_context.schema()["parameters"]["properties"]


def test_tool_schema_hoists_modmex_model_definitions_to_parameters_root():
    class Address(BaseModel):
        city: str

    class Carrier(BaseModel):
        address: Address

    @tool
    def register_carrier(carrier: Carrier) -> str:
        return carrier.address.city

    parameters = register_carrier.schema()["parameters"]

    assert "$defs" not in parameters["properties"]["carrier"]
    assert parameters["properties"]["carrier"]["properties"]["address"] == {
        "$ref": "#/$defs/Address",
    }
    assert parameters["$defs"]["Address"]["properties"]["city"] == {
        "type": "string",
    }


def test_tool_decorator_accepts_options():
    @tool(name="custom", description="Custom description")
    def search(term: str) -> str:
        return term

    assert search.name == "custom"
    assert search.description == "Custom description"


def test_as_tool_returns_existing_tool():
    @tool
    def search(term: str) -> str:
        return term

    assert as_tool(search) is search


def test_as_tool_wraps_callable():
    def search(term: str) -> str:
        return term

    assert isinstance(as_tool(search), Tool)


def test_tool_execution_error_wraps_runtime_failures():
    @tool
    def broken() -> None:
        raise RuntimeError("boom")

    with pytest.raises(ToolExecutionError):
        broken.run({})


def test_tool_rejects_invalid_scalar_coercion():
    @tool
    def search(limit: int) -> int:
        return limit

    with pytest.raises(ToolValidationError):
        search.run({"limit": "not-int"})


class CarrierSignal(BaseModel):
    carrier_name: str
    rate_per_mile: float


def test_tool_validates_and_materializes_modmex_model_arguments():
    received = []

    @tool
    def capture(signal: CarrierSignal) -> str:
        received.append(signal)
        return signal.carrier_name

    assert capture.run({
        "signal": {"carrier_name": "Northstar", "rate_per_mile": 2.45},
    }) == "Northstar"
    assert received == [CarrierSignal(carrier_name="Northstar", rate_per_mile=2.45)]


def test_tool_rejects_unknown_or_invalid_modmex_arguments():
    @tool
    def capture(signal: CarrierSignal) -> str:
        return signal.carrier_name

    with pytest.raises(ToolValidationError):
        capture.run({"signal": {"carrier_name": "Northstar"}})
    with pytest.raises(ToolValidationError):
        capture.run({
            "signal": {"carrier_name": "Northstar", "rate_per_mile": 2.45},
            "unexpected": True,
        })


def test_tool_executor_executes_an_async_tool():
    @tool
    async def lookup(order_number: str) -> str:
        await asyncio.sleep(0)
        return f"found:{order_number}"

    async def run():
        result = await ToolExecutor([lookup]).execute_async(
            ToolCall(tool_call_id="call-async", name="lookup", arguments={"order_number": "A-1"})
        )
        assert result.output == "found:A-1"

    asyncio.run(run())


def test_tool_handles_async_sync_context_and_rejected_approval():
    @tool
    async def async_tool():
        return "ok"

    async def run_sync_tool_inside_loop():
        with pytest.raises(ToolExecutionError):
            async_tool.run({})

    asyncio.run(run_sync_tool_inside_loop())

    class Policy(ApprovalPolicy):
        def verify(self, request, decision):
            return True

    @tool(requires_approval=True)
    def protected():
        return "never"

    context = RunContext(input=None)
    context.state.update({
        "current_tool_call_id": "call",
        "approval_policy": Policy(),
        "approval_decisions": {"call": ApprovalDecision(request_id="call", decision=ApprovalDecisionType.REJECTED)},
    })
    with pytest.raises(ApprovalRejected):
        protected.run({}, context=context)
