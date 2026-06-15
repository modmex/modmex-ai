from __future__ import annotations

from collections.abc import Iterable
from dataclasses import field
from time import perf_counter
from typing import Any

from modmex import BaseModel

from modmex_ai.flows import Flow, FlowResult


class EvalCase(BaseModel):
    """A reproducible flow input and its expected observable behavior."""

    name: str
    input: Any
    expected_output: Any = None
    expected_agent: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    max_latency_ms: int | None = None
    max_total_tokens: int | None = None


class EvalResult(BaseModel):
    case_name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    latency_ms: int = 0
    total_tokens: int = 0
    flow_result: FlowResult | None = None


class EvalRunner:
    """Runs deterministic conversation fixtures against a Flow."""

    def __init__(self, flow: Flow) -> None:
        self.flow = flow

    def run(self, cases: Iterable[EvalCase]) -> list[EvalResult]:
        return [self.run_case(case) for case in cases]

    def run_case(self, case: EvalCase) -> EvalResult:
        started = perf_counter()
        result = self.flow.run(case.input)
        failures = self._failures(case, result)
        latency_ms = int((perf_counter() - started) * 1000)
        if case.max_latency_ms is not None and latency_ms > case.max_latency_ms:
            failures.append("latency exceeded max_latency_ms")
        if case.max_total_tokens is not None and result.usage.total_tokens > case.max_total_tokens:
            failures.append("total tokens exceeded max_total_tokens")
        return EvalResult(
            case_name=case.name,
            passed=not failures,
            failures=failures,
            latency_ms=latency_ms,
            total_tokens=result.usage.total_tokens,
            flow_result=result,
        )

    def replay(self, previous: EvalResult) -> EvalResult:
        """Re-run a recorded fixture and preserve its assertions as the baseline."""
        if previous.flow_result is None:
            raise ValueError("Cannot replay an eval result without a flow result")
        input_items = previous.flow_result.input_items
        return self.run_case(EvalCase(
            name=f"{previous.case_name}:replay",
            input=[item.model_dump() for item in input_items],
            expected_output=previous.flow_result.output,
            expected_agent=previous.flow_result.last_agent_name,
        ))

    @staticmethod
    def _failures(case: EvalCase, result: FlowResult) -> list[str]:
        failures: list[str] = []
        if case.expected_output is not None and result.output != case.expected_output:
            failures.append("output did not match expected_output")
        if case.expected_agent is not None and result.last_agent_name != case.expected_agent:
            failures.append("last agent did not match expected_agent")
        invoked_tools = [item.name for agent in result.agent_results for item in agent.items if item.type == "function_call"]
        for tool_name in case.expected_tools:
            if tool_name not in invoked_tools:
                failures.append(f"expected tool was not called: {tool_name}")
        return failures
