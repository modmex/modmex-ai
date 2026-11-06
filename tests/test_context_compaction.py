import asyncio

import pytest

from modmex_ai import (
    ContextCompactor,
    CompactionSnapshot,
    FakeModel,
    Message,
    ModelResponse,
    SessionItem,
    Usage,
    context_bytes,
    execution_budget,
    RunTimeoutError,
)
from modmex_ai.tracing import Trace


def history(turns=8):
    return [
        Message(role="system", content="Keep this instruction verbatim."),
        *[
            Message(role=role, content=f"turn-{i}-{role} " + "long content " * 30)
            for i in range(turns)
            for role in ("user", "assistant")
        ],
    ]


def compactor(model, **kwargs):
    return ContextCompactor(
        model=model, threshold=1500, keep_recent_turns=2, max_input_size=20000, **kwargs
    )


@pytest.mark.parametrize("native_async", [False, True])
def test_compacts_preserves_recent_turns_and_instructions(native_async):
    model = FakeModel(
        [
            ModelResponse(
                output_text="Load A is pending; arrival is unconfirmed.",
                usage=Usage(input_tokens=100, output_tokens=20, total_tokens=120),
            )
        ]
    )
    trace = Trace()
    original = history()
    if native_async:

        async def acomplete(request):
            return model.complete(request)

        model.acomplete = acomplete
        result = asyncio.run(compactor(model).compact_async(original, trace=trace))
    else:
        result = compactor(model).compact(original, trace=trace)
    assert result.status == "compacted"
    assert result.snapshot.covered_items == 13
    assert result.messages[0] is original[0]
    assert result.messages[-4:] == original[-4:]
    assert result.messages[1].role == "user"
    assert "unconfirmed" in result.snapshot.summary
    assert result.usage.total_tokens == 120
    assert result.after_size < result.before_size
    assert trace.steps[-1].type == "context_compaction"
    assert "Load A" not in str(trace.steps[-1].data)
    assert len(original) == 17


def test_incremental_summary_does_not_repeat_covered_history_and_roundtrips():
    model = FakeModel(["First summary.", "Merged summary."])
    service = compactor(model)
    old = history()
    first = service.compact(old)
    snapshot = CompactionSnapshot(**first.snapshot.model_dump())
    assert service.compact(old, previous=snapshot).status in {"unchanged", "blocked"}
    assert len(model.requests) == 1
    extended = history(12)
    second = service.compact(extended, previous=snapshot)
    assert second.status == "compacted"
    payload = model.requests[-1].messages[-1].content
    assert "First summary." in payload
    assert "turn-0-user" not in payload
    assert "turn-6-user" in payload
    assert second.snapshot.covered_items > first.snapshot.covered_items
    extended[1] = Message(role="user", content="edited")
    with pytest.raises(ValueError, match="does not match"):
        service.compact(extended, previous=snapshot)


def test_parallel_tool_calls_and_outputs_never_split_at_turn_boundary():
    items = history(4)
    calls = [
        SessionItem(
            type="function_call", tool_call_id=str(i), name="lookup", arguments={}
        )
        for i in (1, 2)
    ]
    outputs = [
        SessionItem(type="function_call_output", tool_call_id=str(i), output="ok")
        for i in (2, 1)
    ]
    # Calls straddle a user boundary. Keep both calls and outputs raw.
    items = [*items[:5], *calls, *items[5:], *outputs]
    service = ContextCompactor(
        model=FakeModel(["Brief summary."]), threshold=100, keep_recent_turns=1
    )
    result = service.compact(items)
    assert result.status == "compacted"
    assert all(item in result.messages for item in [*calls, *outputs])
    assert result.snapshot.covered_items == 3


def test_pending_tool_call_retained_and_orphan_output_rejected():
    items = history()
    pending = SessionItem(
        type="function_call", tool_call_id="pending", name="save", arguments={}
    )
    items.insert(5, pending)
    result = compactor(FakeModel(["Brief summary."])).compact(items)
    assert pending in result.messages
    assert result.snapshot.covered_items <= 5
    items.insert(
        1,
        SessionItem(type="function_call_output", tool_call_id="unknown", output="bad"),
    )
    with pytest.raises(ValueError, match="Orphan"):
        compactor(FakeModel([])).compact(items)


@pytest.mark.parametrize("summary", ["", " " * 3, "x" * 6001])
def test_invalid_summary_keeps_original_context(summary):
    items = history()
    result = compactor(FakeModel([summary])).compact(items)
    assert result.status == "failed"
    assert result.messages == items
    assert result.snapshot is None


def test_provider_failure_keeps_previous_summary_and_unsummarized_items():
    items = history()
    initial = compactor(FakeModel(["Existing summary."])).compact(items)

    class BrokenModel(FakeModel):
        def complete(self, request):
            raise RuntimeError("secret internal endpoint")

    result = compactor(BrokenModel([])).compact(history(12), previous=initial.snapshot)
    assert result.status == "failed"
    assert result.reason == "RuntimeError"
    assert result.snapshot == initial.snapshot
    assert result.messages[-4:] == history(12)[-4:]
    assert "secret" not in str(result.model_dump())


def test_oversized_turn_is_not_silently_truncated_and_small_context_skips_model():
    model = FakeModel([])
    service = ContextCompactor(
        model=model, threshold=100, max_input_size=10, keep_recent_turns=1
    )
    items = history()
    result = service.compact(items)
    assert result.status == "blocked"
    assert result.messages == items
    assert not model.requests
    small = compactor(model).compact([Message(role="user", content="hi")])
    assert small.status == "unchanged"
    assert not model.requests


def test_custom_measure_preservation_instructions_and_bounded_batch():
    model = FakeModel(["Brief summary."])
    service = ContextCompactor(
        model=model,
        threshold=100,
        keep_recent_turns=1,
        max_input_size=2500,
        measure=lambda items: context_bytes(items) // 2,
        preservation_instructions="Preserve assignment identifiers.",
    )
    result = service.compact(history())
    assert result.status == "compacted"
    assert 0 < result.snapshot.covered_items < 15
    assert "assignment identifiers" in model.requests[0].messages[0].content
    assert context_bytes(model.requests[0].messages) // 2 <= 2500


def test_async_budget_and_external_cancellation_propagate():
    class SlowModel(FakeModel):
        async def acomplete(self, request):
            await asyncio.sleep(10)

    async def timed():
        with execution_budget(0.01):
            await compactor(SlowModel([])).compact_async(history())

    with pytest.raises(RunTimeoutError):
        asyncio.run(timed())

    async def cancelled():
        task = asyncio.create_task(compactor(SlowModel([])).compact_async(history()))
        await asyncio.sleep(0)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0},
        {"keep_recent_turns": 0},
        {"max_input_size": -1},
        {"max_summary_bytes": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ContextCompactor(model=FakeModel([]), **kwargs)


def test_completed_tool_block_can_be_summarized_and_developer_instruction_survives():
    items = history()
    pinned = Message(role="developer", content="Authoritative permission boundary")
    call = SessionItem(
        type="function_call", tool_call_id="lookup-1", name="lookup", arguments={}
    )
    output = SessionItem(
        type="function_call_output", tool_call_id="lookup-1", output={"ok": True}
    )
    items[3:3] = [pinned, call, output]
    model = FakeModel(["Lookup succeeded; other requests remain pending."])
    result = compactor(model).compact(items)
    assert result.status == "compacted"
    assert pinned in result.messages
    assert call not in result.messages and output not in result.messages
    payload = model.requests[0].messages[-1].content
    assert "lookup-1" in payload
    assert "Authoritative permission boundary" not in payload


def test_non_reducing_summary_keeps_original_snapshot_and_reports_usage():
    items = history(2)
    model = FakeModel(
        [ModelResponse(output_text="x" * 2000, usage=Usage(total_tokens=99))]
    )
    service = ContextCompactor(model=model, threshold=100, keep_recent_turns=1)
    result = service.compact(items)
    assert result.status == "blocked"
    assert result.reason == "no_size_reduction"
    assert result.messages == items
    assert result.snapshot is None
    assert result.usage.total_tokens == 99
