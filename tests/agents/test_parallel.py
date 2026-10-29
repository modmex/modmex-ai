import asyncio

import pytest

from modmex_ai import Agent
from modmex_ai.models import FakeModel, ModelResponse, ToolCall
from modmex_ai.tools.tool import _current_tool_call_id


@pytest.mark.parametrize("streaming", [False, True])
def test_parallel_tools_overlap_and_preserve_call_order(streaming):
    async def run():
        started = set()
        ready = asyncio.Event()
        finished = []

        async def lookup(value: int):
            started.add(value)
            if len(started) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 1)
            assert _current_tool_call_id.get() == str(value)
            if value == 1:
                await asyncio.sleep(0.01)
            finished.append(value)
            return value

        class StreamingFakeModel(FakeModel):
            async def astream(self, request):
                yield self.complete(request)

        agent = Agent(
            name="parallel", instructions="", tools=[lookup], parallel_tool_calls=True,
            model=StreamingFakeModel([
                ModelResponse(tool_calls=[ToolCall(tool_call_id=str(i), name="lookup", arguments={"value": i}) for i in (1, 2)]),
                ModelResponse(output_text="done"),
            ]),
        )
        if streaming:
            events = [event async for event in agent.arun_stream("go")]
            result = events[-1].result
        else:
            result = await agent.run_async("go")
        assert finished == [2, 1]
        outputs = [item for item in result.items if item.type == "function_call_output"]
        assert [item.tool_call_id for item in outputs] == ["1", "2"]

    asyncio.run(run())


def test_parallel_failure_cancels_other_tools():
    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def waiting():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def failing():
            await started.wait()
            raise ValueError("failed")

        agent = Agent(name="parallel", instructions="", tools=[waiting, failing], parallel_tool_calls=True,
                      model=FakeModel([ModelResponse(tool_calls=[
                          ToolCall(tool_call_id="1", name="waiting", arguments={}),
                          ToolCall(tool_call_id="2", name="failing", arguments={}),
                      ])]))
        with pytest.raises(Exception, match="failed"):
            await agent.run_async("go")
        assert stopped.is_set()

    asyncio.run(run())
