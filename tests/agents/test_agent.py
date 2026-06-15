import asyncio

import pytest
from modmex import BaseModel

from modmex_ai import (
    Agent,
    FakeModel,
    Handoff,
    ModelResponse,
    RECOMMENDED_PROMPT_PREFIX,
    ToolCall,
    Usage,
    prompt_with_handoff_instructions,
    tool,
)
from modmex_ai.messages import Message
from modmex_ai.errors import MaxToolCallsExceeded, OutputValidationError


class RoutedOutput(BaseModel):
    intent: str


class SummaryOutput(BaseModel):
    summary: str


class HandoffInput(BaseModel):
    reason: str
    priority: str


def test_agent_run_async_uses_native_async_model_and_async_tool():
    class AsyncModel:
        name = "async-fake"
        profile = FakeModel([]).profile

        def __init__(self):
            self.requests = []

        async def acomplete(self, request):
            self.requests.append(request)
            await asyncio.sleep(0)
            if len(self.requests) == 1:
                return ModelResponse(tool_calls=[ToolCall(
                    tool_call_id="call-1",
                    name="lookup",
                    arguments={},
                )])
            return ModelResponse(output_text="done")

        def complete(self, request):
            raise AssertionError("sync complete must not be used")

        def stream(self, request):
            raise NotImplementedError

    async def lookup():
        await asyncio.sleep(0)
        return {"ok": True}

    async def run():
        model = AsyncModel()
        result = await Agent(
            name="async",
            instructions="Use the tool.",
            model=model,
            tools=[lookup],
        ).run_async("hello")
        assert result.output == "done"
        assert len(model.requests) == 2

    asyncio.run(run())


def test_agent_does_not_automatically_prefix_handoff_context():
    agent = Agent(
        name="triage",
        instructions="Route the request.",
        output_type=RoutedOutput,
        handoffs=["sales", "billing"],
    )

    messages = agent._initial_messages("hello")

    assert not messages[0].content.startswith(RECOMMENDED_PROMPT_PREFIX)
    assert "Route the request." in messages[0].content
    assert "Handoffs are achieved by calling a handoff function" not in messages[0].content
    assert "sales" not in messages[0].content
    assert "billing" not in messages[0].content


def test_prompt_with_handoff_instructions_prefixes_prompt():
    assert prompt_with_handoff_instructions("Route the request.") == (
        f"{RECOMMENDED_PROMPT_PREFIX}\n\nRoute the request."
    )


def test_agent_exposes_handoffs_as_tools_and_returns_handoff_result():
    captured = []
    agent = Agent(
        name="triage",
        instructions="Route the request.",
        handoffs=[
            Handoff(
                "support",
                description="Transfer to support.",
                input_type=HandoffInput,
                on_handoff=lambda _context, value: captured.append(value),
            )
        ],
    )
    model = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call-1",
                    name="transfer_to_support",
                    arguments={"reason": "Needs help", "priority": "high"},
                )
            ]
        )
    ])

    result = agent.run("hello", model=model)

    assert result.handoff_target == "support"
    assert result.handoff_name == "transfer_to_support"
    assert result.output is None
    assert result.handoff_input == HandoffInput(reason="Needs help", priority="high")
    assert captured == [HandoffInput(reason="Needs help", priority="high")]
    assert [item.type for item in result.items] == ["handoff_call", "handoff_call_output"]
    assert result.items[0].to_input() == {
        "type": "handoff_call",
        "tool_call_id": "call-1",
        "name": "transfer_to_support",
        "arguments": '{"reason":"Needs help","priority":"high"}',
    }
    assert model.requests[0].tools[0]["name"] == "transfer_to_support"
    assert model.requests[0].tools[0]["parameters"]["title"] == "HandoffInput"


def test_agent_accepts_explicit_role_messages():
    agent = Agent(name="agent", instructions="Continue conversation.")

    messages = agent._initial_messages([
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello"),
        {"role": "user", "content": "I want pizza"},
    ])

    assert [message.role for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1].content == "I want pizza"


def test_agent_public_context_is_available_to_tools():
    @tool
    def read_context(ctx=None):
        return ctx.context["customer_id"]

    agent = Agent(
        name="agent",
        instructions="Use tool.",
        tools=[read_context],
        output_type=RoutedOutput,
    )
    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(tool_call_id="call-1", name="read_context", arguments={})]),
        '{"intent":"support"}',
    ])

    result = agent.run("hello", model=model, context={"customer_id": "cust-1"})

    assert result.output.intent == "support"


def test_agent_accumulates_usage_across_tool_loop():
    @tool
    def lookup() -> str:
        return "done"

    agent = Agent(name="agent", instructions="Use tool.", tools=[lookup])
    model = FakeModel([
        ModelResponse(
            tool_calls=[ToolCall(tool_call_id="call-1", name="lookup", arguments={})],
            usage=Usage(input_tokens=4, output_tokens=2, total_tokens=6),
        ),
        ModelResponse(
            output_text="ok",
            usage=Usage(input_tokens=7, output_tokens=3, total_tokens=10),
        ),
    ])

    result = agent.run("hello", model=model)

    assert result.output == "ok"
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 16
    assert [item.type for item in result.items] == ["function_call", "function_call_output"]
    assert model.requests[1].messages[-2].to_input() == {
        "type": "function_call",
        "tool_call_id": "call-1",
        "name": "lookup",
        "arguments": "{}",
    }
    assert model.requests[1].messages[-1].to_input() == {
        "type": "function_call_output",
        "tool_call_id": "call-1",
        "output": "done",
    }
    assert result.items[0].to_input() == {
        "type": "function_call",
        "tool_call_id": "call-1",
        "name": "lookup",
        "arguments": "{}",
    }
    assert result.items[1].to_input() == {
        "type": "function_call_output",
        "tool_call_id": "call-1",
        "output": "done",
    }


def test_agent_tool_decorator_registers_tool():
    agent = Agent(name="agent", instructions="Use tools.")

    @agent.tool
    def hello() -> str:
        return "hello"

    assert len(agent.tools) == 1
    assert agent.tools[0].name == "hello"
    assert hello is agent.tools[0]


def test_agent_tool_decorator_accepts_options():
    agent = Agent(name="agent", instructions="Use tools.")

    @agent.tool(name="say_hello", description="Say hello")
    def hello() -> str:
        return "hello"

    assert len(agent.tools) == 1
    assert agent.tools[0].name == "say_hello"
    assert agent.tools[0].description == "Say hello"
    assert hello is agent.tools[0]


def test_agent_add_tool_registers_callable_and_tool():
    agent = Agent(name="agent", instructions="Use tools.")

    def hello() -> str:
        return "hello"

    @tool
    def goodbye() -> str:
        return "goodbye"

    first = agent.add_tool(hello)
    second = agent.add_tool(goodbye)

    assert [item.name for item in agent.tools] == ["hello", "goodbye"]
    assert first is agent.tools[0]
    assert second is goodbye


def test_agent_as_tool_runs_specialist_and_returns_output():
    specialist = Agent(
        name="summarizer",
        instructions="Summarize text.",
        output_type=SummaryOutput,
        model=FakeModel(['{"summary":"short version"}']),
    )
    manager = Agent(
        name="manager",
        instructions="Use specialist tools.",
        tools=[
            specialist.as_tool(
                tool_name="summarize_text",
                tool_description="Summarize supplied text.",
            )
        ],
    )
    manager_model = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call-1",
                    name="summarize_text",
                    arguments={"input": "Long text"},
                )
            ]
        ),
        "Final answer based on short version",
    ])

    result = manager.run("Please summarize this.", model=manager_model)

    assert result.output == "Final answer based on short version"
    assert manager_model.requests[0].tools[0]["name"] == "summarize_text"
    assert manager_model.requests[1].messages[-1].type == "function_call_output"
    assert manager_model.requests[1].messages[-1].output == SummaryOutput(summary="short version")


def test_agent_as_tool_usage_is_counted_by_manager():
    specialist = Agent(
        name="summarizer",
        instructions="Summarize text.",
        output_type=SummaryOutput,
        model=FakeModel([
            ModelResponse(
                output_text='{"summary":"short version"}',
                usage=Usage(input_tokens=5, output_tokens=2, total_tokens=7),
            )
        ]),
    )
    manager = Agent(
        name="manager",
        instructions="Use specialist tools.",
        tools=[specialist.as_tool(tool_name="summarize_text")],
    )
    manager_model = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call-1",
                    name="summarize_text",
                    arguments={"input": "Long text"},
                )
            ],
            usage=Usage(input_tokens=4, output_tokens=1, total_tokens=5),
        ),
        ModelResponse(
            output_text="Final answer based on short version",
            usage=Usage(input_tokens=6, output_tokens=3, total_tokens=9),
        ),
    ])

    result = manager.run("Please summarize this.", model=manager_model)

    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 6
    assert result.usage.total_tokens == 21


def test_agent_as_tool_passes_runtime_context_to_specialist_tools():
    specialist = Agent(
        name="user_lookup",
        instructions="Lookup user.",
        output_type=SummaryOutput,
        model=FakeModel([
            ModelResponse(tool_calls=[ToolCall(tool_call_id="call-2", name="read_user", arguments={})]),
            '{"summary":"User John is 47"}',
        ]),
    )

    @specialist.tool
    def read_user(ctx=None) -> str:
        return f"{ctx.context['name']} is 47"

    manager = Agent(
        name="manager",
        instructions="Use specialist tools.",
        tools=[specialist.as_tool(tool_name="fetch_user_age")],
    )
    manager_model = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call-1",
                    name="fetch_user_age",
                    arguments={"input": "What is the age?"},
                )
            ]
        ),
        "The user John is 47 years old.",
    ])

    result = manager.run("What is the age?", model=manager_model, context={"name": "John"})

    assert result.output == "The user John is 47 years old."


def test_agent_enforces_max_tool_calls():
    @tool
    def lookup() -> str:
        return "ok"

    agent = Agent(
        name="agent",
        instructions="Use tools.",
        tools=[lookup],
        output_type=RoutedOutput,
        max_tool_calls=0,
    )
    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(tool_call_id="call-1", name="lookup", arguments={})])
    ])

    with pytest.raises(MaxToolCallsExceeded):
        agent.run("hello", model=model)


def test_agent_returns_modmex_output_model():
    agent = Agent(
        name="triage",
        instructions="Route the request.",
        output_type=RoutedOutput,
    )
    result = agent.run("hello", model=FakeModel(['{"intent":"support"}']))

    assert isinstance(result.output, RoutedOutput)
    assert result.output.model_dump() == {"intent": "support"}


def test_agent_does_not_duplicate_schema_in_prompt_for_native_structured_output():
    agent = Agent(
        name="triage",
        instructions="Route the request.",
        output_type=RoutedOutput,
    )
    model = FakeModel(['{"intent":"support"}'])

    agent.run("hello", model=model)

    assert "Return only JSON that matches this schema" not in model.requests[0].messages[0].content


def test_agent_requires_model():
    agent = Agent(name="agent", instructions="Say hi.")

    with pytest.raises(ValueError):
        agent.run("hello")


def test_agent_without_output_type_returns_text():
    agent = Agent(name="agent", instructions="Say hi.")

    result = agent.run("hello", model=FakeModel(["plain text"]))

    assert result.output == "plain text"


def test_agent_without_output_type_returns_empty_string_for_empty_output():
    agent = Agent(name="agent", instructions="Say hi.")

    result = agent.run("hello", model=FakeModel([ModelResponse(output_text=None)]))

    assert result.output == ""


def test_agent_with_output_type_rejects_missing_text():
    agent = Agent(name="agent", instructions="Return JSON.", output_type=RoutedOutput)

    with pytest.raises(OutputValidationError):
        agent.run("hello", model=FakeModel([ModelResponse(output_text=None)]))
