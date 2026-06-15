import pytest

from modmex_ai import FakeModel, ModelResponse, RECOMMENDED_PROMPT_PREFIX, ToolCall
from modmex_ai.errors import OutputGuardrailTriggered

from tests.examples.utils import load_openai_flow_example


def test_openai_flow_example_runs_with_mocked_handoff(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    example = load_openai_flow_example()
    flow = example.build_flow()
    mocked = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call_order_1",
                    name="transfer_to_order",
                    arguments={
                        "intent": "order",
                        "confidence": 0.95,
                        "reason": "The customer wants to place a pizza order.",
                    },
                )
            ]
        ),
        (
            '{"event":{"type":"pizza.order.clarification.requested",'
            '"payload":{"size":"large","quantity":2}},'
            '"missing_fields":["crust","toppings","delivery_method"],'
            '"reply":"What toppings, crust, and delivery method would you like?",'
            '"confidence":0.95}'
        ),
    ])

    for agent in flow.agents.values():
        agent.model = mocked
    flow.model = mocked

    result = flow.run("Hi, I want to order two large pizzas.")

    assert [item.agent for item in result.agent_results] == ["triage", "order"]
    assert result.output.reply == "What toppings, crust, and delivery method would you like?"
    assert result.events == [{
        "type": "pizza.order.clarification.requested",
        "payload": {
            "size": "large",
            "crust": None,
            "toppings": [],
            "quantity": 2,
            "customer_name": None,
            "delivery_method": None,
            "notes": None,
        },
    }]


def test_openai_flow_example_blocks_inconsistent_order_output(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    example = load_openai_flow_example()
    flow = example.build_flow()
    mocked = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call_order_1",
                    name="transfer_to_order",
                    arguments={
                        "intent": "order",
                        "confidence": 0.95,
                        "reason": "The customer wants to place a pizza order.",
                    },
                )
            ]
        ),
        (
            '{"event":{"type":"pizza.order.clarification.requested",'
            '"payload":{"size":"large","crust":"regular",'
            '"toppings":[],"quantity":2,"delivery_method":"pickup"}},'
            '"missing_fields":["crust","toppings","delivery_method"],'
            '"reply":"What toppings would you like?","confidence":0.95}'
        ),
    ])

    for agent in flow.agents.values():
        agent.model = mocked
    flow.model = mocked
    flow.agents["order"].max_output_guardrail_retries = 0

    with pytest.raises(OutputGuardrailTriggered, match="missing_fields"):
        flow.run("Hi, I want to order two large pizzas.")


def test_openai_flow_example_retries_an_inconsistent_order_output_once(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    example = load_openai_flow_example()
    flow = example.build_flow()
    mocked = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call_order_1",
                    name="transfer_to_order",
                    arguments={
                        "intent": "order",
                        "confidence": 0.95,
                        "reason": "The customer wants to place a pizza order.",
                    },
                )
            ]
        ),
        (
            '{"event":{"type":"pizza.order.clarification.requested",'
            '"payload":{"size":"large","crust":"regular",'
            '"toppings":[],"quantity":2,"delivery_method":"pickup"}},'
            '"missing_fields":["crust","toppings","delivery_method"],'
            '"reply":"What toppings would you like?","confidence":0.95}'
        ),
        (
            '{"event":{"type":"pizza.order.clarification.requested",'
            '"payload":{"size":"large","quantity":2}},'
            '"missing_fields":["crust","toppings","delivery_method"],'
            '"reply":"What crust, toppings, and delivery method would you like?",'
            '"confidence":0.95}'
        ),
    ])

    for agent in flow.agents.values():
        agent.model = mocked
    flow.model = mocked

    result = flow.run("Hi, I want to order two large pizzas.")

    assert result.output.event.payload.crust is None
    assert result.output.event.payload.delivery_method is None
    assert len(mocked.requests) == 3
    assert mocked.requests[-1].messages[-1].role == "developer"


def test_openai_flow_example_runs_menu_tool_loop(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    example = load_openai_flow_example()
    flow = example.build_flow()
    mocked = FakeModel([
        ModelResponse(
            tool_calls=[
                ToolCall(
                    tool_call_id="call_menu_1",
                    name="transfer_to_menu",
                    arguments={
                        "intent": "menu",
                        "confidence": 0.95,
                        "reason": "The customer asks about available sizes.",
                    },
                )
            ]
        ),
        ModelResponse(
            tool_calls=[
                ToolCall(tool_call_id="call_get_menu_1", name="get_menu", arguments={})
            ]
        ),
        "We have small, medium, and large pizzas.",
    ])

    for agent in flow.agents.values():
        agent.model = mocked
    flow.model = mocked

    result = flow.run("What sizes do you have?")
    menu_request_after_tool = mocked.requests[2]

    assert [item.agent for item in result.agent_results] == ["triage", "menu"]
    assert result.output == "We have small, medium, and large pizzas."
    assert menu_request_after_tool.messages[-2].to_input() == {
        "type": "function_call",
        "tool_call_id": "call_get_menu_1",
        "name": "get_menu",
        "arguments": "{}",
    }
    assert menu_request_after_tool.messages[-1].to_input() == {
        "type": "function_call_output",
        "tool_call_id": "call_get_menu_1",
        "output": (
            '{"sizes":["small","medium","large"],'
            '"crusts":["thin","regular","deep_dish"],'
            '"toppings":["pepperoni","mushrooms","onions","olives"],'
            '"prices":{"small":8.99,"medium":12.99,"large":15.99,"toppings":1.5}}'
        ),
    }


def test_openai_flow_example_applies_handoff_prompt_to_specialists(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    example = load_openai_flow_example()
    flow = example.build_flow()

    assert flow.agents["triage"].instructions.startswith("You are a pizzeria triage agent.")
    assert flow.agents["order"].instructions.startswith(RECOMMENDED_PROMPT_PREFIX)
    assert flow.agents["order"].output_strict is False
    assert flow.agents["menu"].instructions.startswith(RECOMMENDED_PROMPT_PREFIX)
    assert flow.agents["support"].instructions.startswith(RECOMMENDED_PROMPT_PREFIX)


def test_openai_flow_example_order_agent_prohibits_inferred_order_details(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    example = load_openai_flow_example()
    instructions = example.build_flow().agents["order"].instructions

    assert "never infer or default any field" in instructions
    assert "never infer regular crust, pickup, delivery" in instructions
    assert "Output consistency is absolute" in instructions
    assert "payload.crust and payload.delivery_method are null" in instructions
    assert "does not mean the order was placed or confirmed" in instructions
