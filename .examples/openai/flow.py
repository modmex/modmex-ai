"""Run a complete modmex-ai Flow against the real OpenAI Responses API.

This example is intentionally outside `tests/` because it calls a real provider
and can incur cost.

Usage:

    MODMEX_AI_RUN_LIVE=1 OPENAI_API_KEY=... poetry run python .examples/openai/flow.py
"""

from __future__ import annotations

import json
import os
from dataclasses import field
from typing import Literal

from modmex import BaseModel

from modmex_ai import (
    Agent,
    Flow,
    GuardrailResult,
    Handoff,
    InMemorySession,
    prompt_with_handoff_instructions,
)
from modmex_ai.errors import ProviderError
from modmex_ai.providers.openai import OpenAIResponsesModel
from modmex_ai.schemas import serialize


class TriageHandoffInput(BaseModel):
    intent: Literal["order", "menu", "support", "out_of_scope"]
    reason: str
    confidence: float


def _record_triage_handoff(context, value: TriageHandoffInput) -> None:
    context.state["triage_handoff"] = serialize(value)


class PizzaOrderPayload(BaseModel):
    size: Literal["small", "medium", "large"] | None = None
    crust: Literal["thin", "regular", "deep_dish"] | None = None
    toppings: list[str] = field(default_factory=list)
    quantity: int | None = None
    customer_name: str | None = None
    delivery_method: Literal["pickup", "delivery"] | None = None
    notes: str | None = None


class PizzaOrderEvent(BaseModel):
    type: Literal[
        "pizza.order.requested",
        "pizza.order.clarification.requested",
        "pizza.order.updated",
    ]
    payload: PizzaOrderPayload


class OrderOutput(BaseModel):
    event: PizzaOrderEvent
    missing_fields: list[str] = field(default_factory=list)
    reply: str
    confidence: float


class PizzaOrderOutputGuardrail:
    name = "pizza_order_output_consistency"

    def check(self, output: OrderOutput, context=None) -> GuardrailResult:
        payload = output.event.payload
        required_fields = ("size", "quantity", "crust", "toppings", "delivery_method")
        missing_fields = set(output.missing_fields)
        unknown_fields = {
            field
            for field in required_fields
            if getattr(payload, field) is None
            or field == "toppings" and getattr(payload, field) == []
        }
        if missing_fields != unknown_fields:
            return GuardrailResult(
                passed=False,
                reason=(
                    "missing_fields must exactly match required payload fields "
                    "whose values are unknown."
                ),
            )
        if output.event.type == "pizza.order.clarification.requested" and not missing_fields:
            return GuardrailResult(
                passed=False,
                reason="A clarification event requires at least one missing field.",
            )
        if output.event.type in {"pizza.order.requested", "pizza.order.updated"} and missing_fields:
            return GuardrailResult(
                passed=False,
                reason="A requested or updated order cannot contain missing fields.",
            )
        return GuardrailResult(passed=True)


def emit_business_events(output, _results) -> list[dict]:
    event = getattr(output, "event", None)
    return [event.model_dump()] if event is not None else []


def require_live_flag() -> None:
    if os.getenv("MODMEX_AI_RUN_LIVE") != "1":
        raise SystemExit(
            "Set MODMEX_AI_RUN_LIVE=1 to run this live API example."
        )


def build_flow() -> Flow:
    model = OpenAIResponsesModel(
        os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    triage = Agent(
        name="triage",
        model=model,
        instructions=(
            "You are a pizzeria triage agent. Your job is to understand what the "
            "customer needs and route the conversation to the best specialist. "
            "Use human_review only when the request is ambiguous, unsafe, or needs "
            "a person. Treat unrelated questions such as math, programming, politics, "
            "medical advice, or general knowledge as out_of_scope."
        ),
        handoffs=[
            Handoff(
                "order",
                description=(
                    "Use this handoff when the customer wants to place, continue, "
                    "or modify a pizza order."
                ),
                input_type=TriageHandoffInput,
                on_handoff=_record_triage_handoff,
            ),
            Handoff(
                "menu",
                description=(
                    "Use this handoff when the customer asks about menu items, "
                    "available toppings, sizes, prices, or recommendations."
                ),
                input_type=TriageHandoffInput,
                on_handoff=_record_triage_handoff,
            ),
            Handoff(
                "support",
                description=(
                    "Use this handoff for pizzeria support requests and for "
                    "out-of-scope questions that should be politely redirected."
                ),
                input_type=TriageHandoffInput,
                on_handoff=_record_triage_handoff,
            ),
        ],
    )

    order = Agent(
        name="order",
        model=model,
        instructions=prompt_with_handoff_instructions(
            "You are a specialized pizza order agent. Help the customer complete "
            "a pizza order while preserving the order state across the conversation. "
            "Extract only details explicitly stated by the customer or present in a "
            "previous structured pizza-order event. Data integrity rules: never infer "
            "or default any field. In particular, never infer regular crust, pickup, "
            "delivery, a customer name, toppings, or quantity. Use null for an "
            "unknown scalar and [] for unknown toppings. Treat size, quantity, crust, "
            "toppings, and delivery_method as required to request an order; list every "
            "unknown required field in missing_fields. Output consistency is absolute: "
            "a field listed in missing_fields must be null in payload, except toppings "
            "which must be []; a non-null payload value means that field is known and "
            "must not appear in missing_fields. Before responding, verify this rule. "
            "For example, for 'I want two large pizzas', payload.size is 'large' and "
            "payload.quantity is 2, but payload.crust and payload.delivery_method are "
            "null, payload.toppings is [], and missing_fields is exactly ['crust', "
            "'toppings', 'delivery_method']. Return "
            "pizza.order.clarification.requested until those fields are known. Return "
            "pizza.order.requested only when the customer has provided all required "
            "details for the first time. It means the order details are ready for a "
            "downstream service; it does not mean the order was placed or confirmed. "
            "Return pizza.order.updated for a later material change to an existing "
            "order. Always provide a concise customer-facing reply that matches the "
            "event semantics."
        ),
        output_type=OrderOutput,
        output_strict=False,
        output_guardrails=[PizzaOrderOutputGuardrail()],
        max_output_guardrail_retries=1,
    )

    menu = Agent(
        name="menu",
        model=model,
        instructions=prompt_with_handoff_instructions(
            "You are a specialized pizzeria menu agent. Answer questions about "
            "menu items, toppings, sizes, prices, and simple recommendations."
        ),
    )

    @menu.tool
    def get_menu() -> dict:
        """Get the current menu."""
        return {
            "sizes": ["small", "medium", "large"],
            "crusts": ["thin", "regular", "deep_dish"],
            "toppings": ["pepperoni", "mushrooms", "onions", "olives"],
            "prices": {
                "small": 8.99,
                "medium": 12.99,
                "large": 15.99,
                "toppings": 1.5,
            },
        }

    support = Agent(
        name="support",
        model=model,
        instructions=prompt_with_handoff_instructions(
            "You are a specialized pizzeria support agent. Handle general support "
            "and boundary messages for the pizzeria."
        ),
    )

    return Flow(
        name="analyze-message",
        entrypoint=triage,
        agents=[order, menu, support],
        emit=emit_business_events,
        max_handoffs=2,
    )


def run_demo_conversation(flow: Flow) -> list[dict]:
    session = InMemorySession(session_id="conversation-001")
    turns = [
        "Hi, I want to order two large pizzas.",
        (
            "Make them thin crust with pepperoni and mushrooms, "
            "for pickup under Alex."
        ),
        "Please add extra olives to both pizzas.",
    ]
    results = []
    continuation = None
    for turn_index, user_message in enumerate(turns, start=1):
        result = flow.run(
            user_message,
            session=session,
            starting_agent=continuation.agent_name if continuation else None,
            provider_state=continuation.provider_state if continuation else None,
        )
        continuation = result.continuation
        results.append({
            "turn": turn_index,
            "input": user_message,
            "starting_agent": result.agent_results[0].agent,
            "last_agent": result.last_agent_name,
            "reply": result.output.reply
            if isinstance(result.output, OrderOutput)
            else result.output,
            "events": result.events,
        })
    return results


def main() -> None:
    require_live_flag()
    flow = build_flow()

    try:
        results = run_demo_conversation(flow)
    except ProviderError as exc:
        payload = {
            "error": str(exc),
            "status_code": exc.status_code,
            "request_id": exc.request_id,
            "response_body": exc.response_body,
        }
        print(json.dumps(payload, indent=2, default=str), flush=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        payload = {
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
        print(json.dumps(payload, indent=2, default=str), flush=True)
        raise SystemExit(1) from exc

    payload = {"turns": results}
    print(json.dumps(payload, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
