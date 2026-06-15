import pytest

from modmex_ai import InMemorySession, SessionItem, SessionSnapshot
from modmex_ai.messages import Message


def test_in_memory_session_manages_items():
    session = InMemorySession(session_id="conversation-1")
    first = SessionItem(role="user", content="hello")
    second = SessionItem(role="assistant", content="hi")

    session.add_items([first, second])

    assert session.id == "conversation-1"
    assert session.get_items() == [first, second]
    assert session.get_items(limit=1) == [second]
    assert session.get_items(limit=0) == []
    assert session.pop_item() == second
    assert session.get_items() == [first]

    session.clear_session()

    assert session.get_items() == []


def test_session_item_converts_to_message_and_input():
    item = SessionItem(
        role="user",
        content="hello",
    )

    message = item.to_message()

    assert message == Message(
        role="user",
        content="hello",
    )
    assert SessionItem.from_message(message).to_input() == {
        "role": "user",
        "content": "hello",
    }
    assert item.to_input() == {
        "role": "user",
        "content": "hello",
    }


def test_function_call_session_item_is_not_a_role_message():
    item = SessionItem(
        type="function_call",
        tool_call_id="call-1",
        name="lookup",
        arguments={"id": "1"},
    )

    with pytest.raises(ValueError, match="Only message session items"):
        item.to_message()
    assert item.to_input() == {
        "type": "function_call",
        "tool_call_id": "call-1",
        "name": "lookup",
        "arguments": '{"id":"1"}',
    }


def test_message_session_item_requires_role_and_content():
    with pytest.raises(ValueError, match="Message session items require role and content"):
        SessionItem().to_message()


def test_handoff_session_item_includes_optional_output():
    item = SessionItem(
        type="handoff_call_output",
        tool_call_id="call-1",
        output={"transferred": True},
    )

    assert item.to_input() == {
        "type": "handoff_call_output",
        "tool_call_id": "call-1",
        "output": '{"transferred":true}',
    }


def test_session_snapshot_restores_memory_history():
    original = InMemorySession(
        session_id="conversation-1",
        items=[SessionItem(role="user", content="hello")],
    )

    snapshot = SessionSnapshot.from_session(original)
    restored = snapshot.to_memory_session()

    assert snapshot.model_dump() == {
        "session_id": "conversation-1",
        "schema_version": 1,
        "revision": 0,
        "summary": None,
        "continuation": None,
        "provider_state": None,
        "items": [{
            "type": "message",
            "role": "user",
            "content": "hello",
            "tool_call_id": None,
            "name": None,
            "arguments": None,
            "output": None,
        }],
    }
    assert restored.id == original.id
    assert restored.get_items() == original.get_items()
