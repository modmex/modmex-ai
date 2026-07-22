import pytest

from modmex_ai.flows import FlowStateConflictError, FlowStateStatus, PersistedFlowState
from modmex_ai.persistence import DynamoDbDurableSessionStore, DynamoDbFlowStateStore
from modmex_ai.sessions import SessionConflictError, SessionSnapshot


class ConditionalFailure(Exception):
    response = {"Error": {"Code": "ConditionalCheckFailedException"}}


class MemoryConnector:
    def __init__(self):
        self.items = {}

    def get(self, params):
        key = tuple(params["Key"].values())
        return {"Item": self.items[key]} if key in self.items else {}

    def put(self, params):
        item = params["Item"]
        key = (item["pk"], item["sk"])
        current = self.items.get(key)
        expected = params["ExpressionAttributeValues"][":expected_revision"]
        if current is not None and current["revision"] != expected:
            raise ConditionalFailure()
        if current is None and expected != 0:
            raise ConditionalFailure()
        self.items[key] = item
        return {}


def test_durable_session_store_uses_dynamodb_compare_and_swap():
    store = DynamoDbDurableSessionStore(MemoryConnector())
    snapshot = SessionSnapshot(session_id="session-1")

    saved = store.save(snapshot, expected_revision=0)

    assert saved.revision == 1
    assert store.load("session-1").revision == 1
    with pytest.raises(SessionConflictError):
        store.save(snapshot, expected_revision=0)


def test_flow_state_store_preserves_idempotency_and_compare_and_swap():
    store = DynamoDbFlowStateStore(MemoryConnector())
    state = PersistedFlowState(
        flow_instance_id="flow-1",
        idempotency_key="message-1",
        status=FlowStateStatus.SUSPENDED,
    )

    saved = store.save(state, expected_revision=0)

    assert saved.revision == 1
    assert store.load("flow-1").idempotency_key == "message-1"
    with pytest.raises(FlowStateConflictError):
        store.save(PersistedFlowState(flow_instance_id="flow-1"), expected_revision=0)
