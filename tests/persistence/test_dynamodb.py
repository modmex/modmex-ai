import pytest

from modmex_ai.flows import FlowStateConflictError, FlowStateStatus, PersistedFlowState
from modmex_ai.persistence import DynamoDbDurableSessionStore, DynamoDbFlowStateStore
from modmex_ai.sessions import SessionConflictError, SessionSnapshot


class ConditionalFailure(Exception):
    response = {"Error": {"Code": "ConditionalCheckFailedException"}}


class MemoryConnector:
    """Evaluates the two condition shapes the store emits, like DynamoDB would."""

    def __init__(self):
        self.items = {}
        self.gets = 0

    def get(self, params):
        self.gets += 1
        key = tuple(params["Key"].values())
        return {"Item": self.items[key]} if key in self.items else {}

    def put(self, params):
        item = params["Item"]
        key = (item["pk"], item["sk"])
        current = self.items.get(key)
        condition = params["ConditionExpression"]
        names = params["ExpressionAttributeNames"]
        values = params.get("ExpressionAttributeValues", {})
        # DynamoDB rejects unused expression names/values.
        assert all(name in condition for name in names) and all(v in condition for v in values)
        if condition == "attribute_not_exists(#pk)":
            ok = current is None
        elif condition == "#revision = :expected_revision":
            ok = current is not None and current["revision"] == values[":expected_revision"]
        else:
            raise AssertionError(condition)
        if not ok:
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


def test_save_does_not_read_before_write_and_rejects_missing_items():
    connector = MemoryConnector()
    store = DynamoDbDurableSessionStore(connector)
    store.save(SessionSnapshot(session_id="s"), expected_revision=0)
    store.save(SessionSnapshot(session_id="s"), expected_revision=1)
    assert connector.gets == 0

    connector.items.clear()  # e.g. removed by TTL between turns
    with pytest.raises(SessionConflictError):
        store.save(SessionSnapshot(session_id="s"), expected_revision=2)
    with pytest.raises(ValueError):
        store.save(SessionSnapshot(session_id="s"), expected_revision=-1)
