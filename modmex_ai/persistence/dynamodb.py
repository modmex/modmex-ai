"""DynamoDB persistence backed by the optional modmex-lambda connector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modmex_ai.flows import FlowStateConflictError, PersistedFlowState
from modmex_ai.sessions import SessionConflictError, SessionSnapshot

if TYPE_CHECKING:
    from modmex_lambda.connectors.idynamodb import IDynamodbConnector


class DynamoDbDurableSessionStore:
    """DynamoDB implementation of :class:`DurableSessionStore`."""

    def __init__(self, connector: IDynamodbConnector) -> None:
        self.connector = connector

    def load(self, session_id: str) -> SessionSnapshot | None:
        response = self.connector.get({"Key": _key("session", session_id)})
        item = response.get("Item")
        return SessionSnapshot(**item["snapshot"]) if item else None

    def save(self, snapshot: SessionSnapshot, *, expected_revision: int) -> SessionSnapshot:
        current = self.load(snapshot.session_id)
        if current is not None and current.revision != expected_revision:
            raise SessionConflictError(f"Session {snapshot.session_id!r} revision conflict")
        if current is None and expected_revision != 0:
            raise SessionConflictError(f"Session {snapshot.session_id!r} revision conflict")
        stored = SessionSnapshot(**{**snapshot.model_dump(), "revision": expected_revision + 1})
        try:
            self.connector.put({
                "Item": {**_key("session", snapshot.session_id), "snapshot": stored.model_dump(), "revision": stored.revision},
                **_revision_condition(expected_revision),
            })
        except Exception as error:
            if _is_conditional_failure(error):
                raise SessionConflictError(f"Session {snapshot.session_id!r} revision conflict") from error
            raise
        return stored


class DynamoDbFlowStateStore:
    """DynamoDB implementation of :class:`FlowStateStore` with CAS writes."""

    def __init__(self, connector: IDynamodbConnector) -> None:
        self.connector = connector

    def load(self, flow_instance_id: str) -> PersistedFlowState | None:
        response = self.connector.get({"Key": _key("flow_state", flow_instance_id)})
        item = response.get("Item")
        return PersistedFlowState(**item["state"]) if item else None

    def save(self, state: PersistedFlowState, *, expected_revision: int) -> PersistedFlowState:
        current = self.load(state.flow_instance_id)
        if current is not None and current.revision != expected_revision:
            raise FlowStateConflictError(f"Flow {state.flow_instance_id!r} revision conflict")
        if current is None and expected_revision != 0:
            raise FlowStateConflictError(f"Flow {state.flow_instance_id!r} revision conflict")
        stored = PersistedFlowState(**{**state.model_dump(), "revision": expected_revision + 1})
        item = {**_key("flow_state", state.flow_instance_id), "state": stored.model_dump(), "revision": stored.revision, "status": stored.status}
        if stored.ttl is not None:
            item["ttl"] = stored.ttl
        try:
            self.connector.put({"Item": item, **_revision_condition(expected_revision)})
        except Exception as error:
            if _is_conditional_failure(error):
                raise FlowStateConflictError(f"Flow {state.flow_instance_id!r} revision conflict") from error
            raise
        return stored


def _key(kind: str, identifier: str) -> dict[str, str]:
    return {"pk": f"{kind}#{identifier}", "sk": kind}


def _revision_condition(expected_revision: int) -> dict[str, Any]:
    return {
        "ConditionExpression": "attribute_not_exists(#revision) OR #revision = :expected_revision",
        "ExpressionAttributeNames": {"#revision": "revision"},
        "ExpressionAttributeValues": {":expected_revision": expected_revision},
    }


def _is_conditional_failure(error: Exception) -> bool:
    response = getattr(error, "response", {})
    return response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"
