import json
import time
import uuid

from app.webhooks import sign


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["knowledge_chunks"] > 0


def test_chat_requires_api_key(client):
    response = client.post("/v1/chat", json={"customer_ref": "c1", "message": "hi"})
    assert response.status_code == 401


def test_chat_answers_from_knowledge_base(client, auth):
    response = client.post("/v1/chat", headers=auth, json={"customer_ref": "c-kb", "message": "What time is check-in?"})
    assert response.status_code == 200
    body = response.json()
    assert "2:00 PM" in body["reply"]
    assert "hotel-policies" in body["sources"]
    assert body["tool_calls"] == []


def test_chat_calls_booking_tool(client, auth):
    body = client.post(
        "/v1/chat", headers=auth, json={"customer_ref": "c-bk", "message": "Check booking SG1234"}
    ).json()
    assert body["tool_calls"][0]["name"] == "lookup_booking"
    assert body["tool_calls"][0]["result"]["found"] is True
    assert "Deluxe Sea View" in body["reply"]


def test_complaint_creates_ticket_and_conversation_is_stored(client, auth):
    body = client.post(
        "/v1/chat", headers=auth, json={"customer_ref": "c-tk", "message": "The shower is broken in room 204"}
    ).json()
    assert body["tool_calls"][0]["name"] == "create_ticket"
    transcript = client.get(f"/v1/conversations/{body['conversation_id']}", headers=auth).json()
    roles = [m["role"] for m in transcript["messages"]]
    assert roles == ["user", "tool", "assistant"]
    assert transcript["messages"][-1]["meta"]["prompt_version"].startswith("support-agent/")


def test_escalation_marks_conversation(client, auth):
    body = client.post(
        "/v1/chat", headers=auth, json={"customer_ref": "c-hu", "message": "Let me talk to a manager"}
    ).json()
    assert body["escalated"] is True


def test_conversation_continues_with_same_id(client, auth):
    first = client.post(
        "/v1/chat", headers=auth, json={"customer_ref": "c-multi", "message": "When is breakfast?"}
    ).json()
    second = client.post(
        "/v1/chat",
        headers=auth,
        json={"customer_ref": "c-multi", "message": "Is there a pool?", "conversation_id": first["conversation_id"]},
    ).json()
    assert second["conversation_id"] == first["conversation_id"]


def test_ingest_new_document_is_searchable(client, auth):
    doc = {"doc_id": "spa", "content": "# Spa\n\nThe Lotus Spa offers ayurvedic massages from 10 AM to 8 PM daily."}
    assert client.post("/v1/knowledge", headers=auth, json=doc).status_code == 201
    body = client.post(
        "/v1/chat", headers=auth, json={"customer_ref": "c-spa", "message": "When is the spa open for massages?"}
    ).json()
    assert "spa" in body["sources"]
    assert "10 AM" in body["reply"]


def _signed(payload: dict, secret: str = "test-secret"):
    raw = json.dumps(payload).encode()
    ts = str(int(time.time()))
    return raw, {"X-Timestamp": ts, "X-Signature": sign(secret, ts, raw), "Content-Type": "application/json"}


def test_webhook_rejects_bad_signature(client):
    raw, headers = _signed({"event_id": "e1", "sender": "+911234", "text": "hi"}, secret="wrong")
    assert client.post("/v1/webhooks/messages", content=raw, headers=headers).status_code == 401


def test_webhook_is_idempotent(client):
    event = {"event_id": f"evt-{uuid.uuid4()}", "sender": "+919999", "text": "What time is check-out?"}
    raw, headers = _signed(event)
    first = client.post("/v1/webhooks/messages", content=raw, headers=headers).json()
    second = client.post("/v1/webhooks/messages", content=raw, headers=headers).json()
    assert first["duplicate"] is False and second["duplicate"] is True
    assert first["reply"] == second["reply"]
    assert "11:00 AM" in first["reply"]


def test_metrics_exposed(client):
    text = client.get("/metrics").text
    assert "agent_replies_total" in text
