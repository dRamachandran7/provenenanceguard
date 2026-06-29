"""Tests for the Flask app skeleton.

Signals are monkeypatched to deterministic values so the endpoints are
exercised without the network, and the audit log is redirected to a temp file.
"""

import pytest

import app as app_module
from provenanceguard import audit
from provenanceguard.signals import SignalResult


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Force deterministic, offline signal results so the test isolates wiring.
    monkeypatch.setattr(
        app_module,
        "classify_with_llm",
        lambda text, **kw: SignalResult(name="llm_classifier", score=0.85, reasoning="stub"),
    )
    monkeypatch.setattr(
        app_module,
        "analyze_stylometry",
        lambda text, **kw: SignalResult(name="stylometric", score=0.85, reasoning="stub"),
    )
    # Redirect the audit log to an isolated temp file.
    monkeypatch.setenv("PROVENANCEGUARD_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def _submit(client, text="a poem", creator_id="test-user-1"):
    return client.post("/submit", json={"text": text, "creator_id": creator_id})


def test_submit_returns_decision_with_required_fields(client):
    resp = _submit(client)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "classified"
    assert body["content_id"]  # present and non-empty
    assert body["creator_id"] == "test-user-1"
    assert body["attribution"] == "likely_ai"  # llm score 0.85 -> likely_ai
    assert body["confidence"] == pytest.approx(0.85)
    assert body["label"] == "ai-generated"
    assert body["signals"][0]["name"] == "llm_classifier"


def test_submit_text_alias_works(client):
    resp = client.post("/submit_text", json={"text": "x", "creator_id": "u"})
    assert resp.status_code == 200
    assert resp.get_json()["content_id"]


def test_submit_requires_text(client):
    resp = client.post("/submit", json={"creator_id": "u"})
    assert resp.status_code == 400


def test_submit_requires_creator_id(client):
    resp = client.post("/submit", json={"text": "a poem"})
    assert resp.status_code == 400


def test_submission_is_written_to_structured_log(client):
    _submit(client)
    entries = client.get("/log").get_json()["entries"]
    assert len(entries) == 1
    e = entries[0]
    # All required structured fields are present.
    for field in (
        "content_id",
        "creator_id",
        "timestamp",
        "attribution",
        "confidence",
        "llm_score",
        "status",
    ):
        assert field in e, f"missing {field}"
    assert e["status"] == "classified"
    assert e["timestamp"].endswith("Z")


def test_content_id_links_submission_to_log(client):
    content_id = _submit(client).get_json()["content_id"]
    entries = client.get("/log").get_json()["entries"]
    assert entries[0]["content_id"] == content_id


def test_appeal_logs_event_and_updates_status(client):
    content_id = _submit(client).get_json()["content_id"]
    resp = client.post("/appeal", json={"content_id": content_id, "reason": "I wrote this!"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "under_review"

    entries = client.get("/log").get_json()["entries"]
    appeal_entry = entries[-1]
    assert appeal_entry["event"] == "appeal"
    assert appeal_entry["content_id"] == content_id
    assert appeal_entry["reason"] == "I wrote this!"
    assert appeal_entry["previous_label"] == "ai-generated"


def test_appeal_unknown_content_id_returns_404(client):
    resp = client.post("/appeal", json={"content_id": "nope", "reason": "x"})
    assert resp.status_code == 404


def test_log_limit_returns_most_recent(client):
    _submit(client, text="first")
    _submit(client, text="second")
    _submit(client, text="third")
    entries = client.get("/log?limit=2").get_json()["entries"]
    assert len(entries) == 2
    assert entries[-1]["text"] == "third"
