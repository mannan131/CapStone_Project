from fastapi.testclient import TestClient


def _client():
    from src.api.app import app

    return TestClient(app)


def test_health():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_validation():
    c = _client()
    # missing customer_id -> 422 (only if model ready; otherwise 503). Accept either.
    r = c.post("/predict-churn", json={})
    assert r.status_code in (422, 503)


def test_explain_not_found_or_unready():
    c = _client()
    r = c.get("/explain/NOPE-123")
    assert r.status_code in (404, 503)
