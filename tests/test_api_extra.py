import base64
import io

from fastapi.testclient import TestClient
from PIL import Image


def _client():
    from src.api.app import app

    return TestClient(app)


def _payload(customer_id="CUST-00001"):
    import pandas as pd

    from src.common.config import artifact_paths

    df = pd.read_csv(artifact_paths()["raw_customers_csv"])
    row = df[df["customer_id"] == customer_id].iloc[0].to_dict()
    row.pop("churned", None)
    return row


def _jpeg_bytes(size=(128, 128)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 180, 60)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def test_predict_churn_ok():
    r = _client().post("/predict-churn", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 100
    assert body["risk_band"] in ("low", "medium", "high")


def test_explain_ok():
    r = _client().get("/explain/CUST-00001")
    assert r.status_code == 200
    body = r.json()
    assert body["top_features"] and body["explanation"]


def test_recommend_ok():
    r = _client().get("/recommend/CUST-00001")
    assert r.status_code == 200
    body = r.json()
    assert body["recommended_actions"] and body["priority_score"] >= 0


def test_upload_image_ok():
    r = _client().post(
        "/upload-image",
        params={"customer_id": "C-TEST", "product_id": "P1"},
        files={"file": ("p.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    assert 0 <= r.json()["quality_score"] <= 1


def test_upload_image_rejects_bad_file():
    r = _client().post(
        "/upload-image",
        params={"customer_id": "C-TEST", "product_id": "bad"},
        files={"file": ("p.jpg", b"not-an-image", "image/jpeg")},
    )
    assert r.status_code == 400


def test_predict_with_image_ok():
    payload = _payload()
    payload["image_b64"] = base64.b64encode(_jpeg_bytes()).decode()
    r = _client().post("/predict-with-image", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 100
    assert body["image_quality_score"] is not None


def test_predict_with_image_no_image_ok():
    r = _client().post("/predict-with-image", json=_payload())
    assert r.status_code == 200
