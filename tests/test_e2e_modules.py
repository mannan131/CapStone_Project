import copy

import pytest
from PIL import Image

from src.common.config import artifact_paths, load_config
from src.data_ingestion.image_upload import ImageUploadHandler
from src.image_quality.analyzer import ImageQualityAnalyzer


def _img(path, size=(128, 128)):
    Image.new("RGB", size, (120, 180, 60)).save(path, "JPEG", quality=90)
    return path


def _fresh_config(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["paths"]["raw_images_dir"] = str(tmp_path / "raw")
    return cfg


def test_upload_and_quality(tmp_path):
    cfg = _fresh_config(tmp_path)
    h = ImageUploadHandler(cfg)
    src = _img(tmp_path / "src.jpg")
    dest = h.save_file(src, "C1", "P1")
    assert dest.exists()
    rep = ImageQualityAnalyzer(cfg).analyze(dest)
    assert 0 <= rep.quality_score <= 1
    # corrupt file rejected
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not-an-image")
    with pytest.raises(ValueError):
        h.save_file(bad, "C1", "P2")
    # oversize rejected
    cfg["image_quality"]["max_file_size_mb"] = 0.0001
    h2 = ImageUploadHandler(cfg)
    with pytest.raises(ValueError):
        h2.save_file(src, "C1", "P3")


def test_save_bytes_cleans_up_on_bad_payload(tmp_path):
    cfg = _fresh_config(tmp_path)
    h = ImageUploadHandler(cfg)
    with pytest.raises(ValueError):
        h.save_bytes(b"not-an-image", "C9", "P9")
    assert not (tmp_path / "raw" / "C9" / "P9.jpg").exists()


def test_predictor_and_shap_recommender():
    import pandas as pd

    from src.explainability.shap_explainer import ShapExplainer
    from src.models.predictor import ChurnPredictor
    from src.recommendation_engine.recommender import RetentionRecommender

    df = pd.read_csv(artifact_paths()["raw_customers_csv"])
    row = df.iloc[0].to_dict()
    cid = row.pop("customer_id")
    row.pop("churned", None)
    p = ChurnPredictor().predict_proba(row)
    assert 0 <= p <= 100
    exp = ShapExplainer().explain_customer(row, customer_id=cid)
    assert len(exp.top_features) > 0 and exp.explanation
    rec = RetentionRecommender().recommend(cid, p, row, exp.top_features)
    assert rec.priority_score >= 0 and len(rec.recommended_actions) > 0
    assert len(rec.recommended_actions) <= 3  # spec: max_actions_per_customer
