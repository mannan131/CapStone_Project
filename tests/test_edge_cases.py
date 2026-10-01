import copy

import pytest

from src.common.config import load_config


def test_config_returns_independent_copies():
    a = load_config()
    a["paths"]["raw_images_dir"] = "/tmp/polluted"
    b = load_config()
    assert b["paths"]["raw_images_dir"] != "/tmp/polluted"


def test_rules_ignore_bad_threshold():
    from src.recommendation_engine.rules import CauseActionRuleEngine

    cfg = copy.deepcopy(load_config())
    engine = CauseActionRuleEngine(cfg)
    engine.rules = [
        {
            "cause": "broken",
            "condition": {"feature": "x", "operator": "lt", "threshold": None},
            "actions": [],
        },
        {
            "cause": "also_broken",
            "condition": {"feature": "x", "operator": "bogus", "threshold": 1},
            "actions": [],
        },
    ]
    assert engine.evaluate({"x": 0.5}) == []


def test_analyzer_zero_weights_raise(tmp_path):
    import copy

    from PIL import Image

    from src.common.config import load_config
    from src.image_quality.analyzer import ImageQualityAnalyzer

    cfg = copy.deepcopy(load_config())
    cfg["image_quality"]["weights"] = {"blur": 0, "resolution": 0, "contrast": 0, "sharpness": 0}
    p = tmp_path / "img.jpg"
    Image.new("RGB", (128, 128), (100, 100, 100)).save(p, "JPEG")
    with pytest.raises(ValueError, match="weights"):
        ImageQualityAnalyzer(cfg).analyze(p)


def test_db_roundtrip(tmp_path):
    import copy
    import sqlite3

    from src.api import db
    from src.common.config import load_config

    cfg = copy.deepcopy(load_config())
    cfg["paths"]["predictions_db_path"] = str(tmp_path / "test_preds.db")
    db.log_prediction("C-EDGE", 42.5, cfg)
    with sqlite3.connect(tmp_path / "test_preds.db") as con:
        rows = con.execute("SELECT customer_id, churn_probability FROM predictions").fetchall()
    assert rows == [("C-EDGE", 42.5)]
