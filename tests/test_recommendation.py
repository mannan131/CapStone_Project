from src.recommendation_engine.recommender import (
    RetentionRecommender,
    eval_priority_formula,
)
from src.recommendation_engine.rules import CauseActionRuleEngine


def test_rules_fire():
    engine = CauseActionRuleEngine()
    matched = engine.evaluate(
        {
            "image_quality_score": 0.2,
            "login_frequency_last_30d": 1.0,
            "support_tickets_last_90d": 5,
            "late_payments_count": 2,
            "avg_days_since_last_login": 30,
            "total_purchases": 0,
            "avg_session_duration_min": 2,
            "feature_usage_score": 10,
        }
    )
    causes = {m["cause"] for m in matched}
    assert "low_image_quality" in causes
    assert "high_support_tickets" in causes
    assert "late_payments" in causes


def test_recommender_priority_and_fallback():
    rec = RetentionRecommender()
    r = rec.recommend(
        "C1",
        80.0,
        {
            "customer_lifetime_value": 1000.0,
            "login_frequency_last_30d": 50,
            "image_quality_score": 0.9,
            "support_tickets_last_90d": 0,
            "late_payments_count": 0,
            "avg_days_since_last_login": 1,
            "total_purchases": 10,
            "avg_session_duration_min": 30,
            "feature_usage_score": 90,
        },
        None,
    )
    # spec formula: churn_probability * customer_lifetime_value
    assert r.priority_score == 80.0 * 1000.0
    assert len(r.recommended_actions) > 0

    r2 = rec.recommend(
        "C2",
        90.0,
        {
            "customer_lifetime_value": 500,
            "image_quality_score": 0.1,
            "login_frequency_last_30d": 0.5,
        },
        [{"feature": "image_quality_score", "direction": "increases_churn", "shap_value": 0.5}],
    )
    assert "low_image_quality" in r2.top_causes


def test_shap_only_cause_gets_playbook_actions():
    # No rule threshold fires, but SHAP flags late payments -> playbook actions, not fallback.
    rec = RetentionRecommender()
    r = rec.recommend(
        "C9",
        70.0,
        {"customer_lifetime_value": 300, "late_payments_count": 1, "total_purchases": 5},
        [{"feature": "late_payments_count", "direction": "increases_churn", "shap_value": 0.4}],
    )
    assert "late_payments" in r.top_causes
    assert any(
        a["cause"] == "late_payments" and a["shap_corroborated"] for a in r.recommended_actions
    )


def test_max_actions_truncation():
    import copy

    from src.common.config import load_config

    cfg = copy.deepcopy(load_config())
    cfg["recommendation_engine"]["max_actions_per_customer"] = 1
    rec = RetentionRecommender(cfg)
    r = rec.recommend(
        "C3",
        95.0,
        {
            "customer_lifetime_value": 100,
            "image_quality_score": 0.1,
            "login_frequency_last_30d": 0.5,
            "support_tickets_last_90d": 9,
            "late_payments_count": 5,
        },
        None,
    )
    assert len(r.recommended_actions) == 1


def test_priority_formula_eval():
    assert (
        eval_priority_formula(
            "churn_probability * customer_lifetime_value",
            {"churn_probability": 50, "customer_lifetime_value": 200},
        )
        == 10000
    )
    import pytest

    with pytest.raises(ValueError):
        eval_priority_formula(
            "__import__('os').system('x')", {"churn_probability": 1, "customer_lifetime_value": 1}
        )
