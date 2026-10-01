"""Streamlit admin dashboard: single-customer view + bulk at-risk ranking."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config, schema  # noqa: E402
from src.data_ingestion.loader import CustomerDataLoader  # noqa: E402
from src.explainability.shap_explainer import ShapExplainer  # noqa: E402
from src.models.predictor import ChurnPredictor  # noqa: E402
from src.recommendation_engine.recommender import RetentionRecommender  # noqa: E402
from src.recommendation_engine.recommender import (
    eval_priority_formula,
)

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")
config = load_config()
SCHEMA = schema(config)
ID_COL = SCHEMA["id_column"]
TARGET_COL = SCHEMA["target_column"]
CLV_COL = SCHEMA.get("clv_column", "customer_lifetime_value")
IMG_COL = config.get("feature_fusion", {}).get("image_score_column", "image_quality_score")
TOP_N_DEFAULT = int(config.get("dashboard", {}).get("top_n_at_risk_default", 20))
PRIORITY_FORMULA = config.get("recommendation_engine", {}).get(
    "priority_formula", "churn_probability * customer_lifetime_value"
)

st.set_page_config(page_title="Churn Intelligence Dashboard", layout="wide")
config = load_config()

st.title("🔍 Customer Churn Prediction & Retention Intelligence")


@st.cache_resource
def _state():
    return {
        "predictor": ChurnPredictor(config),
        "explainer": ShapExplainer(config),
        "recommender": RetentionRecommender(config),
        "loader": CustomerDataLoader(config),
    }


def _waterfall_fig(top_features: list[dict], customer_id: str):
    """SHAP-style waterfall: cumulative contribution of top features."""
    feats = [f["feature"] for f in top_features]
    vals = [float(f["shap_value"]) for f in top_features]
    order = sorted(range(len(vals)), key=lambda i: abs(vals[i]))
    feats = [feats[i] for i in order]
    vals = [vals[i] for i in order]
    cum = [0.0]
    for v in vals:
        cum.append(cum[-1] + v)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (f, v) in enumerate(zip(feats, vals)):
        color = "#d62728" if v > 0 else "#2ca02c"
        ax.barh(f, v, left=cum[i], color=color)
        ax.plot([cum[i], cum[i + 1]], [i, i], color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"SHAP waterfall — {customer_id}")
    fig.tight_layout()
    return fig


try:
    state = _state()
    df = state["loader"].load_default()
except Exception as e:
    st.error(f"Model artifacts not ready. Run training first: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Single customer", "Bulk at-risk view"])

with tab1:
    cid = st.selectbox("Search customer", sorted(df[ID_COL].unique()))
    row = df[df[ID_COL] == cid].iloc[0].to_dict()
    record = {k: v for k, v in row.items() if k != TARGET_COL}
    proba = state["predictor"].predict_proba(record)
    col1, col2 = st.columns(2)
    with col1:
        color = "red" if proba >= 70 else "orange" if proba >= 40 else "green"
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba,
                title={"text": "Churn probability (%)"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}},
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Image quality score", f"{record.get(IMG_COL, 0):.3f}")
    with col2:
        exp = state["explainer"].explain_customer(record, customer_id=cid)
        st.subheader("Why will this customer churn?")
        st.write(exp.explanation)
        st.pyplot(_waterfall_fig(exp.top_features, cid), use_container_width=True)
        cust_features = {k: v for k, v in record.items() if k != ID_COL}
        rec = state["recommender"].recommend(cid, proba, cust_features, exp.top_features)
        st.subheader("Recommended retention actions")
        for a in rec.recommended_actions:
            flag = " ✅ SHAP-confirmed" if a.get("shap_corroborated") else ""
            st.markdown(f"- **{a['action']}** ({a['cause']}){flag} — {a.get('detail', '')}")
        st.metric("Priority score", f"{rec.priority_score:.2f}")

with tab2:
    top_n = st.slider("Top-N at-risk customers", 5, 50, TOP_N_DEFAULT)
    with st.spinner("Scoring customers..."):
        scored = state["predictor"].batch_predict(df)
        scored["priority_score"] = scored.apply(
            lambda r: eval_priority_formula(
                PRIORITY_FORMULA,
                {
                    "churn_probability": float(r["churn_probability"]),
                    "customer_lifetime_value": float(r.get(CLV_COL, 0) or 0),
                },
            ),
            axis=1,
        )
        ranked = scored.sort_values("priority_score", ascending=False).head(top_n)
    st.dataframe(
        ranked[
            [
                ID_COL,
                "churn_probability",
                "priority_score",
                CLV_COL,
                IMG_COL,
                TARGET_COL,
            ]
        ]
    )
    st.download_button(
        "Export to CSV", ranked.to_csv(index=False), "at_risk_customers.csv", "text/csv"
    )
