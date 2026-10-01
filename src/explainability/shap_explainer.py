"""Per-customer SHAP explanations."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import asdict, dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.common.config import artifact_paths, load_config, schema
from src.common.logging import get_logger
from src.feature_fusion.pipeline import FeatureFusionPipeline

log = get_logger("shap")

# Fallback model-class -> family mapping when the config map has no entry.
_FAMILY_BY_CLASS = (
    ("XGB", "xgboost"),
    ("Forest", "random_forest"),
    ("LogisticRegression", "logistic_regression"),
    ("TorchMLP", "neural_network"),
    ("MLP", "neural_network"),
)


@dataclass
class FeatureContribution:
    feature: str
    shap_value: float
    direction: str  # "increases_churn" | "decreases_churn"

    def to_dict(self):
        return asdict(self)


@dataclass
class ShapExplanation:
    customer_id: str
    churn_probability: float
    base_value: float
    top_features: list[dict]
    explanation: str
    force_plot_path: str | None = None

    def to_dict(self):
        return asdict(self)


class ShapExplainer:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        sh = self.config.get("shap", self.config.get("explainability", {}))
        self.top_n = int(sh.get("top_n_features", sh.get("top_n", 5)))
        self.background_samples = int(
            sh.get(
                "background_samples",
                self.config.get("explainability", {}).get("background_samples", 50),
            )
        )
        self.explainer_map: dict = sh.get("explainer_type_map", {})
        paths = artifact_paths(self.config)
        s = schema(self.config)
        self.id_col = s["id_column"]
        self.target_col = s["target_column"]
        self.model = joblib.load(paths["best_model"])
        self.fusion: FeatureFusionPipeline = FeatureFusionPipeline.load(
            str(paths["fusion_pipeline"])
        )
        self.feature_names = self.fusion.get_feature_names()
        self._explainer = None
        self._background: np.ndarray | None = None

    def _model_family(self) -> str:
        name = type(self.model).__name__
        for token, family in _FAMILY_BY_CLASS:
            if token in name:
                return family
        return "logistic_regression"

    def _explainer_type(self) -> str:
        family = self._model_family()
        default = "kernel" if family == "logistic_regression" else "tree"
        return str(self.explainer_map.get(family, default))

    def _build_explainer(self, background: np.ndarray):
        kind = self._explainer_type()
        m = self.model
        if kind == "tree":
            try:
                return shap.TreeExplainer(m)
            except Exception:
                pass
        if kind == "deep":
            try:
                import torch

                return shap.DeepExplainer(m.model_, torch.from_numpy(background.astype("float32")))
            except Exception as e:
                log.info("shap_deep_fallback", error=str(e))
        # kernel (also the fallback for tree/deep failures)

        def f(x):
            return m.predict_proba(x)[:, 1]

        n = min(20, len(background))
        return shap.KernelExplainer(f, shap.sample(background, n))

    def _default_background(self, X: np.ndarray) -> np.ndarray:
        """Real-data background sample when available, else seeded noise around X."""
        if self._background is None:
            try:
                from src.data_ingestion.loader import CustomerDataLoader

                df = CustomerDataLoader(self.config).load_default()
                drop = [c for c in (self.id_col, self.target_col) if c in df.columns]
                sample = df.drop(columns=drop).sample(
                    min(self.background_samples, len(df)), random_state=42
                )
                self._background = self.fusion.transform(sample)
            except Exception as e:
                log.info("shap_background_fallback", error=str(e))
                rng = np.random.default_rng(42)
                self._background = np.repeat(X, 10, axis=0) + rng.normal(
                    0, 0.05, size=(10, X.shape[1])
                )
        return self._background

    def explain_customer(
        self, record: dict, customer_id: str = "unknown", background_df: pd.DataFrame | None = None
    ) -> ShapExplanation:
        df = pd.DataFrame([record])
        for drop in (self.id_col, self.target_col):
            if drop in df.columns:
                df = df.drop(columns=[drop])
        X = self.fusion.transform(df)
        if background_df is not None:
            bg = self.fusion.transform(background_df)
        else:
            bg = self._default_background(X)
        if self._explainer is None:
            self._explainer = self._build_explainer(bg)
        base_value = self._expected_value()
        try:
            sv = self._explainer.shap_values(X)  # type: ignore
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            shap_values = np.asarray(sv).ravel()
        except Exception:
            vals = self._explainer(X)  # type: ignore
            shap_values = (
                np.asarray(vals.values).ravel() if hasattr(vals, "values") else np.zeros(X.shape[1])
            )

        proba = float(self.model.predict_proba(X)[0, 1]) * 100
        contribs = []
        for feat, v in zip(self.feature_names, shap_values):
            contribs.append(
                FeatureContribution(
                    feat, round(float(v), 4), "increases_churn" if v > 0 else "decreases_churn"
                )
            )
        contribs.sort(key=lambda c: abs(c.shap_value), reverse=True)
        top = contribs[: self.top_n]
        inc = [c.feature for c in top if c.direction == "increases_churn"]
        dec = [c.feature for c in top if c.direction == "decreases_churn"]
        text = (
            f"Customer {customer_id} has a {proba:.1f}% churn risk. "
            + (f"Main risk drivers: {', '.join(inc)}. " if inc else "No dominant risk drivers. ")
            + (f"Protective factors: {', '.join(dec)}." if dec else "")
        )
        plot_path = self._save_bar_plot(customer_id, top)
        exp = ShapExplanation(
            customer_id=customer_id,
            churn_probability=round(proba, 2),
            base_value=round(float(base_value), 4),
            top_features=[c.to_dict() for c in top],
            explanation=text.strip(),
            force_plot_path=plot_path,
        )
        log.info("shap_explained", customer_id=customer_id, proba=proba)
        return exp

    def _expected_value(self) -> float:
        """Model's expected output (SHAP base value); falls back to 0.0."""
        try:
            ev = getattr(self._explainer, "expected_value", None)
            if ev is None:
                return 0.0
            arr = np.asarray(ev).ravel()
            return float(arr[1]) if arr.size > 1 else float(arr[0])
        except Exception:
            return 0.0

    def _save_bar_plot(self, customer_id: str, top: list[FeatureContribution]) -> str | None:
        try:
            d = artifact_paths(self.config)["shap_plots"]
            d.mkdir(parents=True, exist_ok=True)
            feats = [c.feature for c in top][::-1]
            vals = [c.shap_value for c in top][::-1]
            colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
            plt.figure(figsize=(8, 4))
            plt.barh(feats, vals, color=colors)
            plt.axvline(0, color="black", linewidth=0.8)
            plt.title(f"SHAP contributions — {customer_id}")
            plt.tight_layout()
            p = d / f"{customer_id}_shap.png"
            plt.savefig(p)
            plt.close()
            return str(p)
        except Exception as e:
            log.info("shap_plot_failed", error=str(e))
            return None
