"""Feature fusion: scale numerics, encode categoricals, merge image quality score."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.common.config import artifact_paths, load_config, schema
from src.common.logging import get_logger
from src.recommendation_engine.rules import extra_rule_features

log = get_logger("feature-fusion")


def _rule_extra_features(config: dict) -> list[str]:
    """Numeric features referenced by cause->action rules but absent from the schema.

    Keeps rule signals in the model without hardcoding column names.
    """
    try:
        return extra_rule_features(config)
    except Exception:
        return []


def _scaler(method: str):
    if method == "minmax":
        return MinMaxScaler()
    if method == "standard":
        return StandardScaler()
    raise ValueError(f"Unknown scaling_method: {method!r} (expected standard|minmax)")


def _encoder(method: str):
    if method == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if method == "target":
        from sklearn.preprocessing import TargetEncoder

        return TargetEncoder()
    raise ValueError(f"Unknown encoding_method: {method!r} (expected onehot|target)")


class FeatureFusionPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        s = schema(self.config)
        cats = list(s.get("categorical_features", []))
        tx = list(s.get("transactional_features", []))
        # Numeric = schema behaviorals + non-categorical transactionals + CLV
        # + any extra numerics referenced by the cause->action rules.
        self.numeric_features: list[str] = [
            f
            for f in list(s.get("behavioral_features", []))
            + [f for f in tx if f not in cats]
            + [s.get("clv_column", "customer_lifetime_value")]
            + _rule_extra_features(self.config)
            if f
        ]
        self.categorical_features: list[str] = cats
        fusion_cfg = self.config.get("feature_fusion", {})
        self.image_col: str = fusion_cfg.get(
            "image_score_column", fusion_cfg.get("image_score_col", "image_quality_score")
        )
        self.target_col: str = s.get("target_column", "churn")
        iq = self.config.get("image_quality", {})
        legacy_default = self.config.get("data", {}).get("missing_image_default_quality", 0.5)
        self.default_quality: float = float(iq.get("missing_image_default_score", legacy_default))
        self.scaling_method: str = fusion_cfg.get("scaling_method", "standard")
        self.encoding_method: str = fusion_cfg.get("encoding_method", "onehot")
        self._pre: ColumnTransformer | None = None
        self.feature_names_out_: list[str] | None = None

    def _build(self) -> ColumnTransformer:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", _scaler(self.scaling_method)),
            ]
        )
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _encoder(self.encoding_method)),
            ]
        )
        cols_num = self.numeric_features + [self.image_col]
        return ColumnTransformer(
            [("num", num_pipe, cols_num), ("cat", cat_pipe, self.categorical_features)]
        )

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.image_col not in X.columns:
            X[self.image_col] = self.default_quality
        else:
            X[self.image_col] = X[self.image_col].fillna(self.default_quality)
        for c in self.numeric_features + [self.image_col]:
            if c not in X.columns:
                X[c] = 0.0
        for c in self.categorical_features:
            if c not in X.columns:
                X[c] = "unknown"
            X[c] = X[c].fillna("unknown").astype(str)
        return X

    def fit(self, X: pd.DataFrame, y=None):
        if self.encoding_method == "target" and y is None:
            raise ValueError("Target encoding requires y; pass y to fit().")
        Xp = self._prepare(X)
        self._pre = self._build()
        self._pre.fit(Xp, y)
        self.feature_names_out_ = list(self._pre.get_feature_names_out())
        log.info("fusion_fitted", n_features=len(self.feature_names_out_))
        return self

    def transform(self, X: pd.DataFrame):
        if self._pre is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        return self._pre.transform(self._prepare(X))

    def get_feature_names(self) -> list[str]:
        if self.feature_names_out_ is None:
            raise RuntimeError("Pipeline not fitted.")
        # strip transformer prefix e.g. "num__login_frequency_last_30d"
        return [n.split("__", 1)[-1] for n in self.feature_names_out_]

    def save(self, path: str | None = None) -> str:
        """Persist the pipeline plus standalone scaler/encoder artifacts (spec paths)."""
        paths = artifact_paths(self.config)
        pipe_path = path or str(paths["fusion_pipeline"])
        joblib.dump(self, pipe_path)
        if self._pre is not None and "num" in self._pre.named_transformers_:
            num_pipe = self._pre.named_transformers_["num"]
            cat_pipe = self._pre.named_transformers_["cat"]
            scaler_p, encoder_p = paths["scaler"], paths["encoder"]
            scaler_p.parent.mkdir(parents=True, exist_ok=True)
            encoder_p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(num_pipe, scaler_p)
            joblib.dump(cat_pipe, encoder_p)
            log.info("fusion_saved", path=pipe_path)
            log.info("scaler_saved", path=str(scaler_p))
            log.info("encoder_saved", path=str(encoder_p))
        else:
            log.info("fusion_saved", path=pipe_path)
        return pipe_path

    @classmethod
    def load(cls, path: str) -> "FeatureFusionPipeline":
        return joblib.load(path)
