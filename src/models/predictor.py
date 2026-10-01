"""ChurnPredictor wrapper around the best saved model."""

from __future__ import annotations

import joblib
import pandas as pd

from src.common.config import artifact_paths, load_config, schema
from src.feature_fusion.pipeline import FeatureFusionPipeline


class ChurnPredictor:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        paths = artifact_paths(self.config)
        s = schema(self.config)
        self.id_col = s["id_column"]
        self.target_col = s["target_column"]
        self.model = joblib.load(paths["best_model"])
        self.fusion: FeatureFusionPipeline = FeatureFusionPipeline.load(
            str(paths["fusion_pipeline"])
        )

    def _to_frame(self, record: dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(record, pd.DataFrame):
            return record
        return pd.DataFrame([record])

    def predict_proba(self, record: dict | pd.DataFrame) -> float:
        """Return churn probability as percentage 0-100."""
        df = self._to_frame(record)
        for drop in (self.id_col, self.target_col):
            if drop in df.columns:
                df = df.drop(columns=[drop])
        X = self.fusion.transform(df)
        p = float(self.model.predict_proba(X)[0, 1])
        return round(p * 100, 2)

    def predict_proba01(self, record: dict | pd.DataFrame) -> float:
        return self.predict_proba(record) / 100.0

    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        probs = []
        for _, row in df.iterrows():
            probs.append(self.predict_proba(row.to_dict()))
        out["churn_probability"] = probs
        return out
