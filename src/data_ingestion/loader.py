"""Customer data loader: CSV ingestion + pydantic validation + join on customer_id."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.common.config import artifact_paths, load_config, resolve_path, schema
from src.common.logging import get_logger
from src.data_ingestion.schemas import CustomerRecord

log = get_logger("data-ingestion")


def required_columns(config: dict | None = None) -> set[str]:
    """Required columns derived from the config `data_schema` block."""
    cfg = config or load_config()
    s = schema(cfg)
    cols = (
        [s["id_column"], s["target_column"], s["clv_column"]]
        + list(s.get("behavioral_features", []))
        + list(s.get("transactional_features", []))
    )
    fusion = cfg.get("feature_fusion", {})
    img_col = fusion.get("image_score_column", fusion.get("image_score_col"))
    if img_col:
        cols.append(img_col)
    return set(cols)


class CustomerDataLoader:
    def __init__(self, config: dict | None = None):
        self.config = config or load_config()

    def load_csv(self, path: str | Path, validate: bool = True) -> pd.DataFrame:
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Customer CSV not found: {p}")
        df = pd.read_csv(p)
        missing = required_columns(self.config) - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        if validate:
            self.validate(df)
        log.info("csv_loaded", path=str(p), rows=len(df))
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        errors: list[str] = []
        for i, row in df.iterrows():
            try:
                CustomerRecord(**row.to_dict())
            except ValidationError as e:
                errors.append(f"row {i}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
        if errors:
            raise ValueError("Validation failed: " + "; ".join(errors[:10]))
        return df

    def join_sources(self, behavioral: pd.DataFrame, transactional: pd.DataFrame) -> pd.DataFrame:
        """Join two frames on customer_id (outer join, dedupe)."""
        id_col = schema(self.config)["id_column"]
        merged = pd.merge(behavioral, transactional, on=id_col, how="outer", suffixes=("", "_tx"))
        # Coalesce duplicated columns
        for col in list(merged.columns):
            if col.endswith("_tx"):
                base = col[:-3]
                if base in merged.columns:
                    merged[base] = merged[base].combine_first(merged[col])
                else:
                    merged[base] = merged[col]
                merged.drop(columns=[col], inplace=True)
        return merged

    def load_default(self) -> pd.DataFrame:
        return self.load_csv(artifact_paths(self.config)["raw_customers_csv"])
