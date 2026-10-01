"""Pydantic schemas built at runtime from the config data contract.

Field NAMES and TYPES come from ``configs/config.yaml:data_schema`` — no column
names are hardcoded here. Rename a column in the config and the models follow.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.common.config import load_config, schema


def _spec_parts(cfg: dict | None = None):
    cfg = cfg or load_config()
    s = schema(cfg)
    fusion = cfg.get("feature_fusion", {})
    img_col = fusion.get("image_score_column", fusion.get("image_score_col", "image_quality_score"))
    cats = list(s.get("categorical_features", []))
    tx = list(s.get("transactional_features", []))
    numerics = (
        list(s.get("behavioral_features", []))
        + [f for f in tx if f not in cats]
        + [s.get("clv_column", "customer_lifetime_value")]
    )
    return s, img_col, [c for c in cats if c], [n for n in numerics if n]


def build_record_model(cfg: dict | None = None) -> type[BaseModel]:
    """Strict row validator: schema fields required (target/image optional)."""
    cfg = cfg or load_config()
    s, img_col, cats, numerics = _spec_parts(cfg)
    fields: dict = {s["id_column"]: (str, Field(..., min_length=1))}
    for f in numerics:
        fields[f] = (float, Field(...))
    for f in cats:
        fields[f] = (str, Field(...))
    fields[img_col] = (Optional[float], Field(default=None, ge=0, le=1))
    fields[s["target_column"]] = (Optional[int], Field(default=None))
    return create_model(
        "CustomerRecord",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def build_ingest_request_model(cfg: dict | None = None) -> type[BaseModel]:
    """Prediction-input model: id required, everything else has neutral defaults."""
    cfg = cfg or load_config()
    s, img_col, cats, numerics = _spec_parts(cfg)
    fields: dict = {s["id_column"]: (str, Field(..., min_length=1))}
    for f in numerics:
        fields[f] = (float, 0.0)
    for f in cats:
        fields[f] = (str, "")
    fields[img_col] = (Optional[float], None)
    return create_model(
        "CustomerIngestRequest",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


CustomerRecord: Any = build_record_model()
CustomerIngestRequest: Any = build_ingest_request_model()
