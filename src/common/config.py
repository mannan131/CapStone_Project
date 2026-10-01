"""Shared helpers: config loading, rules loading, and path resolution."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def _load_config_cached(cfg_path_str: str) -> dict:
    with open(cfg_path_str, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config. Returns a deep copy so callers can't pollute the cache."""
    cfg_path = Path(path) if path else ROOT / "configs" / "config.yaml"
    return copy.deepcopy(_load_config_cached(str(cfg_path)))


def load_rules(path: str | Path | None = None, config: dict | None = None) -> dict:
    cfg = config or load_config()
    rec = cfg.get("recommendation_engine", cfg.get("recommendation", {}))
    default = str(ROOT / rec["rules_path"])
    rules_path = Path(path) if path else Path(default)
    with open(rules_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return ROOT / path


def project_seed(config: dict | None = None) -> int:
    """Random seed (new `project.random_seed`, legacy `project.seed` fallback)."""
    proj = (config or load_config()).get("project", {})
    return int(proj.get("random_seed", proj.get("seed", 42)))


def schema(cfg: dict) -> dict:
    """Data schema block (new `data_schema`, legacy `features` fallback)."""
    if "data_schema" in cfg:
        return cfg["data_schema"]
    feats = cfg.get("features", {})
    return {
        "id_column": "customer_id",
        "target_column": feats.get("target_col", "churn"),
        "behavioral_features": [],
        "transactional_features": [],
        "categorical_features": feats.get("categorical", []),
        "clv_column": "customer_lifetime_value",
    }


def artifact_paths(config: dict | None = None) -> dict[str, Path]:
    """Resolve model artifact locations.

    New-spec keys first (`paths.models_dir`, `scaler_path`, ...), legacy keys
    as fallback. Operational paths absent from the spec (plots, prediction db,
    model/pipeline/report files) default under `models/` and `artifacts/`.
    """
    cfg = config or load_config()
    p = cfg.get("paths", {})
    models_dir = str(p.get("models_dir", "models"))

    def _r(key: str, default: str) -> Path:
        return resolve_path(p.get(key, default))

    return {
        "best_model": _r("best_model_path", f"{models_dir}/best_model.joblib"),
        "fusion_pipeline": _r("fusion_pipeline_path", f"{models_dir}/fusion_pipeline.joblib"),
        "model_report": _r("model_report_path", f"{models_dir}/model_comparison.json"),
        "scaler": _r("scaler_path", f"{models_dir}/artifacts/scaler.joblib"),
        "encoder": _r("encoder_path", f"{models_dir}/artifacts/encoder.joblib"),
        "shap_plots": _r("shap_plots_dir", "artifacts/shap_plots"),
        "predictions_db": _r("predictions_db_path", "artifacts/predictions.db"),
        "processed_dir": _r("processed_data_dir", "data/processed"),
        "raw_customers_csv": _r(
            "raw_customer_data",
            p.get("synthetic_customers_csv", "data/synthetic/customers.csv"),
        ),
        "raw_images_dir": _r("raw_images_dir", "data/raw_images"),
    }
