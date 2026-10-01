# Architecture

Pipeline stages (matches README mermaid diagram). The two YAML files are the single
source of truth and are read at runtime — no schema or rule values are hardcoded:
`configs/config.yaml` (v0.1.0 spec: `data_schema`, `feature_fusion`, `model_training`,
`shap`, `recommendation_engine`, `api`, `dashboard`) and
`configs/cause_action_rules.yaml` (causes, conditions, actions, fallbacks).

1. **Data ingestion** (`src/data_ingestion`): `CustomerDataLoader` validates CSV rows with
   a pydantic `CustomerRecord` **generated from `data_schema` at import** (names + types
   from config; extra columns allowed), joins behavioral + transactional sources on the
   schema `id_column`. The synthetic generator emits exactly the schema columns (plus
   rule-referenced extras, first of which carries the recency signal).
   `ImageUploadHandler` accepts path/bytes/base64/URL, validates extension + size, normalizes
   to JPEG under `data/raw_images/{customer_id}/{product_id}.jpg`, deletes partial files,
   and rejects corrupt files.
2. **Image quality** (`src/image_quality`): variance-of-Laplacian blur, pixel-count resolution
   (against `min_resolution`/`target_resolution`), grayscale-std contrast (against
   `contrast_min_std`), Sobel-gradient sharpness → weighted composite `quality_score ∈ [0,1]`
   (`image_quality.weights`). Returns `QualityReport`.
3. **Feature fusion** (`src/feature_fusion`): sklearn-compatible transformer; median-imputes +
   scales numerics (`scaling_method`: `standard`|`minmax`, incl. the image score column with
   `missing_image_default_score` imputation), most-frequent-imputes + encodes categoricals
   (`encoding_method`: `onehot`|`target` via sklearn's `TargetEncoder`, which needs `y` at fit).
   Numeric extras are **derived at runtime from rule conditions** absent from the schema
   (`extra_rule_features()`), never hardcoded.
   Persists the full pipeline plus standalone scaler/encoder artifacts per `paths`.
4. **ML prediction** (`src/models`): stratified train/val/test split (`test_size`/`val_size`),
   Optuna-tuned (`optuna_trials`) RandomForest + XGBoost over the subset in `models_to_train`,
   plain LogisticRegression + PyTorch MLP (`hidden_layers`, `dropout`, `early_stopping_patience`;
   sklearn-MLP fallback with manual patience stopping when torch is absent).
   `cv_folds`-fold CV on the `primary_metric`, MLflow tracking, best model +
   `model_comparison.json` under `models/`, processed splits under `data/processed/`.
   `ChurnPredictor.predict_proba()` → 0–100%.
5. **Explainability** (`src/explainability`): explainer per `shap.explainer_type_map`
   (`tree` for RF/XGBoost, `kernel` for LR, `deep` for the NN with kernel fallback);
   top-`top_n_features` with direction, narrative string, matplotlib plot
   saved under `artifacts/shap_plots/`. Backgrounds are real sampled customers.
6. **Recommendation** (`src/recommendation_engine`): YAML rule engine maps causes
   (e.g. `low_image_quality`, `late_payments`) to ranked actions; orchestrator merges rule
   hits with SHAP corroboration — the SHAP feature→cause map is **built at runtime from
   the rules YAML** (`feature_to_cause_map()`), including SHAP-only causes, which receive
   their playbook actions — into `RetentionRecommendation`, truncated to
   `max_actions_per_customer`, with `priority_score` from the configured `priority_formula`
   (`churn_probability * customer_lifetime_value`).
7. **API** (`src/api`): FastAPI + pydantic v2, where `PredictRequest` **subclasses the
   config-generated ingest model** (same runtime contract as validation), SQLite prediction
   log, OpenAPI at `/docs`, log level from `api.log_level`.
8. **Dashboard** (`src/dashboard`): Streamlit gauge + SHAP waterfall + actions + bulk ranking
   (default top-N from `dashboard.top_n_at_risk_default`, priority via the same formula) +
   CSV export.

## As-built notes (v0.1.0)
- **Best-model selection is automatic** by `primary_metric` (`roc_auc`) on the held-out
  test split — the winner varies with the data seed (random_forest @ 0.735 in the latest
  verified run; logistic_regression won earlier runs). No model is hardcoded as best.
- **torch is optional**: Docker images install CPU torch; elsewhere the NN uses the
  sklearn-MLP fallback with manual patience stopping. Torch-only branches are excluded
  from coverage for this reason.
- **Containers self-bootstrap**: `docker/api_entrypoint.py` generates data + trains on
  first boot when `models/` or the CSV is missing; `docker/dashboard_entrypoint.py`
  waits for a healthy API. `models/`, `data/`, `artifacts/` are compose-mounted volumes
  (SQLite prediction log, no separate database server). Healthchecks gate startup order.
- **Prediction store** is SQLite (`artifacts/predictions.db`), satisfying the
  Postgres/SQLite option in favor of zero-ops locality.
