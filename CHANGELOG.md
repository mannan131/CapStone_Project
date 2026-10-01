# Changelog

All notable changes to the Customer Churn Prediction & Retention Intelligence Engine.

## [0.1.0] — 2026-09-22 — Initial release
- **Config-driven contracts**: `configs/config.yaml` (v0.1.0: data schema, fusion,
  training, SHAP, recommender, API, dashboard) and `configs/cause_action_rules.yaml`
  are the single source of truth. Pydantic schemas, fusion feature lists, SHAP
  cause mapping, and API fields are all derived from them at runtime.
- **Data ingestion**: pydantic-validated customer CSV loader (schema-required columns,
  join on id column), image upload handler (path/bytes/base64/URL, type/size/corrupt
  checks, partial-file cleanup), synthetic generator (600 customers, churn ~0.34,
  sample product images).
- **Image quality**: blur (Laplacian variance), resolution, contrast, sharpness →
  weighted 0–1 composite `QualityReport`; zero-weight config guard.
- **Feature fusion**: sklearn-compatible pipeline, standard/minmax scaling, onehot/target
  encoding, image-score imputation; rule-referenced extras auto-included; persists
  pipeline plus standalone scaler/encoder artifacts.
- **Model training**: Logistic Regression, Random Forest, XGBoost (Optuna-tuned),
  PyTorch MLP (`[64, 32, 16]`, dropout 0.3, patience-5 early stopping; sklearn-MLP
  fallback with manual early stopping when torch is absent). Stratified 433/77/90
  train/val/test split, 5-fold CV on ROC-AUC, MLflow tracking (SQLite/remote with
  offline fallback), best-model auto-selection, NaN-safe JSON report.
- **Explainability**: per-model explainer map (tree/kernel/deep→kernel fallback),
  top-5 features with direction, narrative string, saved plots, real-data backgrounds.
- **Recommendation engine**: YAML rule engine (8 causes + fallbacks), SHAP
  corroboration including SHAP-only causes with playbook actions, max 3 actions per
  customer, safe-evaluated priority formula, confidence scores.
- **API**: FastAPI + OpenAPI docs — predict, explain, recommend, image upload,
  predict-with-image, health with model-loaded flag; SQLite prediction log.
- **Dashboard**: Streamlit churn gauge, SHAP waterfall, ranked actions, bulk at-risk
  ranking with priority formula, CSV export.
- **Deployment**: multi-stage Dockerfiles with self-bootstrapping entrypoints
  (first boot generates data + trains with zero intervention), compose stack
  (API + dashboard + MLflow) with healthchecks, `.dockerignore`.
- **Quality gates**: 38 pytest tests at 82% coverage; CI runs black, isort, flake8,
  mypy, and pytest — all green.
