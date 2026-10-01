"""Training pipeline: 4 model types, Optuna tuning, CV metrics, MLflow tracking."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import nullcontext

import joblib
import mlflow
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

from src.common.config import artifact_paths, load_config, project_seed, schema
from src.common.logging import get_logger
from src.data_ingestion.loader import CustomerDataLoader
from src.feature_fusion.pipeline import FeatureFusionPipeline
from src.models.torch_nn import TorchMLPClassifier

log = get_logger("training")
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_REGISTRY = ("logistic_regression", "random_forest", "xgboost", "neural_network")


def _finite(value: float, fallback: float = 0.0) -> float:
    """Guard against NaN/inf leaking into metrics JSON."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    return f if math.isfinite(f) else fallback


def compute_metrics(y_true, y_prob) -> dict:
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    out = {
        "accuracy": round(_finite(accuracy_score(y_true, y_pred)), 4),
        "precision": round(_finite(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(_finite(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(_finite(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    try:
        out["auc_roc"] = round(_finite(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        out["auc_roc"] = 0.0
    return out


def tune_rf(X, y, trials: int, seed: int, space: dict | None = None):
    space = space or {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                int(space.get("n_estimators_low", 50)),
                int(space.get("n_estimators_high", 200)),
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                int(space.get("max_depth_low", 3)),
                int(space.get("max_depth_high", 12)),
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
        }
        m = RandomForestClassifier(**params, random_state=seed)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores = cross_validate(m, X, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores["test_score"]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study.best_params


def tune_xgb(X, y, trials: int, seed: int, space: dict | None = None):
    space = space or {}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                int(space.get("n_estimators_low", 50)),
                int(space.get("n_estimators_high", 200)),
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                int(space.get("max_depth_low", 3)),
                int(space.get("max_depth_high", 8)),
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                float(space.get("learning_rate_low", 0.03)),
                float(space.get("learning_rate_high", 0.3)),
                log=True,
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        m = XGBClassifier(**params, random_state=seed, eval_metric="logloss", n_jobs=1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores = cross_validate(m, X, y, cv=cv, scoring="roc_auc")
        return float(np.mean(scores["test_score"]))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study.best_params


def _setup_mlflow(cfg: dict) -> bool:
    # Env override lets containers point at the compose MLflow service.
    uri = os.environ.get("MLFLOW_TRACKING_URI") or str(
        cfg.get("paths", {}).get("mlflow_tracking_uri", "")
    )
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        if uri.endswith(".db"):
            mlflow.set_tracking_uri(f"sqlite:///{uri}")
        else:
            mlflow.set_tracking_uri(uri or "file:./artifacts/mlruns")
        mlflow.set_experiment("churn-prediction")
        return True
    except Exception as e:
        # No local server running (e.g. http://localhost:5000 unreachable) -> file on.
        log.info("mlflow_disabled", error=str(e))
        return False


def train_all(config: dict | None = None) -> dict:
    cfg = config or load_config()
    mt = cfg.get("model_training", cfg.get("models", {}))
    seed = project_seed(cfg)
    trials = int(mt.get("optuna_trials", 10))
    cv_folds = int(mt.get("cv_folds", 3))
    test_size = float(mt.get("test_size", 0.15))
    val_size = float(mt.get("val_size", 0.15))
    stratify = bool(mt.get("stratify", True))
    primary_metric = str(mt.get("primary_metric", "roc_auc"))
    nn_cfg = mt.get("neural_network", mt.get("nn", {}))
    wanted = [m for m in mt.get("models_to_train", list(MODEL_REGISTRY)) if m in MODEL_REGISTRY]
    if not wanted:
        raise ValueError(f"models_to_train must list from {MODEL_REGISTRY}")

    _mlflow_ok = _setup_mlflow(cfg)
    paths = artifact_paths(cfg)
    s = schema(cfg)
    id_col, target = s["id_column"], s["target_column"]

    loader = CustomerDataLoader(cfg)
    df = loader.load_default()
    X_raw = df.drop(columns=[target, id_col])
    y = df[target].astype(int).values

    strat_y = y if stratify else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=seed, stratify=strat_y
    )
    X_val_raw, y_val = None, None
    if val_size > 0:
        # val_size is a fraction of the remaining (post-test) data.
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_train_raw,
            y_train,
            test_size=val_size,
            random_state=seed,
            stratify=y_train if stratify else None,
        )
    fusion = FeatureFusionPipeline(cfg)
    X_train = fusion.fit_transform(X_train_raw, y_train)
    X_test = fusion.transform(X_test_raw)
    X_val = fusion.transform(X_val_raw) if X_val_raw is not None else None

    # Persist processed splits for auditability.
    proc = paths["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    X_train_raw.assign(**{target: y_train}).to_csv(proc / "train.csv", index=False)
    X_test_raw.assign(**{target: y_test}).to_csv(proc / "test.csv", index=False)
    if X_val_raw is not None:
        X_val_raw.assign(**{target: y_val}).to_csv(proc / "val.csv", index=False)

    factories: dict[str, object] = {}
    if "logistic_regression" in wanted:
        factories["logistic_regression"] = LogisticRegression(max_iter=1000, random_state=seed)
    if "random_forest" in wanted:
        factories["random_forest"] = RandomForestClassifier(n_estimators=100, random_state=seed)
    if "xgboost" in wanted:
        factories["xgboost"] = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=1,
        )
    if "neural_network" in wanted:
        hidden = nn_cfg.get("hidden_layers", [nn_cfg.get("hidden_dim", 64)])
        factories["neural_network"] = TorchMLPClassifier(
            hidden_layers=tuple(hidden),
            dropout=float(nn_cfg.get("dropout", 0.3)),
            epochs=int(nn_cfg.get("epochs", 50)),
            lr=float(nn_cfg.get("lr", 1e-3)),
            batch_size=int(nn_cfg.get("batch_size", 32)),
            early_stopping_patience=int(nn_cfg.get("early_stopping_patience", 5)),
            random_state=seed,
        )
    # Optuna tuning for tree models (search spaces live under model_training.*_param_space).
    if "random_forest" in factories:
        rf_best = tune_rf(X_train, y_train, trials, seed, mt.get("rf_param_space"))
        factories["random_forest"] = RandomForestClassifier(**rf_best, random_state=seed)
    if "xgboost" in factories:
        xgb_best = tune_xgb(X_train, y_train, trials, seed, mt.get("xgb_param_space"))
        factories["xgboost"] = XGBClassifier(
            **xgb_best, random_state=seed, eval_metric="logloss", n_jobs=1
        )

    metric_key = {"roc_auc": "auc_roc"}.get(primary_metric, primary_metric)
    results: dict = {}
    for name, model in factories.items():
        ctx = mlflow.start_run(run_name=name) if _mlflow_ok else nullcontext()
        try:
            with ctx:
                if _mlflow_ok:
                    try:
                        mlflow.log_params(getattr(model, "get_params", lambda: {})())
                    except Exception:
                        pass
                assert hasattr(model, "fit") and hasattr(model, "predict_proba")
                model.fit(X_train, y_train)  # type: ignore
                proba = model.predict_proba(X_test)[:, 1]  # type: ignore
                metrics = compute_metrics(y_test, proba)
                if X_val is not None:
                    val_proba = model.predict_proba(X_val)[:, 1]  # type: ignore
                    val_metrics = compute_metrics(y_val, val_proba)
                    metrics["val_auc_roc"] = val_metrics["auc_roc"]
                if _mlflow_ok:
                    try:
                        mlflow.log_metrics(metrics)
                    except Exception:
                        pass
                # Cross-validated primary metric.
                try:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
                    scorer = "roc_auc" if metric_key == "auc_roc" else metric_key
                    cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scorer)
                    metrics["cv_primary_mean"] = round(_finite(np.mean(cv_scores["test_score"])), 4)
                except Exception:
                    metrics["cv_primary_mean"] = metrics.get(metric_key, 0.0)
                results[name] = {"metrics": metrics, "model": model}
                log.info("model_trained", name=name, **metrics)
        except Exception as e:
            log.info("model_failed", name=name, error=str(e))

    if not results:
        raise RuntimeError("All model trainings failed.")
    best_name = max(results, key=lambda k: results[k]["metrics"].get(metric_key, 0.0))
    best_model = results[best_name]["model"]
    paths["best_model"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, paths["best_model"])
    fusion.save()
    report = {
        "best_model": best_name,
        "primary_metric": primary_metric,
        "models": {k: v["metrics"] for k, v in results.items()},
        "feature_names": fusion.get_feature_names(),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)) if y_val is not None else 0,
        "n_test": int(len(y_test)),
    }
    with open(paths["model_report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info("training_complete", best_model=best_name, report=report)
    print(f"Best model: {best_name} | {results[best_name]['metrics']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    train_all(cfg)


if __name__ == "__main__":
    main()
