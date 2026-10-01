"""FastAPI app: predict / explain / recommend / upload-image / health."""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api import db
from src.api.schemas import (
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictWithImageRequest,
    RecommendResponse,
    UploadResponse,
)
from src.common.config import load_config, schema
from src.common.logging import get_logger
from src.data_ingestion.image_upload import ImageUploadHandler
from src.data_ingestion.loader import CustomerDataLoader
from src.explainability.shap_explainer import ShapExplainer
from src.image_quality.analyzer import ImageQualityAnalyzer
from src.models.predictor import ChurnPredictor
from src.recommendation_engine.recommender import RetentionRecommender

config = load_config()
log = get_logger("api", level=config.get("api", {}).get("log_level", "info"))

predictor: ChurnPredictor | None = None
explainer: ShapExplainer | None = None
recommender: RetentionRecommender | None = None
loader: CustomerDataLoader | None = None


def _load_state() -> None:
    global predictor, explainer, recommender, loader
    predictor = ChurnPredictor(config)
    explainer = ShapExplainer(config)
    recommender = RetentionRecommender(config)
    loader = CustomerDataLoader(config)
    db.init_db(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_state()
        log.info("api_startup_ok")
    except Exception as e:
        log.info("api_startup_deferred", error=str(e))
    yield


app = FastAPI(title="Churn Intelligence Engine", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _require_state():
    if predictor is None or explainer is None or recommender is None:
        try:
            _load_state()
        except Exception as e:
            raise HTTPException(
                status_code=503, detail=f"Model artifacts not ready. Train first: {e}"
            )
    assert predictor is not None and explainer is not None and recommender is not None
    return predictor, explainer, recommender


def _risk_band(p: float) -> str:
    if p >= 70:
        return "high"
    if p >= 40:
        return "medium"
    return "low"


def _lookup_customer(customer_id: str) -> dict:
    global loader
    if loader is None:
        loader = CustomerDataLoader(config)
    s = schema(config)
    id_col, target_col = s["id_column"], s["target_column"]
    try:
        df = loader.load_default()
        hit = df[df[id_col] == customer_id]
        if not hit.empty:
            row = hit.iloc[0].to_dict()
            row.pop(target_col, None)
            return row
    except Exception:
        pass
    raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        _require_state()
        ok = True
    except HTTPException:
        ok = False
    return HealthResponse(status="ok", version="1.0.0", model_loaded=ok)


@app.post("/predict-churn", response_model=PredictResponse)
def predict_churn(req: PredictRequest):
    pred, _, _ = _require_state()
    try:
        proba = pred.predict_proba(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    db.log_prediction(req.customer_id, proba, config)
    log.info("predict", customer_id=req.customer_id, proba=proba)
    return PredictResponse(
        customer_id=req.customer_id, churn_probability=proba, risk_band=_risk_band(proba)
    )


@app.get("/explain/{customer_id}", response_model=ExplainResponse)
def explain(customer_id: str):
    pred, exp, _ = _require_state()
    record = _lookup_customer(customer_id)
    proba = pred.predict_proba(record)
    explanation = exp.explain_customer(record, customer_id=customer_id)
    return ExplainResponse(
        customer_id=customer_id,
        churn_probability=proba,
        base_value=explanation.base_value,
        top_features=explanation.top_features,
        explanation=explanation.explanation,
    )


@app.get("/recommend/{customer_id}", response_model=RecommendResponse)
def recommend(customer_id: str):
    pred, exp, rec = _require_state()
    record = _lookup_customer(customer_id)
    proba = pred.predict_proba(record)
    explanation = exp.explain_customer(record, customer_id=customer_id)
    features = {k: v for k, v in record.items() if k != "customer_id"}
    recommendation = rec.recommend(customer_id, proba, features, explanation.top_features)
    return RecommendResponse(**recommendation.to_dict())


@app.post("/upload-image", response_model=UploadResponse)
async def upload_image(customer_id: str, product_id: str = "default", file: UploadFile = File(...)):
    handler = ImageUploadHandler(config)
    try:
        data = await file.read()
        dest = handler.save_bytes(data, customer_id, product_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    analyzer = ImageQualityAnalyzer(config)
    try:
        report = analyzer.analyze(dest)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return UploadResponse(
        customer_id=customer_id,
        product_id=product_id,
        image_path=str(dest),
        quality_score=report.quality_score,
        report=report.to_dict(),
    )


@app.post("/predict-with-image")
def predict_with_image(req: PredictWithImageRequest):
    """Predict churn fusing an optional base64 product image (JSON body)."""
    pred, _, _ = _require_state()
    fusion_cfg = config.get("feature_fusion", {})
    img_col = fusion_cfg.get("image_score_column", fusion_cfg.get("image_score_col"))
    features = req.model_dump(exclude={"image_b64"})
    if req.image_b64:
        handler = ImageUploadHandler(config)
        try:
            b64 = req.image_b64
            raw = base64.b64decode(b64.split(",", 1)[-1] if b64.startswith("data:") else b64)
            dest = handler.save_bytes(raw, req.customer_id, "upload")
            quality = ImageQualityAnalyzer(config).analyze(dest).quality_score
            features[img_col] = quality
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing failed: {e}")
    proba = pred.predict_proba(features)
    return {
        "customer_id": req.customer_id,
        "churn_probability": proba,
        "risk_band": _risk_band(proba),
        img_col: features.get(img_col),
    }
