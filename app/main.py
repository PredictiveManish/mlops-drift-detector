import logging
import os
import pickle
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app import config
from app.discord_alert import send_alert
from app.drift_detector import compute_drift_score, should_alert
from app.logger import (
    baseline_source,
    count_predictions,
    distribution,
    freeze_baseline,
    get_baseline,
    get_recent_predictions,
    init_db,
    log_prediction,
    reset_baseline,
)

logging.basicConfig(level=logging.INFO)

# Initialize database on startup
init_db()

app = FastAPI(title="MLOps Drift Detection API")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None
if not os.path.exists(model_path):
    logging.warning(f"Model file not found at {model_path}")
else:
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as exc:  # corrupt or incompatible pickle
        logging.error(f"Could not load model: {exc}")


class InputData(BaseModel):
    features: List[float] = Field(..., min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    prediction: int
    timestamp: str


class DriftResponse(BaseModel):
    drift_score: Optional[float]
    alert: bool
    threshold: float
    recent_samples: int
    baseline_samples: int
    sufficient_data: bool
    baseline_source: str
    method: str = "psi"


class BaselineResponse(BaseModel):
    frozen: int
    source: str
    baseline_samples: int


@app.get("/")
async def root():
    return {
        "message": "MLOps Drift Detection API",
        "status": "operational",
        "model_loaded": model is not None,
    }


@app.get("/healthz")
async def healthz():
    """Health check endpoint for Render"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: InputData, background_tasks: BackgroundTasks):
    """Make a prediction and log it for drift detection"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and len(data.features) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected} features, got {len(data.features)}",
        )

    features = np.array(data.features).reshape(1, -1)
    pred = model.predict(features)[0]

    timestamp = datetime.now()
    background_tasks.add_task(log_prediction, data.features, int(pred), timestamp)

    return {"prediction": int(pred), "timestamp": timestamp.isoformat()}


@app.post("/baseline", response_model=BaselineResponse)
async def freeze_reference(
    limit: Optional[int] = Query(None, ge=1, le=10000),
    force: bool = Query(False),
):
    """Freeze the current earliest predictions as the reference distribution.

    This is the explicit version of what /drift_score does on first use. Call
    it once the traffic you have seen represents normal operation, and again
    after a deliberate distribution change.
    """
    frozen = freeze_baseline(limit=limit, force=force, source="explicit")
    return BaselineResponse(
        frozen=frozen,
        source=baseline_source(),
        baseline_samples=len(get_baseline()),
    )


@app.delete("/baseline", response_model=BaselineResponse)
async def clear_reference():
    """Forget the reference distribution."""
    reset_baseline()
    return BaselineResponse(frozen=0, source="none", baseline_samples=0)


@app.get("/drift_score", response_model=DriftResponse)
async def get_drift(background_tasks: BackgroundTasks):
    """Compare the recent window against the frozen reference distribution.

    The baseline is a separate, immutable table. It is never a slice of the
    same window it is being compared against - that made every score 0.0.
    """
    recent = get_recent_predictions(limit=config.WINDOW_SIZE)
    baseline = get_baseline()

    # First use: freeze the earliest predictions once, and say so in the
    # response so nobody mistakes it for a curated reference.
    if not baseline and len(recent) >= config.MIN_BASELINE_SAMPLES:
        freeze_baseline(limit=config.WINDOW_SIZE, source="auto")
        baseline = get_baseline()

    source = baseline_source()
    enough = (
        len(baseline) >= config.MIN_BASELINE_SAMPLES
        and len(recent) >= config.MIN_RECENT_SAMPLES
    )

    if not enough:
        return DriftResponse(
            drift_score=None,
            alert=False,
            threshold=config.ALERT_THRESHOLD,
            recent_samples=len(recent),
            baseline_samples=len(baseline),
            sufficient_data=False,
            baseline_source=source,
        )

    score = compute_drift_score(baseline, recent, method="psi")
    alert = should_alert(score, config.ALERT_THRESHOLD)

    if alert:
        background_tasks.add_task(send_alert, f"🚨 Drift detected! Score: {score:.3f}")

    return DriftResponse(
        drift_score=round(score, 3),
        alert=alert,
        threshold=config.ALERT_THRESHOLD,
        recent_samples=len(recent),
        baseline_samples=len(baseline),
        sufficient_data=True,
        baseline_source=source,
    )


@app.get("/stats")
async def get_stats():
    """Prediction and baseline statistics"""
    recent = get_recent_predictions(limit=config.WINDOW_SIZE)
    baseline = get_baseline()

    return {
        "total_predictions": count_predictions(),
        "window_size": config.WINDOW_SIZE,
        "baseline_size": len(baseline),
        "baseline_source": baseline_source(),
        "recent_distribution": distribution(recent) if recent else {},
        "baseline_distribution": distribution(baseline) if baseline else {},
    }
