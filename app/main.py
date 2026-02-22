from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os
from datetime import datetime
from typing import List, Optional
import json

# Import your modules
from app.logger import log_prediction, get_recent_predictions, get_initial_predictions, init_db
from app.drift_detector import compute_drift_score, should_alert
from app.discord_alert import send_alert
from app import config

# Initialize database on startup
init_db()

app = FastAPI(title="MLOps Drift Detection API")

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
if not os.path.exists(model_path):
    print(f"Warning: Model file not found at {model_path}")
    model = None
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

class InputData(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    timestamp: str

class DriftResponse(BaseModel):
    drift_score: float
    alert: bool
    threshold: float
    recent_samples: int
    baseline_samples: int

@app.get("/")
async def root():
    return {
        "message": "MLOps Drift Detection API",
        "status": "operational",
        "model_loaded": model is not None
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
    
    # Convert to numpy array and reshape
    features = np.array(data.features).reshape(1, -1)
    
    # Make prediction
    pred = model.predict(features)[0]
    
    # Log prediction asynchronously
    timestamp = datetime.now()
    background_tasks.add_task(log_prediction, data.features, int(pred), timestamp)
    
    return {
        "prediction": int(pred),
        "timestamp": timestamp.isoformat()
    }

@app.get("/drift_score", response_model=DriftResponse)
async def get_drift(background_tasks: BackgroundTasks):
    """Calculate current drift score"""
    # Get recent predictions (last 100)
    recent = get_recent_predictions(limit=100)
    
    # Get baseline predictions (first 100)
    baseline = get_initial_predictions(limit=100)
    
    if len(recent) < 10 or len(baseline) < 10:
        return {
            "drift_score": 0.0,
            "alert": False,
            "threshold": config.ALERT_THRESHOLD,
            "recent_samples": len(recent),
            "baseline_samples": len(baseline)
        }
    
    # Compute drift score
    score = compute_drift_score(baseline, recent, method='psi')
    alert = should_alert(score, config.ALERT_THRESHOLD)
    
    # Send alert if needed
    if alert:
        background_tasks.add_task(send_alert, f"🚨 Drift detected! Score: {score:.3f}")
    
    return {
        "drift_score": round(score, 3),
        "alert": alert,
        "threshold": config.ALERT_THRESHOLD,
        "recent_samples": len(recent),
        "baseline_samples": len(baseline)
    }

@app.get("/stats")
async def get_stats():
    """Get prediction statistics"""
    recent = get_recent_predictions(limit=1000)
    baseline = get_initial_predictions(limit=100)
    
    return {
        "total_predictions": len(recent),
        "baseline_size": len(baseline),
        "recent_distribution": {
            "0": recent.count(0) if recent else 0,
            "1": recent.count(1) if recent else 0,
            "2": recent.count(2) if recent else 0
        } if recent else {}
    }