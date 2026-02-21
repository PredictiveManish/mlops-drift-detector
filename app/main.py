from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from datetime import datetime
from logger import log_prediction, get_recent_predictions
from drift_detector import compute_drift_score, should_alert
from discord_alert import send_alert
import asyncio

