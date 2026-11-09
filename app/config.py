# app/config.py
import os

# Alert threshold (15%)
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.15))

# Discord webhook URL. No default: a webhook baked into the source is a leaked
# secret, and anyone reading the repo can post to it.
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")

# Database path
DB_PATH = os.getenv("DB_PATH", "predictions.db")

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# --- Drift window -----------------------------------------------------------
# How many recent predictions to compare against the baseline.
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "100"))

# Minimum samples on each side before a score is meaningful. Below this the
# PSI is dominated by sampling noise, so we report "insufficient data" instead
# of a number that looks calm.
MIN_BASELINE_SAMPLES = int(os.getenv("MIN_BASELINE_SAMPLES", "30"))
MIN_RECENT_SAMPLES = int(os.getenv("MIN_RECENT_SAMPLES", "30"))
