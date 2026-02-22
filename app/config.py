# app/config.py
import os

# Alert threshold (15%)
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", 0.15))

# Discord webhook URL
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")

# Database path
DB_PATH = os.getenv("DB_PATH", "predictions.db")

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")