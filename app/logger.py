import sqlite3
import json
from datetime import datetime

DB_FILE="predictions.db"

def init_db():
    conn=sqlite3.connect(DB_FILE)
    c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              features TEXT,
              prediciton INTEGER,
              timestamp TEXT''')
    conn.commit()
    conn.close()

def log_prediction(features, pred, timestamp):
    conn=sqlite3.connect(DB_FILE)
    c=conn.cursor()
    c.execute("INSERT INTO predictions (features, prediction, timestamp) VALUES (?,?,?)", (json.dumps(features), pred, timestamp.isformat()))
    conn.commit()
    conn.close()

def get_recent_predictions(limit=100):
    conn=sqlite3.connect(DB_FILE)
    c=conn.cursor()
    c.execute("SELECT prediction FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows=c.fetchall()
    conn.close()
    return [row[0] for row in rows]


def initial_get_predictions(limit=100):
    # For demo, using first 100 predictions as baseline
    conn=sqlite3.connect(DB_FILE)
    c=conn.cursor()
    c.execute("SELECT prediction FROM predictions ORDER BY id ASC LIMIT ?", (limit,))
    rows=c.fetchall()
    conn.close()
    return [row[0] for row in rows]

init_db() # ensures table exists


