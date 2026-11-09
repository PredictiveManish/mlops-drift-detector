import json
import sqlite3
from datetime import datetime

from app import config


def _connect():
    """Open the configured database.

    Read at call time rather than import time so tests and deployments can
    point DB_PATH somewhere else.
    """
    return sqlite3.connect(config.DB_PATH)


def init_db():
    conn = _connect()
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS predictions
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            features TEXT,
            prediction INTEGER,
            timestamp TEXT)"""
    )
    # The reference distribution, frozen once and never updated by traffic.
    # Comparing the recent window against the same table it came from is what
    # made every drift score 0.0.
    c.execute(
        """CREATE TABLE IF NOT EXISTS baseline
           (id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction INTEGER NOT NULL,
            frozen_at TEXT NOT NULL)"""
    )
    conn.commit()
    conn.close()


def log_prediction(features, pred, timestamp):
    conn = _connect()
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (features, prediction, timestamp) VALUES (?,?,?)",
        (json.dumps(features), pred, timestamp.isoformat()),
    )
    conn.commit()
    conn.close()


def get_recent_predictions(limit=100):
    conn = _connect()
    c = conn.cursor()
    c.execute("SELECT prediction FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]


def count_predictions():
    conn = _connect()
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    return total


# --------------------------------------------------------------- baseline
def get_baseline():
    """Return the frozen reference predictions (empty if not frozen yet)."""
    conn = _connect()
    c = conn.cursor()
    rows = c.execute("SELECT prediction FROM baseline ORDER BY id ASC").fetchall()
    conn.close()
    return [row[0] for row in rows]


def baseline_source():
    """'explicit' if an operator froze it, 'auto' if it was frozen on first use."""
    conn = _connect()
    c = conn.cursor()
    row = c.execute("SELECT frozen_at FROM baseline LIMIT 1").fetchone()
    conn.close()
    if not row:
        return "none"
    return "explicit" if row[0].startswith("explicit:") else "auto"


def freeze_baseline(limit=None, force=False, source="explicit"):
    """Freeze the earliest `limit` predictions as the reference distribution.

    Returns the number of rows frozen (0 if a baseline already exists and
    force is False). Re-freezing replaces the old reference.
    """
    limit = limit or config.WINDOW_SIZE
    conn = _connect()
    c = conn.cursor()
    existing = c.execute("SELECT COUNT(*) FROM baseline").fetchone()[0]
    if existing and not force:
        conn.close()
        return 0
    rows = c.execute(
        "SELECT prediction FROM predictions ORDER BY id ASC LIMIT ?", (limit,)
    ).fetchall()
    c.execute("DELETE FROM baseline")
    stamp = f"{source}:{datetime.now().isoformat()}"
    c.executemany(
        "INSERT INTO baseline (prediction, frozen_at) VALUES (?,?)",
        [(r[0], stamp) for r in rows],
    )
    conn.commit()
    conn.close()
    return len(rows)


def reset_baseline():
    conn = _connect()
    conn.execute("DELETE FROM baseline")
    conn.commit()
    conn.close()


def distribution(values):
    """Class counts as a dict keyed by string, for API responses."""
    return {str(k): values.count(k) for k in sorted(set(values))}
