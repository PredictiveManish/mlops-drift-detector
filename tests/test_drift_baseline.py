"""Regression tests for drift scoring.

The service used to compare the last 100 predictions against the first 100 of
the same table. Under 200 rows those are the same multiset, and PSI is
order-invariant, so the score was always 0.0 - the demo printed "no drift"
while the input distribution was changing underneath it.

These tests fail against that design and pass only when the baseline is a
separate, frozen reference.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app import config, logger
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client backed by a throwaway database."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "predictions.db"))
    logger.init_db()
    with TestClient(app) as c:
        yield c


def seed(count, prediction):
    for _ in range(count):
        logger.log_prediction([0.0, 0.0, 0.0, 0.0], prediction, datetime.now())


def test_no_data_is_reported_as_insufficient(client):
    body = client.get("/drift_score").json()
    assert body["sufficient_data"] is False
    assert body["drift_score"] is None
    assert body["alert"] is False


def test_identical_distributions_score_near_zero(client):
    seed(40, 0)
    client.post("/baseline?force=true")
    seed(40, 0)

    body = client.get("/drift_score").json()
    assert body["sufficient_data"] is True
    assert body["drift_score"] < 0.05
    assert body["alert"] is False


def test_score_rises_when_the_distribution_shifts(client):
    """The test that the old design could never pass."""
    seed(40, 0)
    client.post("/baseline?force=true")
    seed(40, 2)

    body = client.get("/drift_score").json()
    assert body["sufficient_data"] is True
    assert body["drift_score"] > config.ALERT_THRESHOLD
    assert body["alert"] is True


def test_baseline_does_not_move_with_new_traffic(client):
    seed(40, 0)
    client.post("/baseline?force=true")
    before = client.get("/drift_score").json()["baseline_samples"]

    seed(25, 1)
    after = client.get("/drift_score").json()["baseline_samples"]

    assert before == after
    assert logger.get_baseline() == [0] * 40


def test_freezing_twice_without_force_keeps_the_first_reference(client):
    seed(40, 0)
    first = client.post("/baseline").json()
    seed(10, 2)
    second = client.post("/baseline").json()

    assert first["frozen"] == 40
    assert second["frozen"] == 0
    assert len(logger.get_baseline()) == 40


def test_refreezing_with_force_replaces_the_reference(client):
    seed(40, 0)
    client.post("/baseline?force=true")
    seed(40, 2)
    body = client.post("/baseline?force=true").json()

    assert body["frozen"] > 0
    assert logger.get_baseline().count(2) > 0


def test_baseline_source_distinguishes_auto_from_explicit(client):
    seed(40, 0)
    # No explicit freeze: the first /drift_score call freezes it automatically.
    body = client.get("/drift_score").json()
    assert body["baseline_source"] == "auto"

    client.post("/baseline?force=true")
    assert client.get("/drift_score").json()["baseline_source"] == "explicit"


def test_delete_clears_the_baseline(client):
    seed(40, 0)
    client.post("/baseline?force=true")
    body = client.delete("/baseline").json()

    assert body["source"] == "none"
    assert logger.get_baseline() == []


def test_predict_rejects_wrong_feature_count(client):
    resp = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert resp.status_code == 422


def test_stats_reports_both_distributions(client):
    seed(40, 0)
    client.post("/baseline?force=true")
    seed(10, 2)

    body = client.get("/stats").json()
    assert body["total_predictions"] == 50
    assert body["baseline_size"] == 40
    assert body["baseline_distribution"] == {"0": 40}
    assert body["recent_distribution"]["2"] == 10
