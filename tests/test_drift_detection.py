"""Unit tests for the drift statistics.

This file used to be a script that called a deployed service at import time,
which meant `pytest tests/` failed whenever that service was down (and the
URL it used was malformed, so it always failed). The end-to-end demo lives in
generate_predictions.py; these tests exercise the maths directly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app.drift_detector import compute_drift_score, compute_psi, should_alert


def test_identical_distributions_score_zero():
    assert compute_psi([0] * 50, [0] * 50) == pytest.approx(0.0, abs=1e-9)


def test_completely_different_distribution_saturates_at_one():
    assert compute_psi([0] * 50, [2] * 50) == pytest.approx(1.0)


def test_mild_shift_lands_between_the_extremes():
    """A small shift must register without saturating the score."""
    score = compute_psi([0] * 50 + [1] * 50, [0] * 40 + [1] * 60)
    assert 0.0 < score < 1.0


def test_significant_shift_saturates_at_one():
    """PSI above 0.2 is the conventional "significant" band, and the score is
    normalised so that band maps to 1.0. Documented here so the convention is
    not mistaken for a bug."""
    baseline = [0] * 40 + [1] * 40 + [2] * 20
    shifted = [0] * 20 + [1] * 40 + [2] * 40
    assert compute_psi(baseline, shifted) == 1.0


def test_score_is_scale_free():
    """Doubling both samples must not change the score."""
    baseline = [0] * 20 + [1] * 20 + [2] * 10
    recent = [0] * 10 + [1] * 20 + [2] * 20
    small = compute_psi(baseline, recent)
    large = compute_psi(baseline * 2, recent * 2)
    assert small == pytest.approx(large, abs=1e-9)


def test_alert_threshold():
    assert should_alert(0.2, 0.15) is True
    assert should_alert(0.1, 0.15) is False
    assert should_alert(0.15, 0.15) is False


def test_empty_input_scores_zero_rather_than_crashing():
    assert compute_drift_score([], [1, 2, 3]) == 0.0
    assert compute_drift_score([1, 2, 3], []) == 0.0
