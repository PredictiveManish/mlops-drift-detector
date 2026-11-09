"""Drive the API to show drift detection working.

Sends a batch of normal traffic, freezes that as the reference distribution,
then sends a batch from a different distribution and checks that the drift
score actually moves. Exits non-zero if it does not.

Usage:
    BASE_URL=http://127.0.0.1:8000 python generate_predictions.py
"""

import json
import os
import random
import sys
import time

import requests

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
BATCH = int(os.getenv("BATCH", "40"))

# Iris-setosa measurements: what the service normally sees.
normal_features = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],
]

# Versicolor / virginica: the distribution shift we want to catch.
drift_features = [
    [6.5, 3.0, 5.2, 2.0],
    [6.4, 3.2, 4.5, 1.5],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
    [6.5, 3.0, 5.8, 2.2],
    [7.6, 3.0, 6.6, 2.1],
    [4.9, 2.5, 4.5, 1.7],
    [5.6, 2.7, 4.2, 1.3],
]


def send(features, label):
    resp = requests.post(f"{BASE_URL}/predict", json={"features": features}, timeout=10)
    if resp.status_code != 200:
        print(f"  x {resp.status_code}: {resp.text[:120]}")
        return None
    return resp.json()["prediction"]


def drift():
    resp = requests.get(f"{BASE_URL}/drift_score", timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"\nTarget: {BASE_URL}\n")

    print(f"Sending {BATCH} normal predictions (setosa)...")
    for i in range(BATCH):
        send(random.choice(normal_features), "normal")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{BATCH}")
    time.sleep(0.5)

    print("\nFreezing this traffic as the reference distribution...")
    frozen = requests.post(f"{BASE_URL}/baseline?force=true", timeout=10).json()
    print(json.dumps(frozen, indent=2))

    before = drift()
    print("\nDrift score against the reference (expected: low):")
    print(json.dumps(before, indent=2))

    print(f"\nSending {BATCH} drifted predictions (versicolor/virginica)...")
    for i in range(BATCH):
        send(random.choice(drift_features), "drift")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{BATCH}")
    time.sleep(0.5)

    after = drift()
    print("\nDrift score after the shift (expected: high):")
    print(json.dumps(after, indent=2))

    if after.get("drift_score") is None:
        print("\nFAIL: not enough data to score.")
        return 1
    if after["drift_score"] <= before.get("drift_score", 0):
        print("\nFAIL: the score did not rise when the distribution changed.")
        return 1
    if not after["alert"]:
        print("\nFAIL: score rose but stayed under the alert threshold.")
        return 1
    print(f"\nPASS: {before.get('drift_score')} -> {after['drift_score']} (threshold "
          f"{after['threshold']}), alert raised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
