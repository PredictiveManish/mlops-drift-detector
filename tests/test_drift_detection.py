# test_drift_detection.py
import requests
import time
import json

base_url = "http://localhost:8000"

print("🧪 Testing Drift Detection\n")

# Check initial state
print("Initial state:")
response = requests.get(f"{base_url}/stats")
print(json.dumps(response.json(), indent=2))

# Send 20 Setosa (class 0)
print("\n🌱 Sending 20 Setosa predictions...")
for i in range(20):
    response = requests.post(
        f"{base_url}/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    if i % 5 == 0:
        print(f"  Sent {i+1}/20")
    time.sleep(0.1)

# Check drift after Setosa only
print("\n📊 Drift after Setosa only:")
response = requests.get(f"{base_url}/drift_score")
print(json.dumps(response.json(), indent=2))

# Send 20 Virginica (class 2) 
print("\n🌋 Sending 20 Virginica predictions...")
for i in range(20):
    response = requests.post(
        f"{base_url}/predict",
        json={"features": [6.5, 3.0, 5.2, 2.0]}
    )
    if i % 5 == 0:
        print(f"  Sent {i+1}/20")
    time.sleep(0.1)

# Final check
print("\n🎯 Final drift score (should show drift):")
response = requests.get(f"{base_url}/drift_score")
print(json.dumps(response.json(), indent=2))

print("\n📈 Final statistics:")
response = requests.get(f"{base_url}/stats")
print(json.dumps(response.json(), indent=2))