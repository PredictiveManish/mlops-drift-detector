import requests
import random
import time
import json

# API endpoint
url = "http://localhost:8000/predict"

# Normal Iris flower measurements (Iris-setosa)
normal_features = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [4.9, 3.0, 1.4, 0.2],  # Setosa
    [4.7, 3.2, 1.3, 0.2],  # Setosa
    [4.6, 3.1, 1.5, 0.2],  # Setosa
    [5.0, 3.6, 1.4, 0.2],  # Setosa
    [5.4, 3.9, 1.7, 0.4],  # Setosa
    [4.6, 3.4, 1.4, 0.3],  # Setosa
    [5.0, 3.4, 1.5, 0.2],  # Setosa
    [4.4, 2.9, 1.4, 0.2],  # Setosa
    [4.9, 3.1, 1.5, 0.1]   # Setosa
]

# Anomalous measurements (Versicolor/Virginica) to simulate drift
drift_features = [
    [6.5, 3.0, 5.2, 2.0],   # Virginica
    [6.4, 3.2, 4.5, 1.5],   # Versicolor
    [6.3, 3.3, 6.0, 2.5],   # Virginica
    [5.8, 2.7, 5.1, 1.9],   # Virginica
    [7.1, 3.0, 5.9, 2.1],   # Virginica
    [6.3, 2.9, 5.6, 1.8],   # Virginica
    [6.5, 3.0, 5.8, 2.2],   # Virginica
    [7.6, 3.0, 6.6, 2.1],   # Virginica
    [4.9, 2.5, 4.5, 1.7],   # Versicolor
    [5.6, 2.7, 4.2, 1.3]    # Versicolor
]

print("\nGenerating predictions to test drift detection...\n")

# Send normal predictions
print("Sending 20 normal predictions (Setosa)...")
for i in range(20):
    features = random.choice(normal_features)
    payload = {"features": features}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"  ✓ Prediction {i+1}/20: {result['prediction']} (Setosa)")
        else:
            print(f"  ✗ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    time.sleep(0.2)

# Check drift score after normal predictions
print("\n📈 Current drift score:")
response = requests.get("http://localhost:8000/drift_score")
print(json.dumps(response.json(), indent=2))

# Send drift-inducing predictions
print("\n🌊 Sending 20 drift-inducing predictions (Virginica/Versicolor)...")
for i in range(20):
    features = random.choice(drift_features)
    payload = {"features": features}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            species = "Virginica" if prediction == 2 else "Versicolor" if prediction == 1 else "Setosa"
            print(f"  ✓ Prediction {i+1}/20: {prediction} ({species})")
        else:
            print(f"  ✗ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  ✗ Exception: {e}")
    
    time.sleep(0.2)

# Final drift score
print("\nFinal drift score (should show drift):")
response = requests.get("http://localhost:8000/drift_score")
print(json.dumps(response.json(), indent=2))

# Final statistics
print("\nFinal statistics:")
response = requests.get("http://localhost:8000/stats")
print(json.dumps(response.json(), indent=2))

print("\nDone!")