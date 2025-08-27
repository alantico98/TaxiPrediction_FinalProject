import json
import requests
from sklearn.metrics import accuracy_score

API_URL = "http://localhost:8000/predict"  # replace with your EC2 URL
TEST_DATA_FILE = "test.json"


def main():
    # Load test data
    with open(TEST_DATA_FILE, "r") as f:
        test_data = json.load(f)

    y_true = []
    y_pred = []

    for entry in test_data:
        input_data = {
            "text": entry["text"],
            "true_label": entry["true_label"]
        }

        try:
            response = requests.post(API_URL, json=input_data)
            response.raise_for_status()
            prediction = response.json()["sentiment"]
            y_true.append(entry["true_label"].lower())
            y_pred.append(prediction.lower())
        except Exception as e:
            print(f"Error making prediction for {entry['text']}: {e}")
            continue

    acc = accuracy_score(y_true, y_pred)
    print(f"Model Accuracy: {acc:.2f}")


if __name__ == "__main__":
    main()
