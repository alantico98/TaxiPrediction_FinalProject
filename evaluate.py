import json
import requests

API_URL = "http://localhost:8000/predict"  # replace with your EC2 URL
TEST_DATA_FILE = "test.json"


def main():
    # Load test data
    with open(TEST_DATA_FILE, "r") as f:
        test_data = json.load(f)

    y_pred = []

    for entry in test_data:
        input_data = {
            "PULocationID": entry["PULocationID"],
            "DOLocationID": entry["DOLocationID"],
            "timestamp": entry["timestamp"]
        }

        try:
            response = requests.post(API_URL, json=input_data)
            response.raise_for_status()
            prediction = response.json()["prediction"]
            y_pred.append(prediction.lower())
        except Exception as e:
            print(f"Error making prediction for {entry['text']}: {e}")
            continue


if __name__ == "__main__":
    main()
