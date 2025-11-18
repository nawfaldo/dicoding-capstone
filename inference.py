import pandas as pd
import requests
import json

API_URL = "http://127.0.0.1:8000/predict"
INPUT_CSV = "datasets/inference_failure.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    for index, row_data in df.iterrows():
        payload = row_data.to_dict()

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            api_result = response.json()
            
            prediction = api_result["binary"]["prediction"]
            failure_type = api_result["multiclass"]["failure_type"]

            print(f"\nRow {index} OK")
            print(f"  Prediction     : {prediction}")
            print(f"  Failure Type   : {failure_type}")
        except requests.exceptions.RequestException as e:
            print(f"Gagal mengirim data row {index}: {e}")

if __name__ == "__main__":
    main()
