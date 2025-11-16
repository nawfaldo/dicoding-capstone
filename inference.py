import pandas as pd
import requests
import json

API_URL = "https://jawirgaming66-dicoding-capstone.hf.space/predict"
INPUT_CSV = "inference_failure.csv"

def main():
    df = pd.read_csv(INPUT_CSV)

    for index, row_data in df.iterrows():
        payload = row_data.to_dict()

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            api_result = response.json()

            print(f"Row {index} OK → Prediction: {api_result['prediction']}")
        except requests.exceptions.RequestException as e:
            print(f"Gagal mengirim data row {index}: {e}")

if __name__ == "__main__":
    main()
