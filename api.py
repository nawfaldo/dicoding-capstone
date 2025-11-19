from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from preprocessing import feature_engineering

binary_dict = joblib.load("models/model_binary.pkl")
binary_pipeline = binary_dict["pipeline"]
binary_threshold = binary_dict["threshold"]

multi_dict = joblib.load("models/model_multiclass.pkl")
multi_pipeline = multi_dict["pipeline"]
multi_label_encoder = multi_dict["label_encoder"]

app = FastAPI()

class MachineInput(BaseModel):
    UDI: int
    Product_ID: str
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: int
    Torque_Nm: float
    Tool_wear_min: int

def build_input_dataframe(data: MachineInput) -> pd.DataFrame:
    """Convert input JSON to DataFrame and build required base columns."""
    df = pd.DataFrame([data.dict()])

    df["quality"] = df["Product_ID"].str[0]
    df["serial"] = df["Product_ID"].str[1:].astype(int)

    return df

@app.post("/predict")
def predict_all(data: MachineInput):
    df = build_input_dataframe(data)

    X = feature_engineering(df)

    prob = binary_pipeline.predict_proba(X)[0][1]
    pred_binary = 1 if prob >= binary_threshold else 0

    if pred_binary == 1:
        probs_multi = multi_pipeline.predict_proba(X)[0]
        pred_idx = probs_multi.argmax()
        failure_type = multi_label_encoder.inverse_transform([pred_idx])[0]
    else:
        failure_type = None

    return {
        "binary": {
            "probability": float(prob),
            "prediction": "FAILURE" if pred_binary == 1 else "NO FAILURE"
        },
        "multiclass": {
            "failure_type": failure_type
        }
    }
