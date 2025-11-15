from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model_dict = joblib.load("model.pkl")
pipeline = model_dict["pipeline"]
threshold = model_dict["threshold"]

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

@app.post("/predict")
def predict_api(data: MachineInput):
    df = pd.DataFrame([data.dict()])

    df["quality"] = df["Product_ID"].str[0]
    df["serial"] = df["Product_ID"].str[1:].astype(int)

    df["temp_diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["load"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]
    df["stress"] = df["load"] / (df["Process_temperature_K"] + 1e-9)
    df["wear_rate"] = df["Tool_wear_min"] / (df["serial"] + 1e-9)
    df["torque_temp_inter"] = df["Torque_Nm"] * df["Process_temperature_K"]
    df["high_rpm"] = (df["Rotational_speed_rpm"] > df["Rotational_speed_rpm"].median()).astype(int)

    df["quality_ord"] = df["quality"].map({'L': 0, 'M': 1, 'H': 2})

    prob = pipeline.predict_proba(df)[0][1]

    pred = 1 if prob >= threshold else 0

    return {
        "probability": float(prob),
        "threshold": float(threshold),
        "prediction": "FAILURE" if pred == 1 else "NO FAILURE"
    }
