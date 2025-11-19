from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

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


def preprocess(df: pd.DataFrame):
    df["quality"] = df["Product_ID"].str[0]
    df["serial"] = df["Product_ID"].str[1:].astype(int)

    df["temp_diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]
    df["load"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]
    df["stress"] = df["load"] / (df["Process_temperature_K"] + 1e-9)
    df["wear_rate"] = df["Tool_wear_min"] / (df["serial"] + 1e-9)
    df["torque_temp_inter"] = df["Torque_Nm"] * df["Process_temperature_K"]

    df["high_rpm"] = (df["Rotational_speed_rpm"] > df["Rotational_speed_rpm"].median()).astype(int)

    df["quality_ord"] = df["quality"].map({'L': 0, 'M': 1, 'H': 2})

    return df

@app.post("/predict")
def predict_all(data: MachineInput):
    df = pd.DataFrame([data.dict()])
    df = preprocess(df)

    prob = binary_pipeline.predict_proba(df)[0][1]
    pred_binary = 1 if prob >= binary_threshold else 0

    if pred_binary == 1:
        probs_multi = multi_pipeline.predict_proba(df)[0]
        pred_multi_idx = probs_multi.argmax()
        failure_label = multi_label_encoder.inverse_transform([pred_multi_idx])[0]
    else:
        failure_label = None

    return {
        "binary": {
            "prediction": "FAILURE" if pred_binary == 1 else "NO FAILURE"
        },
        "multiclass": {
            "failure_type": failure_label,
        }
    }
