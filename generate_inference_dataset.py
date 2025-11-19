import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    raw_path = "datasets/dataset_raw.csv"

    df = pd.read_csv(raw_path).drop_duplicates().dropna(how="all")
    df.columns = df.columns.str.replace(r"[\[\]]", "", regex=True).str.replace(" ", "_")
    df = df.reset_index(drop=True)

    X_temp, X_test, y_temp, _ = train_test_split(
        df,
        df["Target"],
        test_size=0.15,
        stratify=df["Target"],
        random_state=42,
    )

    inf = X_test.drop(columns=["Target", "Failure_Type"])
    inf.to_csv("datasets/inference.csv", index=False)

    inf_fail = X_test[X_test["Target"] == 1].drop(columns=["Target", "Failure_Type"])
    inf_fail.to_csv("datasets/inference_failure.csv", index=False)

    print("Inference CSV berhasil dibuat.")
