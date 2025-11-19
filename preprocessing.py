import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

raw_path = "datasets/dataset_raw.csv"


def load_and_clean():
    df = pd.read_csv(raw_path).drop_duplicates().dropna(how="all")

    df.columns = df.columns.str.replace(r"[\[\]]", "", regex=True).str.replace(" ", "_")

    df["quality"] = df["Product_ID"].str[0]
    df["serial"] = df["Product_ID"].str[1:].astype(int)

    df = df.drop(columns=[c for c in ["UDI", "Product_ID"] if c in df.columns])

    return df


def feature_engineering(df):
    X = df.drop(columns=["Target", "Failure_Type"])
    X = X.copy()

    if 'process_temperature_K' in X.columns and 'air_temperature_K' in X.columns:
        X['temp_diff'] = X['process_temperature_K'] - X['air_temperature_K']

    if 'torque_Nm' in X.columns and 'rotational_speed_rpm' in X.columns:
        X['load'] = X['torque_Nm'] * X['rotational_speed_rpm']

    if 'load' in X.columns and 'process_temperature_K' in X.columns:
        X['stress'] = X['load'] / (X['process_temperature_K'] + 1e-9)

    if 'tool_wear_min' in X.columns and 'serial' in X.columns:
        X['wear_rate'] = X['tool_wear_min'] / (X['serial'] + 1e-9)

    if 'torque_Nm' in X.columns and 'process_temperature_K' in X.columns:
        X['torque_temp_inter'] = X['torque_Nm'] * X['process_temperature_K']

    if 'rotational_speed_rpm' in X.columns:
        X['high_rpm'] = (X['rotational_speed_rpm'] > X['rotational_speed_rpm'].median()).astype(int)

    X['quality_ord'] = X['quality'].map({'L': 0, 'M': 1, 'H': 2})

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    for c in cat_cols:
        X[c] = X[c].fillna(X[c].mode().iloc[0])

    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())

    return X


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def prepare_binary_split(df, X):
    y = df["Target"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1764706, stratify=y_temp, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
