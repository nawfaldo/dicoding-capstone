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
    df = df.copy()

    drop_cols = [c for c in ["Target", "Failure_Type"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    if 'Process_temperature_K' in df.columns and 'Air_temperature_K' in df.columns:
        df['temp_diff'] = df['Process_temperature_K'] - df['Air_temperature_K']

    if 'Torque_Nm' in df.columns and 'Rotational_speed_rpm' in df.columns:
        df['load'] = df['Torque_Nm'] * df['Rotational_speed_rpm']

    if 'load' in df.columns and 'Process_temperature_K' in df.columns:
        df['stress'] = df['load'] / (df['Process_temperature_K'] + 1e-9)

    if 'Tool_wear_min' in df.columns and 'serial' in df.columns:
        df['wear_rate'] = df['Tool_wear_min'] / (df['serial'] + 1e-9)

    if 'Torque_Nm' in df.columns and 'Process_temperature_K' in df.columns:
        df['torque_temp_inter'] = df['Torque_Nm'] * df['Process_temperature_K']

    if 'Rotational_speed_rpm' in df.columns:
        df['high_rpm'] = (df['Rotational_speed_rpm'] > df['Rotational_speed_rpm'].median()).astype(int)

    if 'quality' in df.columns:
        df['quality_ord'] = df['quality'].map({'L': 0, 'M': 1, 'H': 2})

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    return df


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
