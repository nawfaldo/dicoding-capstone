import joblib
import numpy as np
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from preprocessing import load_and_clean, feature_engineering, build_preprocessor

def main():
    df = load_and_clean()
    X = feature_engineering(df)

    df_fail = df[df["Target"] == 1].reset_index(drop=True)
    X_fail = X.loc[df["Target"] == 1].copy()
    y_fail = df_fail["Failure_Type"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_fail)
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_fail, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    preprocessor = build_preprocessor(X_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_estimators=400,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

    pipeline = ImbPipeline([
        ("pre", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    probs = pipeline.predict_proba(X_test)
    pred = np.argmax(probs, axis=1)

    print("\n=== Multiclass Classification Report ===")
    print(classification_report(y_test, pred, target_names=le.classes_))

    joblib.dump({
        "pipeline": pipeline,
        "label_encoder": le
    }, "models/model_multiclass.pkl")
