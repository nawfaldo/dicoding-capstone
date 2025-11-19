import joblib
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve, classification_report

from preprocessing import load_and_clean, feature_engineering, build_preprocessor, prepare_binary_split

def main():
    df = load_and_clean()
    X = feature_engineering(df)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_binary_split(df, X)

    preprocessor = build_preprocessor(X_train)


    scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
            "use_label_encoder": False,
            "verbosity": 0,
            "scale_pos_weight": scale_pos,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)

        pipeline = ImbPipeline([
            ("pre", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_val)[:, 1]

        prec, rec, th = precision_recall_curve(y_val, probs)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-9)

        return np.nanmax(f1s)


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)

    best_params = study.best_params
    best_params.update({
        "random_state": 42,
        "use_label_encoder": False,
        "verbosity": 0,
        "scale_pos_weight": scale_pos,
        "eval_metric": "logloss",
    })

    final_model = XGBClassifier(**best_params)

    pipeline = ImbPipeline([
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", final_model),
    ])

    X_comb = pd.concat([X_train, X_val], axis=0)
    y_comb = pd.concat([y_train, y_val], axis=0)

    pipeline.fit(X_comb, y_comb)

    probs_val = pipeline.predict_proba(X_val)[:, 1]
    prec, rec, th = precision_recall_curve(y_val, probs_val)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_threshold = th[np.nanargmax(f1s)]

    print("Best Threshold: ", best_threshold)

    probs_test = pipeline.predict_proba(X_test)[:, 1]
    pred_test = (probs_test >= best_threshold).astype(int)

    print("\n=== Test Classification Report ===")
    print(classification_report(y_test, pred_test))

    joblib.dump({
        "pipeline": pipeline,
        "threshold": best_threshold,
        "optuna_study": study
    }, "models/model_binary.pkl")
