# train.py
import argparse
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
TARGET_COL = "y"

def build_preprocessor(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    ohe = OneHotEncoder(handle_unknown="ignore")# sparse_output=False)
    scaler = StandardScaler()
    pre = ColumnTransformer([("num", scaler, num_cols), ("cat", ohe, cat_cols)], remainder="drop")
    return pre, num_cols, cat_cols

def build_models(pre):
    models = {}

    models["Logistic Regression"] = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs", random_state=RANDOM_STATE))
    ])

    models["Decision Tree"] = Pipeline([
        ("pre", pre),
        ("clf", DecisionTreeClassifier(max_depth=None, class_weight="balanced", random_state=RANDOM_STATE))
    ])

    models["kNN"] = Pipeline([
        ("pre", pre),
        ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance"))
    ])

    models["Naive Bayes"] = Pipeline([
        ("pre", pre),
        ("clf", GaussianNB())
    ])

    models["Random Forest"] = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    ])

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.07,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1,
                objective="binary:logistic", eval_metric="logloss", tree_method="hist"
            ))
        ])
    else:
        print("Note: xgboost not installed; skipping XGBoost.")

    return models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to bank-additional-full.csv")
    parser.add_argument("--sep", type=str, default=";", help="CSV separator (default: ';')")
    args = parser.parse_args()

    path = Path(args.data).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=args.sep, encoding="utf-8", engine="python")
    if TARGET_COL not in df.columns:
        raise KeyError(f"Missing target column '{TARGET_COL}'")

    y = df[TARGET_COL].astype(str).str.lower().map({"yes":1,"no":0}).astype(int)
    X = df.drop(columns=[TARGET_COL])

    pre, _, _ = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models(pre)

    rows = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        auc = roc_auc_score(y_test, proba) if proba is not None else np.nan
        mcc = matthews_corrcoef(y_test, preds)
        rows.append({
            "ML Model Name": name, "Accuracy": acc, "AUC": auc,
            "Precision": precision, "Recall": recall, "F1": f1, "MCC": mcc
        })

    out = pd.DataFrame(rows).sort_values(["F1","AUC","MCC"], ascending=[False, False, False]).reset_index(drop=True)
    out.to_csv("model_comparison.csv", index=False)
    print("Saved model_comparison.csv")

if __name__ == "__main__":
    main()