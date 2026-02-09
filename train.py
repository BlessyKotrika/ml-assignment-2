# train.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
)

from models.logistic_regression import build as build_logreg
from models.decision_tree     import build as build_tree
from models.knn               import build as build_knn
from models.naive_bayes       import build as build_nb
from models.random_forest     import build as build_rf
from models.xgboost_model     import build as build_xgb

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-additional-full.csv"

def metrics(y_true, y_pred, y_score=None):
    out = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),
        "AUC":       np.nan,
    }
    if y_score is not None:
        try:
            out["AUC"] = roc_auc_score(y_true, y_score)
        except Exception:
            pass
    return out

def main():
    df = pd.read_csv(DATA_PATH, sep=";", quotechar='"')
    df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)
    y = df["y"].copy()
    X = df.drop(columns=["y"]).copy()

    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(exclude=["object"]).columns.tolist()

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
         ("num", "passthrough", num)],
        remainder="drop"
    )

    models = {
        "Logistic Regression": Pipeline([("prep", pre), ("clf", build_logreg())]),
        "Decision Tree":       Pipeline([("prep", pre), ("clf", build_tree())]),
        "kNN":                 Pipeline([("prep", pre), ("clf", build_knn())]),
        "Naive Bayes":         Pipeline([("prep", pre), ("clf", build_nb())]),
        "Random Forest":       Pipeline([("prep", pre), ("clf", build_rf())]),
        "XGBoost":             Pipeline([("prep", pre), ("clf", build_xgb())]),
    }

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    res = {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_score = pipe.predict_proba(X_te)[:, 1] if hasattr(pipe, "predict_proba") else None
        res[name] = metrics(y_te, y_pred, y_score)

    df_res = pd.DataFrame(res).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    df_res = df_res.sort_values(by="F1", ascending=False).round(4)
    print(df_res)
    df_res.to_csv(ROOT / "model_comparison.csv")

if __name__ == "__main__":
    main()