# app.py
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef
)

# --- ensure project root on sys.path ---
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- import model builders ---
from models.logistic_regression import build as build_logreg
from models.decision_tree     import build as build_tree
from models.knn               import build as build_knn
from models.naive_bayes       import build as build_nb
from models.random_forest     import build as build_rf
from models.xgboost_model     import build as build_xgb

st.set_page_config(page_title="ML Assignment 2 — Bank Marketing (additional)", page_icon="🏦", layout="wide")
st.title("ML Assignment 2 — UCI Bank Marketing (additional)")

DATA_PATH = ROOT / "data" / "bank-additional-full.csv"  # semicolon-separated CSV
st.caption(f"Data path: `{DATA_PATH}`")

# ---------- Load & prepare dataset ----------
@st.cache_data(show_spinner=True)
def load_dataset(path: Path):
    # 'additional' CSV is semicolon-delimited and quoted
    df = pd.read_csv(path, sep=";", quotechar='"')

    # Target mapping: 'yes'/'no' -> 1/0
    df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)

    y = df["y"].copy()
    X = df.drop(columns=["y"]).copy()

    # Identify types
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # ColumnTransformer with OneHot for categoricals (dense)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    meta = {
        "cat": cat_cols, "num": num_cols, "all": cat_cols + num_cols,
        "rows": len(X), "cols": len(cat_cols) + len(num_cols)
    }
    return X, y, preprocessor, meta

try:
    X, y, preprocessor, meta = load_dataset(DATA_PATH)
    st.success(f"Loaded dataset: {meta['rows']} rows, {meta['cols']} features.")
    st.write("Categorical:", meta["cat"])
    st.write("Numeric:", meta["num"])
except Exception as e:
    st.error(f"Failed to load dataset at {DATA_PATH}: {e}")
    st.stop()

# ---------- Build model pipelines ----------

@st.cache_resource  # cache global resources like model pipelines
def get_models(cat_cols: Tuple[str, ...], num_cols: Tuple[str, ...]):
    # Build the preprocessor here from hashable inputs
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)),
            ("num", "passthrough", list(num_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return {
        "Logistic Regression": Pipeline([("prep", preprocessor), ("clf", build_logreg())]),
        "Decision Tree":       Pipeline([("prep", preprocessor), ("clf", build_tree())]),
        "kNN":                 Pipeline([("prep", preprocessor), ("clf", build_knn())]),
        "Naive Bayes":         Pipeline([("prep", preprocessor), ("clf", build_nb())]),
        "Random Forest":       Pipeline([("prep", preprocessor), ("clf", build_rf())]),
        "XGBoost":             Pipeline([("prep", preprocessor), ("clf", build_xgb())]),
    }


models = get_models(tuple(meta["cat"]), tuple(meta["num"]))

def compute_metrics(y_true, y_pred, y_score=None):
    out = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "MCC":       matthews_corrcoef(y_true, y_pred),  # required by assignment
        "AUC":       np.nan,
    }
    if y_score is not None:
        try:
            out["AUC"] = roc_auc_score(y_true, y_score)
        except Exception:
            pass
    return out

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    selection = st.selectbox("Select a model (or All Models)",
                             ["All Models (comparison)"] + list(models.keys()))
    st.markdown("---")
    st.caption("Optional: Upload a CSV for test-only evaluation "
               "(must have the SAME columns as training features).")
    up = st.file_uploader("Upload CSV", type=["csv"])

# ---------- Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# ---------- Optional test-set override via upload ----------
if up is not None:
    try:
        raw = up.getvalue().decode(errors="ignore")
        # Auto-detect semicolon; fall back to comma
        sep = ";" if ";" in raw.splitlines()[0] else ","
        df_up = pd.read_csv(up, sep=sep)
        missing = set(meta["all"]) - set(df_up.columns)
        if missing:
            st.error(f"Uploaded CSV is missing columns: {sorted(list(missing))}")
        else:
            X_test = df_up[meta["all"]].copy()
            st.success(f"Using uploaded CSV as test set. Shape: {X_test.shape}")
            st.write(X_test.head())
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")

# ---------- Fit & evaluate all models ----------
def fit_and_score(models_dict, X_tr, y_tr, X_te, y_te):
    results, cms, reports = {}, {}, {}
    for name, pipe in models_dict.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        y_score = None
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_te)[:, 1]
        elif hasattr(pipe, "decision_function"):
            y_score = pipe.decision_function(X_te)

        metrics = compute_metrics(y_te, y_pred, y_score)
        cm = confusion_matrix(y_te, y_pred)
        cr = classification_report(y_te, y_pred, target_names=["no", "yes"])
        results[name], cms[name], reports[name] = metrics, cm, cr
    return results, cms, reports

all_results, all_cms, all_reports = fit_and_score(models, X_train, y_train, X_test, y_test)

def as_table(res_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(res_dict).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    return df.sort_values(by="F1", ascending=False)

# ---------- UI render ----------
if selection == "All Models (comparison)":
    st.subheader("Model Comparison")
    comp = as_table(all_results)
    st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)

    st.markdown("### Observations (for README)")
    st.markdown(
        "- Compare **F1**/**AUC** across models.\n"
        "- Tree ensembles (**Random Forest**, **XGBoost**) often perform well on tabular data.\n"
        "- Check **MCC** for balanced quality under any class imbalance."
    )
else:
    st.subheader(f"Model: {selection}")
    m = all_results[selection]
    st.write(pd.DataFrame([m], index=[selection]).style.format("{:.4f}"))

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(all_cms[selection], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("#### Classification Report")
    st.text(all_reports[selection])

st.markdown(
    """
**Dataset:** Bank Marketing (additional) — `bank-additional-full.csv` (semicolon-delimited)  
**Goal:** Predict term-deposit subscription (`y`: yes/no).  
*Source: UCI Machine Learning Repository.*  
"""
)
# Dataset & file details referenced from UCI repository. [2](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
