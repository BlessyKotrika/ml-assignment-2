# app.py
import os
import sys
from pathlib import Path
from typing import Tuple, Dict

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

# -------- Project paths --------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------- Model builders (your files in models/) --------
from models.logistic_regression import build as build_logreg
from models.decision_tree     import build as build_tree
from models.knn               import build as build_knn
from models.naive_bayes       import build as build_nb
from models.random_forest     import build as build_rf
from models.xgboost_model     import build as build_xgb

st.set_page_config(page_title="ML Assignment 2 — Bank Marketing (additional)", page_icon="🏦", layout="wide")
st.title("ML Assignment 2 — UCI Bank Marketing (additional)")

DATA_PATH = ROOT / "data" / "bank-additional-full.csv"
st.caption(f"Data path: `{DATA_PATH}`")

# -------- Helper: compatible OneHotEncoder dense factory (works across sklearn versions) --------
def make_ohe_dense() -> OneHotEncoder:
    import inspect
    params = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        params["sparse_output"] = False  # sklearn >= 1.2
    else:
        params["sparse"] = False         # sklearn < 1.2 fallback
    return OneHotEncoder(**params)

# -------- Load dataset (cached) --------
@st.cache_data(show_spinner=True)
def load_dataset(path: Path):
    df = pd.read_csv(path, sep=";", quotechar='"')

    # Map target 'y' from "yes"/"no" -> 1/0 (int)
    df["y"] = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(int)

    y = df["y"].copy()
    X = df.drop(columns=["y"]).copy()

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # We only describe/precompute schema here; preprocessor is built later to avoid caching issues
    meta = {
        "cat": cat_cols,
        "num": num_cols,
        "all": cat_cols + num_cols,
        "rows": len(X),
        "cols": len(cat_cols) + len(num_cols)
    }
    return X, y, meta

try:
    X, y, meta = load_dataset(DATA_PATH)
    st.success(f"Loaded dataset: {meta['rows']} rows, {meta['cols']} features.")
    with st.expander("Columns detected", expanded=False):
        st.write("**Categorical**:", meta["cat"])
        st.write("**Numeric**:", meta["num"])
except Exception as e:
    st.error(f"Failed to load dataset at {DATA_PATH}: {e}")
    st.stop()

# -------- Build model pipelines (resource-cached with hashable args) --------
@st.cache_resource
def get_models(cat_cols: Tuple[str, ...], num_cols: Tuple[str, ...]) -> Dict[str, Pipeline]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_ohe_dense(), list(cat_cols)),
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

# -------- Metrics --------
def compute_metrics(y_true, y_pred, y_score=None):
    # Guard against length mismatch (e.g., if only X was replaced earlier)
    if len(y_true) != len(y_pred):
        st.error(f"Length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred)}. "
                 "This usually means you replaced features but not labels for evaluation.")
        st.stop()
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

# -------- Sidebar controls --------
with st.sidebar:
    st.header("Controls")

    # Ensure default test size is consistent
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    # Default dropdown to 'All Models (comparison)' on a fresh session
    if "model_selection" not in st.session_state:
        st.session_state["model_selection"] = "All Models (comparison)"

    selection = st.selectbox(
        "Select a model (or All Models)",
        options=["All Models (comparison)"] + list(models.keys()),
        index=0,  # default index
        key="model_selection"
    )

    st.markdown("---")
    st.caption(
        "Optional: Upload a CSV.\n\n"
        "- **With `y`** → evaluates metrics on your file\n"
        "- **Without `y`** → inference only on uploaded rows"
    )
    up = st.file_uploader("Upload CSV", type=["csv"])

# -------- Base split --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# Evaluation set defaults to internal split; can be overridden by labeled upload
eval_X, eval_y = X_test, y_test
inference_df = None  # will be used when upload has no 'y'

# -------- Upload handler (supports labeled and features-only) --------
if up is not None:
    try:
        raw = up.getvalue().decode(errors="ignore")
        sep = ";" if ";" in raw.splitlines()[0] else ","
        df_up = pd.read_csv(up, sep=sep)

        if "y" in df_up.columns:
            # EVALUATION MODE: uploaded file has labels
            if df_up["y"].dtype == object:
                y_up = (df_up["y"].astype(str).str.strip().str.lower() == "yes").astype(int)
            else:
                y_up = df_up["y"].astype(int)
            X_up = df_up.drop(columns=["y"])

            missing = set(meta["all"]) - set(X_up.columns)
            extra   = set(X_up.columns) - set(meta["all"])
            if missing:
                st.error(f"Uploaded CSV (with y) is missing columns: {sorted(list(missing))}")
            else:
                eval_X = X_up[meta["all"]].copy()
                eval_y = y_up.reset_index(drop=True)  # <-- ensures same length as predictions
                st.success(f"Using uploaded CSV (with labels) as the test set — shape: {eval_X.shape}")
                if extra:
                    st.info(f"Ignored extra columns: {sorted(list(extra))}")

        else:
            # INFERENCE-ONLY MODE: no labels present
            missing = set(meta["all"]) - set(df_up.columns)
            extra   = set(df_up.columns) - set(meta["all"])
            if missing:
                st.error(f"Uploaded CSV (no y) is missing columns: {sorted(list(missing))}")
            else:
                inference_df = df_up[meta["all"]].copy()
                st.info(
                    "Uploaded CSV has **no `y` column** → running **inference only** on uploaded rows.\n"
                    "Metrics below still use the internal train/test split."
                )
                if extra:
                    st.info(f"Ignored extra columns: {sorted(list(extra))}")
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")

# -------- Fit & evaluate all models on eval_X/eval_y --------
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

        met = compute_metrics(y_te, y_pred, y_score)
        cm = confusion_matrix(y_te, y_pred)
        cr = classification_report(y_te, y_pred, target_names=["no", "yes"], zero_division=0)

        results[name], cms[name], reports[name] = met, cm, cr
    return results, cms, reports

# -------- Fit & evaluate all models on eval_X/eval_y --------
try:
    with st.spinner("Training and evaluating models..."):
        all_results, all_cms, all_reports = fit_and_score(models, X_train, y_train, eval_X, eval_y)
except Exception as ex:
    st.error("❌ An error occurred while training/evaluating models.")
    st.exception(ex)
    st.stop()

# Safety: if something returned empty, show a helpful message
if not all_results:
    st.warning("No results produced. Please try rerunning the app (↻) or check the dataset and logs.")
def as_table(res_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame(res_dict).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    return df.sort_values(by="F1", ascending=False)

# -------- UI render --------
# -------- UI render --------
if selection == "All Models (comparison)":
    st.subheader("📊 Model Comparison")
    comp = as_table(all_results)
    st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)

    st.markdown("### Observations (add to README)")
    st.markdown(
        "- Compare **F1** and **AUC** across models.\n"
        "- Tree ensembles (**Random Forest**, **XGBoost**) often perform well on mixed tabular data.\n"
        "- **MCC** provides a balanced view under class imbalance."
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


# -------- Predictions on uploaded CSV (inference-only) --------
if inference_df is not None:
    st.markdown("---")
    st.subheader("Predictions on uploaded CSV (inference-only)")

    chosen = selection if selection != "All Models (comparison)" else "Logistic Regression"
    infer_models = get_models(tuple(meta["cat"]), tuple(meta["num"]))
    infer_model = infer_models[chosen]
    infer_model.fit(X_train, y_train)

    preds = infer_model.predict(inference_df)
    probs = infer_model.predict_proba(inference_df)[:, 1] if hasattr(infer_model, "predict_proba") else None

    out = pd.DataFrame({"pred": preds})
    if probs is not None:
        out["prob_yes"] = probs

    st.dataframe(out.head(20), use_container_width=True)
    st.download_button(
        label="⬇️ Download predictions",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="uploaded_predictions.csv",
        mime="text/csv"
    )

# -------- Dataset footer --------
st.markdown(
    """
**Dataset:** Bank Marketing (additional) — `bank-additional-full.csv` (semicolon‑delimited)  
**Target (`y`):** term-deposit subscription (yes/no → 1/0)
"""
)