# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef
)

# Model builders
from models.logistic_regression import build as build_logreg
from models.decision_tree import build as build_tree
from models.knn import build as build_knn
from models.naive_bayes import build as build_nb
from models.random_forest import build as build_rf
from models.xgboost_model import build as build_xgb

st.set_page_config(page_title="ML Assignment 2 — Bank Marketing (additional)", page_icon="🏦", layout="wide")
st.title("ML Assignment 2 — UCI Bank Marketing (additional)")

DATA_PATH = "data/bank-additional-full.csv"  # semicolon-separated CSV

@st.cache_data
def load_raw_csv(path: str) -> pd.DataFrame:
    # UCI 'additional' datasets are semicolon-delimited and quoted
    return pd.read_csv(path, sep=";", quotechar='"')

@st.cache_data
def load_dataset():
    df = load_raw_csv(DATA_PATH)

    # Target: 'y' ('yes'/'no') -> 1/0
    df["y"] = (df["y"].str.strip().str.lower() == "yes").astype(int)

    # Split features / target
    y = df["y"].copy()
    X = df.drop(columns=["y"]).copy()

    # Identify categorical vs numeric
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # ColumnTransformer: One-Hot for categoricals, passthrough numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    feature_info = {
        "categorical": cat_cols,
        "numeric": num_cols,
        "all": cat_cols + num_cols
    }
    return X, y, preprocessor, feature_info

X, y, preprocessor, feature_info = load_dataset()

st.markdown(
    f"""
**Dataset:** UCI Bank Marketing (additional) — `bank-additional-full.csv`  
**Instances:** {len(X)} | **Features:** {len(feature_info['all'])} | **Target:** `y` (yes/no → 1/0)

*Source: UCI ML Repository.*  
"""
)
# (UCI Bank Marketing additional version description) [1](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

@st.cache_data
def get_models(preprocessor):
    """Wrap each classifier in a unified Pipeline with the same preprocessor."""
    return {
        "Logistic Regression": Pipeline([("prep", preprocessor), ("clf", build_logreg())]),
        "Decision Tree":       Pipeline([("prep", preprocessor), ("clf", build_tree())]),
        "kNN":                 Pipeline([("prep", preprocessor), ("clf", build_knn())]),
        "Naive Bayes":         Pipeline([("prep", preprocessor), ("clf", build_nb())]),
        "Random Forest":       Pipeline([("prep", preprocessor), ("clf", build_rf())]),
        "XGBoost":             Pipeline([("prep", preprocessor), ("clf", build_xgb())]),
    }

models = get_models(preprocessor)

def compute_metrics(y_true, y_pred, y_score=None):
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

with st.sidebar:
    st.header("Controls")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    selected = st.selectbox(
        "Select a model (or All Models)",
        ["All Models (comparison)"] + list(models.keys())
    )
    st.markdown("---")
    st.caption("Optional: Upload a CSV with the same columns as training data for test-only evaluation.")
    upload = st.file_uploader("Upload CSV", type=["csv"])

# Split base train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# If user uploaded a CSV, use it as test set (schema check)
if upload is not None:
    try:
        df_up = pd.read_csv(upload, sep=";", quotechar='"') if ";" in upload.getvalue().decode(errors="ignore")[:200] else pd.read_csv(upload)
        missing = set(feature_info["all"]) - set(df_up.columns)
        if missing:
            st.error(f"Uploaded CSV is missing columns: {sorted(list(missing))}")
        else:
            X_test = df_up[feature_info["all"]].copy()
            st.success("Using uploaded CSV as the test set.")
            st.write(X_test.head())
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")

def fit_eval(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # For AUC scores, use predict_proba or decision_function
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    met = compute_metrics(y_test, y_pred, y_score)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["no", "yes"])
    return met, cm, cr

# Evaluate all
all_results = {}
all_cms = {}
all_crs = {}
for name, pipe in models.items():
    m, cm, cr = fit_eval(pipe)
    all_results[name] = m
    all_cms[name] = cm
    all_crs[name] = cr

def as_table(res_dict):
    df = pd.DataFrame(res_dict).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    return df.sort_values(by="F1", ascending=False)

if selected == "All Models (comparison)":
    st.subheader("Model Comparison")
    comp = as_table(all_results)
    st.dataframe(comp.style.format("{:.4f}"), use_container_width=True)

    st.markdown("### Observations (draft)")
    st.markdown(
        "- Compare **F1** and **AUC** across models.\n"
        "- Tree ensembles (RF, XGBoost) often perform well on mixed tabular data.\n"
        "- Check MCC for balanced quality under any class imbalance."
    )
else:
    st.subheader(f"Model: {selected}")
    m = all_results[selected]
    st.write(pd.DataFrame([m], index=[selected]).style.format("{:.4f}"))

    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(all_cms[selected], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("#### Classification Report")
    st.text(all_crs[selected])

st.markdown(
    """
**About the dataset (additional version):** Contains bank client data, campaign outcomes, and economic indicators  
(e.g., `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`).  
*Source: UCI Machine Learning Repository.*  
"""
)
# (Source/description for 'additional' dataset) [1](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
