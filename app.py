# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, matthews_corrcoef
)

# Import model builders
from model.logistic_regression import build as build_logreg
from model.decision_tree import build as build_tree
from model.knn import build as build_knn
from model.naive_bayes import build as build_nb
from model.random_forest import build as build_rf
from model.xgboost_model import build as build_xgb

st.set_page_config(page_title="ML Assignment 2 - WDBC", page_icon="🧪", layout="wide")
st.title("Machine Learning Assignment 2 — Breast Cancer (WDBC)")

@st.cache_data
def load_dataset():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target  # 0 = malignant, 1 = benign (per sklearn docs)
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)
    return X, y, feature_names, target_names

X, y, feature_names, target_names = load_dataset()

st.markdown(
    f"""
    **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic)  
    **Instances:** {X.shape[0]} | **Features:** {X.shape[1]}  
    **Target classes:** {target_names}
    """
)

@st.cache_data
def get_models():
    return {
        "Logistic Regression": build_logreg(),
        "Decision Tree": build_tree(),
        "kNN": build_knn(),
        "Naive Bayes (Gaussian)": build_nb(),
        "Random Forest": build_rf(),
        "XGBoost": build_xgb()
    }

models = get_models()

def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["F1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    if y_prob is not None:
        try:
            metrics["AUC"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["AUC"] = np.nan
    else:
        metrics["AUC"] = np.nan
    return metrics

with st.sidebar:
    st.header("Controls")
    split = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    selected = st.selectbox(
        "Select a model (or view all)",
        ["All Models (comparison table)"] + list(models.keys())
    )
    st.markdown("---")
    uploaded = st.file_uploader(
        "Upload CSV with test-only data (same 30 feature columns). Optional.",
        type=["csv"]
    )

# Prepare train/test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=split, random_state=42, stratify=y
)

# If user uploaded a CSV, override test set
if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        # Ensure correct column order / presence
        missing = set(feature_names) - set(test_df.columns)
        extra = set(test_df.columns) - set(feature_names)
        if missing:
            st.error(f"Uploaded CSV is missing columns: {sorted(list(missing))}")
        else:
            X_test = test_df[feature_names]
            st.success("Using your uploaded CSV as test data.")
            st.write(X_test.head())
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    m = model
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    # Try to get probabilities for AUC; fall back to decision_function if available
    y_prob = None
    if hasattr(m, "predict_proba"):
        y_prob = m.predict_proba(X_test)[:, 1]
    elif hasattr(m, "decision_function"):
        # Map scores to probabilities via logistic if needed, but AUC accepts scores
        y_prob = m.decision_function(X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm, y_pred

# Train/evaluate
results = {}
conf_mats = {}
preds = {}

for name, model in models.items():
    metrics, cm, y_pred = evaluate_model(name, model, X_train, y_train, X_test, y_test)
    results[name] = metrics
    conf_mats[name] = cm
    preds[name] = y_pred

def results_table(res_dict):
    df = pd.DataFrame(res_dict).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    return df.sort_values(by="F1", ascending=False)

if selected == "All Models (comparison table)":
    st.subheader("Model Comparison")
    table = results_table(results)
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

    st.markdown("### Observations (fill in your README)")
    st.markdown(
        "- Which model has the best **F1**?\n"
        "- How do **tree-based** models compare to **linear** models?\n"
        "- Any overfitting signs (e.g., perfect train but lower test performance)?"
    )
else:
    st.subheader(f"Model: {selected}")
    met = results[selected]
    st.write(pd.DataFrame([met], index=[selected]).style.format("{:.4f}"))

    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(conf_mats[selected], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification report
    st.markdown("#### Classification Report")
    st.text(classification_report(y_test, preds[selected], target_names=target_names))