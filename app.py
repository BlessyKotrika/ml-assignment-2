# app.py
# Streamlit app for UCI Bank Marketing dataset (bank-additional-full.csv)
# Author: Blessy Kotrika (GE Appliances, Haier) – tailored by M365 Copilot
# ---------------------------------------------------------------
# Features:
# - Robust CSV loading (semicolon separator) + optional upload
# - Quick Mode switch (sampling + lighter visuals)
# - EDA: schema, missing values, distribution, correlations
# - ML: Logistic Regression / Random Forest with proper preprocessing
# - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC + ROC curve
# - Feature importance (for Random Forest)
# - Single-sample prediction form
# - Optional data profiling if ydata_profiling is installed
# ---------------------------------------------------------------

import io
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------
# Configuration & Constants
# -----------------------------
st.set_page_config(
    page_title="Bank Marketing – EDA & ML",
    page_icon="📊",
    layout="wide",
)

DEFAULT_DATA_PATH = "/Users/blessykotrika/Documents/GitHub/ml-assignment-2/data/bank-additional-full.csv"
TARGET_COL = "y"
RANDOM_STATE = 42


# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=True)
def load_csv_from_path(path_str: str, sep: str = ";", nrows: Optional[int] = None) -> pd.DataFrame:
    """Loads CSV from a filesystem path with safe defaults."""
    p = Path(path_str).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p, sep=sep, encoding="utf-8", engine="python", nrows=nrows)
    return df


def load_csv_from_bytes(bytes_data: bytes, sep: str = ";", nrows: Optional[int] = None) -> pd.DataFrame:
    """Loads CSV from uploaded bytes (no caching for simplicity)."""
    df = pd.read_csv(io.BytesIO(bytes_data), sep=sep, encoding="utf-8", engine="python", nrows=nrows)
    return df


def split_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Separates features and target, detects numeric and categorical columns."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data.")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Convert target 'yes'/'no' to 1/0 if needed
    if y.dtype == object:
        y = y.str.strip().str.lower().map({"yes": 1, "no": 0}).astype("Int64")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, num_cols, cat_cols


def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Builds a ColumnTransformer with scaling for numeric and OHE for categorical."""
    # Note: OneHotEncoder(sparse=False) for compatibility across sklearn versions.
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler(with_mean=True, with_std=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


def build_model(
        model_name: str,
        preprocessor: ColumnTransformer,
        lr_C: float = 1.0,
        rf_estimators: int = 300,
        rf_max_depth: Optional[int] = None,
) -> Pipeline:
    """Returns a sklearn Pipeline with preprocessing + classifier."""
    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=lr_C,
            solver="lbfgs",
            n_jobs=None,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=rf_max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError("Unsupported model.")

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
    return pipe


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes standard metrics and ROC curve data."""
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: some classifiers don't implement predict_proba
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # Normalize to 0-1 for ROC-AUC compatibility
            proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None else np.nan
    cm = confusion_matrix(y_test, preds)

    fpr, tpr, _ = roc_curve(y_test, proba) if proba is not None else (None, None, None)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "proba": proba,
        "preds": preds,
    }


def plot_roc(fpr, tpr, auc_val):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    if fpr is not None and tpr is not None:
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})", color="tab:blue")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    """Retrieves output feature names after ColumnTransformer (numeric + OHE categories)."""
    # For OneHotEncoder, scikit-learn >=1.0 has get_feature_names_out
    ohe = None
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = transformer
            break

    num_out = num_cols  # StandardScaler retains names (not expanded)
    cat_out = []
    if ohe is not None:
        try:
            cat_out = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            # Compatibility fallback
            cat_out = []
            for col_idx, col in enumerate(cat_cols):
                # If categories_ exist:
                if hasattr(ohe, "categories_"):
                    for cat_val in ohe.categories_[col_idx]:
                        cat_out.append(f"{col}_{cat_val}")
                else:
                    cat_out.append(col)

    return [*num_out, *cat_out]


def extract_feature_importance(model: Pipeline, num_cols: List[str], cat_cols: List[str]) -> Optional[pd.DataFrame]:
    """Extracts feature importances for tree-based model (RandomForest)."""
    try:
        clf = model.named_steps["model"]
        pre = model.named_steps["preprocessor"]
        if hasattr(clf, "feature_importances_"):
            names = get_feature_names_from_preprocessor(pre, num_cols, cat_cols)
            importances = clf.feature_importances_
            fi = pd.DataFrame({"feature": names, "importance": importances})
            fi = fi.sort_values("importance", ascending=False).head(30).reset_index(drop=True)
            return fi
    except Exception:
        return None
    return None


def safe_bar_chart(series: pd.Series, title: str, color: str = "tab:blue"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    series.plot(kind="bar", ax=ax, color=color)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel(series.name if series.name else "")
    st.pyplot(fig)


# -----------------------------
# Sidebar Controls (incl. quick_mode)
# -----------------------------
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio("Dataset Source", ["Use default path", "Upload CSV"])
sep_choice = st.sidebar.text_input("CSV separator", value=";")

# Define quick_mode to avoid NameError and to drive behavior throughout.
quick_mode = st.sidebar.checkbox("Quick Mode (faster & lighter)", value=True)

nrows_opt = st.sidebar.number_input(
    "Rows to load (0 = all; use for sampling in Quick Mode)",
    min_value=0,
    value=5000 if quick_mode else 0,
    step=1000,
    help="If > 0, only this many rows are loaded (random sample).",
)

default_path = st.sidebar.text_input("Default dataset path", value=DEFAULT_DATA_PATH)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload bank-additional-full.csv", type=["csv"])

st.sidebar.markdown("---")
enable_profile = st.sidebar.checkbox("Generate Profiling Report (if available)", value=False)
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05, format="%.2f")
cv_folds = st.sidebar.slider("Cross-Validation folds", min_value=3, max_value=10, value=5, step=1)
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest"])
if model_choice == "Logistic Regression":
    lr_C = st.sidebar.slider("LogReg: C (inverse regularization)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    rf_estimators = None
    rf_max_depth = None
else:
    lr_C = None
    rf_estimators = st.sidebar.slider("RandomForest: n_estimators", min_value=100, max_value=1000, value=300, step=50)
    rf_max_depth = st.sidebar.select_slider(
        "RandomForest: max_depth",
        options=[None] + list(range(3, 31, 1)),
        value=None,
    )

# -----------------------------
# Main Layout
# -----------------------------
st.title("📊 Bank Marketing – EDA & ML")
st.caption("UCI Bank Marketing dataset (bank-additional-full.csv) • Streamlit app")

# Load Data
st.subheader("1) Load Data")

try:
    df = None
    if data_source == "Use default path":
        if nrows_opt and nrows_opt > 0:
            # For sampling reproducibly: read all then sample; but reading all defeats sampling purpose for huge files.
            # This dataset (~41k rows) is small; reading fully is fine.
            tmp_df = load_csv_from_path(default_path, sep=sep_choice, nrows=None)
            df = tmp_df.sample(n=min(nrows_opt, len(tmp_df)), random_state=RANDOM_STATE)
        else:
            df = load_csv_from_path(default_path, sep=sep_choice, nrows=None)
    else:
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            tmp_df = load_csv_from_bytes(file_bytes, sep=sep_choice, nrows=None)
            if nrows_opt and nrows_opt > 0:
                df = tmp_df.sample(n=min(nrows_opt, len(tmp_df)), random_state=RANDOM_STATE)
            else:
                df = tmp_df
        else:
            st.info("Please upload a CSV file to continue.")
            st.stop()

    st.success(f"Loaded dataset with shape: {df.shape}")
    st.write(df.head(10) if quick_mode else df.head(20))

    with st.expander("Column info & dtypes"):
        info_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
        st.dataframe(info_df, use_container_width=True)

except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()


# -----------------------------
# Optional Profiling
# -----------------------------
if enable_profile:
    st.subheader("🔎 Profiling Report")
    try:
        # Import lazily to avoid dependency errors when not installed
        try:
            from ydata_profiling import ProfileReport  # type: ignore
        except Exception:
            from pandas_profiling import ProfileReport  # fallback for older envs

        profile = ProfileReport(
            df,
            minimal=quick_mode,       # <- The flag is explicitly defined
            explorative=True
        )
        html = profile.to_html()
        st.download_button(
            "Download profiling report (HTML)",
            data=html,
            file_name="bank_profile.html",
            mime="text/html",
        )
        st.success("Profiling report generated.")
        st.caption("Note: If the button doesn't appear, ensure ydata_profiling/pandas_profiling is installed.")
    except Exception as e:
        st.warning(f"Profiling not available or failed to generate: {e}")


# -----------------------------
# EDA
# -----------------------------
st.subheader("2) Exploratory Data Analysis")

# Target distribution
if TARGET_COL in df.columns:
    tgt_counts = df[TARGET_COL].astype(str).str.strip().str.lower().value_counts()
    st.write("**Target distribution (y):**")
    safe_bar_chart(tgt_counts, "Target Distribution (y)")

# Missing values
missing = df.isna().sum()
if missing.sum() > 0:
    st.write("**Missing values per column:**")
    safe_bar_chart(missing[missing > 0].sort_values(ascending=False), "Missing Values", color="tab:red")
else:
    st.info("No missing values detected.")

# Correlations (numeric only)
num_df = df.select_dtypes(include=[np.number])
if not num_df.empty:
    st.write("**Numeric feature correlations (heatmap):**")
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
else:
    st.caption("No numeric columns found for correlation heatmap.")

# -----------------------------
# Modeling
# -----------------------------
st.subheader("3) Modeling & Evaluation")

try:
    X, y, num_cols, cat_cols = split_features(df, TARGET_COL)
except Exception as e:
    st.error(f"Cannot split features/target: {e}")
    st.stop()

# Show detected columns
with st.expander("Detected feature types"):
    st.write("**Numeric columns:**", num_cols)
    st.write("**Categorical columns:**", cat_cols)

# Train-test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int),
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y.astype(int)
    )
except Exception as e:
    st.error(f"Train-test split failed: {e}")
    st.stop()

preprocessor = make_preprocessor(num_cols, cat_cols)

# Build the model pipeline
pipe = build_model(
    model_name=model_choice,
    preprocessor=preprocessor,
    lr_C=lr_C if lr_C is not None else 1.0,
    rf_estimators=rf_estimators if rf_estimators is not None else 300,
    rf_max_depth=rf_max_depth,
)

col_train, col_cv = st.columns(2)
with col_train:
    if st.button("🚀 Train model"):
        with st.spinner("Training..."):
            pipe.fit(X_train, y_train)
        st.success(f"{model_choice} trained.")

        # Eval on test
        metrics = evaluate_model(pipe, X_test, y_test)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("F1-score", f"{metrics['f1']:.3f}")
        m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if not np.isnan(metrics["roc_auc"]) else "N/A")

        # Confusion matrix
        st.write("**Confusion Matrix:**")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC curve
        st.write("**ROC Curve:**")
        plot_roc(metrics["fpr"], metrics["tpr"], metrics["roc_auc"])

        # Feature importance (RF only)
        if model_choice == "Random Forest":
            fi = extract_feature_importance(pipe, num_cols, cat_cols)
            if fi is not None and not fi.empty:
                st.write("**Top Feature Importances (Random Forest):**")
                st.dataframe(fi, use_container_width=True)
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.barplot(y="feature", x="importance", data=fi.head(20), ax=ax, palette="viridis")
                ax.set_title("Top 20 Feature Importances")
                st.pyplot(fig)
            else:
                st.caption("Feature importances unavailable.")

with col_cv:
    if st.button("📐 Cross-Validate"):
        with st.spinner("Running cross-validation..."):
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            # Use ROC-AUC where possible
            try:
                scores = cross_val_score(pipe, X, y.astype(int), cv=cv, scoring="roc_auc", n_jobs=-1)
                st.success("Cross-validation (ROC-AUC) complete.")
                st.write(f"ROC-AUC: mean={scores.mean():.3f}, std={scores.std():.3f}")
            except Exception:
                scores = cross_val_score(pipe, X, y.astype(int), cv=cv, scoring="accuracy", n_jobs=-1)
                st.success("Cross-validation (Accuracy) complete.")
                st.write(f"Accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")


# -----------------------------
# Inference (Single Prediction)
# -----------------------------
st.subheader("4) Predict on a Single Example")

st.markdown("Use the sidebar form to construct a single input and predict.")

with st.sidebar.expander("🧪 Single Prediction Input", expanded=False):
    # Build widgets based on training columns
    single_input = {}
    for col in X.columns:
        if col in num_cols:
            # Guess a reasonable range from data
            col_min, col_max = float(X[col].min()), float(X[col].max())
            default_val = float(X[col].median() if np.isfinite(X[col].median()) else 0.0)
            val = st.number_input(f"{col}", value=default_val, step=1.0, format="%.3f")
            single_input[col] = val
        else:
            # Categorical: pick from observed
            opts = sorted(list(map(str, X[col].dropna().unique())))
            default_opt = opts[0] if opts else ""
            val = st.selectbox(f"{col}", options=opts if opts else [""], index=0 if opts else None)
            single_input[col] = val

    do_predict = st.button("🔮 Predict (Positive vs Negative)")

if do_predict:
    if "model" not in st.session_state:
        # Try fitting a quick model if not yet trained during this session
        with st.spinner("No trained model found. Training quickly with current settings..."):
            pipe.fit(X_train, y_train)
            st.session_state["model"] = pipe
    else:
        pipe = st.session_state["model"]

    single_df = pd.DataFrame([single_input], columns=X.columns)
    try:
        proba = pipe.predict_proba(single_df)[:, 1][0]
        pred = int(proba >= 0.5)
        st.write(f"**Predicted class:** {'yes' if pred == 1 else 'no'}  •  **Probability (positive):** {proba:.3f}")
    except Exception as e:
        # Some models might not have predict_proba; fallback
        try:
            pred = pipe.predict(single_df)[0]
            st.write(f"**Predicted class:** {'yes' if int(pred) == 1 else 'no'}")
        except Exception as e2:
            st.error(f"Prediction failed: {e2}")


# Persist the trained model in session (if trained above)
if st.button("💾 Save current model in session"):
    try:
        pipe.fit(X_train, y_train)
        st.session_state["model"] = pipe
        st.success("Model saved in session. You can now run single predictions reliably.")
    except Exception as e:
        st.error(f"Failed to save model: {e}")


# -----------------------------
# Footer / Tips
# -----------------------------
st.markdown("---")
st.caption(
    "Tips:\n"
    "- The dataset uses a semicolon (;) separator.\n"
    "- Use **Quick Mode** to sample rows for faster exploration.\n"
    "- If you enable profiling, ensure `ydata_profiling` (or `pandas_profiling`) is installed.\n"
    "- Target column `y` is mapped to 1 (yes) / 0 (no)."
)