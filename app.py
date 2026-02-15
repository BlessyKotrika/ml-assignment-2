# app.py

import io
import sys
import importlib.util
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
    matthews_corrcoef,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# -----------------------------
# Configuration & Constants
# -----------------------------
st.set_page_config(
    page_title="Bank Marketing – EDA & Model Comparison",
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
    """Loads CSV from uploaded bytes."""
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
        y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype("Int64")

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, num_cols, cat_cols


def _make_ohe() -> OneHotEncoder:
    """
    Build a OneHotEncoder that plays well with sklearn 1.8+ (Python 3.13)
    and older sklearn versions.

    - In sklearn >=1.8, `sparse` is gone; use `sparse_output=False`.
    - In older versions, `sparse_output` may not exist; fall back to `sparse=False`.
    """
    try:
        # Newer sklearn (1.3+), and required for 1.8+
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Builds a ColumnTransformer with scaling for numeric and OHE for categorical."""
    ohe = _make_ohe()
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


def build_any_model(
        model_name: str,
        preprocessor: ColumnTransformer,
        lr_C: float = 1.0,
        rf_estimators: int = 300,
        rf_max_depth: Optional[int] = None,
        dt_max_depth: Optional[int] = None,
        knn_k: int = 15,
) -> Pipeline:
    """Returns a sklearn Pipeline with preprocessing + classifier for multiple algorithms."""
    if model_name == "Logistic Regression":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=lr_C,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(
            max_depth=dt_max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif model_name == "kNN":
        clf = KNeighborsClassifier(n_neighbors=knn_k, weights="distance")
    elif model_name == "Naive Bayes":
        clf = GaussianNB()
    elif model_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=rf_max_depth,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "XGBoost":
        if importlib.util.find_spec("xgboost") is None:
            raise ImportError("xgboost is not installed. Try: pip install xgboost")
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", clf)])
    return pipe


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes standard metrics and ROC curve data."""
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None else np.nan
    cm = confusion_matrix(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)

    fpr, tpr, _ = roc_curve(y_test, proba) if proba is not None else (None, None, None)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "mcc": mcc,
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
    ohe = None
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = transformer
            break

    num_out = num_cols
    cat_out = []
    if ohe is not None:
        try:
            cat_out = ohe.get_feature_names_out(cat_cols).tolist()
        except Exception:
            cat_out = []
            if hasattr(ohe, "categories_"):
                for col_idx, col in enumerate(cat_cols):
                    for cat_val in ohe.categories_[col_idx]:
                        cat_out.append(f"{col}_{cat_val}")
            else:
                cat_out = cat_cols

    return [*num_out, *cat_out]


def extract_feature_importance(model: Pipeline, num_cols: List[str], cat_cols: List[str]) -> Optional[pd.DataFrame]:
    """Extracts feature importances for tree-based model."""
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


def profiling_available() -> bool:
    return importlib.util.find_spec("ydata_profiling") is not None


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Configuration")

data_source = st.sidebar.radio("Dataset Source", ["Use default path", "Upload CSV"])
sep_choice = st.sidebar.text_input("CSV separator", value=";")

# Quick mode (sampling)
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
# OFF by default due to Python 3.13 ecosystem variance
enable_profile = st.sidebar.checkbox("Generate Profiling Report (if available)", value=False,
                                     help="Turn on if ydata-profiling is installed and works in your env.")
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05, format="%.2f")
cv_folds = st.sidebar.slider("Cross-Validation folds", min_value=3, max_value=10, value=5, step=1)
st.sidebar.markdown("---")

# Single-model training selection + a few hyperparams
model_choice = st.sidebar.selectbox("Single-Model Training", ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"])
lr_C = st.sidebar.slider("LogReg: C (inverse regularization)", 0.01, 10.0, 1.0, 0.01)
dt_depth = st.sidebar.select_slider("DecisionTree: max_depth", options=[None] + list(range(3, 31)), value=None)
knn_k = st.sidebar.slider("kNN: k", 3, 75, 15, 1)
rf_estimators = st.sidebar.slider("RandomForest: n_estimators", 100, 1000, 300, 50)
rf_max_depth = st.sidebar.select_slider("RandomForest: max_depth", options=[None] + list(range(3, 31)), value=None)

# Choose models to compare
compare_models = st.sidebar.multiselect(
    "Models to compare (for the Test Set table)",
    ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"],
    default=["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"],
)
auto_run_compare = st.sidebar.checkbox("Run comparison automatically", value=False)


# -----------------------------
# Main Layout
# -----------------------------
st.title("📊 Bank Marketing – EDA & Model Comparison")
st.caption("UCI Bank Marketing dataset (bank-additional-full.csv) • Streamlit app")

# Load Data
st.subheader("1) Load Data")

try:
    df = None
    if data_source == "Use default path":
        if nrows_opt and nrows_opt > 0:
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
    if not profiling_available():
        st.info("Install profiling with: `pip install ydata-profiling` (pandas_profiling is deprecated).")
    else:
        try:
            from ydata_profiling import ProfileReport  # type: ignore
            profile = ProfileReport(df, minimal=quick_mode, explorative=True)
            html = profile.to_html()
            st.download_button(
                "Download profiling report (HTML)",
                data=html,
                file_name="bank_profile.html",
                mime="text/html",
            )
            st.success("Profiling report ready.")
        except Exception as e:
            st.warning(f"Profiling failed: {e}")


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
    st.caption("No missing values detected.")

# Correlations (numeric only)
num_df = df.select_dtypes(include=[np.number])
if not num_df.empty:
    st.write("**Numeric feature correlations (heatmap):**")
    try:
        corr = num_df.corr(numeric_only=True)  # pandas >= 1.5
    except TypeError:
        corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)
else:
    st.caption("No numeric columns found for correlation heatmap.")


# -----------------------------
# Modeling (Single Model)
# -----------------------------
st.subheader("3) Modeling & Evaluation (Single Model)")

try:
    X, y, num_cols, cat_cols = split_features(df, TARGET_COL)
except Exception as e:
    st.error(f"Cannot split features/target: {e}")
    st.stop()

with st.expander("Detected feature types"):
    st.write("**Numeric columns:**", num_cols)
    st.write("**Categorical columns:**", cat_cols)

# Train-test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int), test_size=test_size, random_state=RANDOM_STATE, stratify=y.astype(int)
    )
except Exception as e:
    st.error(f"Train-test split failed: {e}")
    st.stop()

preprocessor = make_preprocessor(num_cols, cat_cols)

# Build the single model pipeline
single_pipe = None
try:
    single_pipe = build_any_model(
        model_name=model_choice,
        preprocessor=preprocessor,
        lr_C=lr_C,
        rf_estimators=rf_estimators,
        rf_max_depth=rf_max_depth,
        dt_max_depth=dt_depth,
        knn_k=knn_k,
    )
except ImportError as e:
    st.warning(str(e))

col_train, col_cv = st.columns(2)
with col_train:
    if st.button("🚀 Train single model"):
        if single_pipe is None:
            st.stop()
        with st.spinner("Training..."):
            single_pipe.fit(X_train, y_train)
            st.session_state["model"] = single_pipe
        st.success(f"{model_choice} trained.")

        # Eval on test
        metrics = evaluate_model(single_pipe, X_test, y_test)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{metrics['precision']:.3f}")
        m3.metric("Recall", f"{metrics['recall']:.3f}")
        m4.metric("F1-score", f"{metrics['f1']:.3f}")
        m5.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if not np.isnan(metrics["roc_auc"]) else "N/A")
        m6.metric("MCC", f"{metrics['mcc']:.3f}")

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

        # Feature importance (tree ensembles)
        if model_choice in ["Random Forest", "XGBoost"]:
            fi = extract_feature_importance(single_pipe, num_cols, cat_cols)
            if fi is not None and not fi.empty:
                st.write("**Top Feature Importances:**")
                st.dataframe(fi, use_container_width=True)
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.barplot(y="feature", x="importance", data=fi.head(20), ax=ax, palette="viridis")
                ax.set_title("Top 20 Feature Importances")
                st.pyplot(fig)
            else:
                st.caption("Feature importances unavailable.")

with col_cv:
    if st.button("📐 Cross-Validate (Single Model)"):
        if single_pipe is None:
            st.stop()
        with st.spinner("Running cross-validation..."):
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
            try:
                scores = cross_val_score(single_pipe, X, y.astype(int), cv=cv, scoring="roc_auc", n_jobs=-1)
                st.success("Cross-validation (ROC-AUC) complete.")
                st.write(f"ROC-AUC: mean={scores.mean():.3f}, std={scores.std():.3f}")
            except Exception:
                scores = cross_val_score(single_pipe, X, y.astype(int), cv=cv, scoring="accuracy", n_jobs=-1)
                st.success("Cross-validation (Accuracy) complete.")
                st.write(f"Accuracy: mean={scores.mean():.3f}, std={scores.std():.3f}")


# -----------------------------
# Comparison (Multiple Models on the same Test Set)
# -----------------------------
st.subheader("4) Comparison Table (Test Set)")
st.caption("Fill this table using the app or run `python train.py` (generates `model_comparison.csv`).")

# Persist comparison results
if "cmp_df" not in st.session_state:
    st.session_state["cmp_df"] = None

def run_model_comparison(
        models: List[str],
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        preprocessor: ColumnTransformer,
        lr_C_val: float, rf_estimators_val: int, rf_max_depth_val: Optional[int],
        dt_max_depth_val: Optional[int] = None, knn_k_val: int = 15
) -> pd.DataFrame:
    rows = []
    for name in models:
        try:
            pipe = build_any_model(
                model_name=name,
                preprocessor=preprocessor,
                lr_C=lr_C_val,
                rf_estimators=rf_estimators_val,
                rf_max_depth=rf_max_depth_val,
                dt_max_depth=dt_max_depth_val,
                knn_k=knn_k_val,
            )
        except ImportError as e:
            st.warning(f"Skipping {name}: {e}")
            continue

        with st.spinner(f"Training {name}..."):
            pipe.fit(X_train, y_train)

        metrics = evaluate_model(pipe, X_test, y_test)
        rows.append({
            "ML Model Name": name,
            "Accuracy": metrics["accuracy"],
            "AUC": metrics["roc_auc"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "MCC": metrics["mcc"],
        })

    if not rows:
        return pd.DataFrame(columns=["ML Model Name","Accuracy","AUC","Precision","Recall","F1","MCC"])

    out = pd.DataFrame(rows)
    return out.sort_values(["F1","AUC","MCC"], ascending=[False, False, False]).reset_index(drop=True)

col_cmp_left, col_cmp_right = st.columns([1,1])
with col_cmp_left:
    do_run = st.button("🔁 Run Comparison", help="Train selected models and score on the test set.")
with col_cmp_right:
    show_autorun = st.checkbox("Auto-run comparison on load", value=auto_run_compare, help="Runs once on first load / when data changes.")

# Trigger comparison run
should_run_now = do_run or (show_autorun and st.session_state.get("cmp_df") is None)
if should_run_now:
    cmp_df = run_model_comparison(
        compare_models,
        X_train, y_train, X_test, y_test,
        preprocessor,
        lr_C_val=lr_C,
        rf_estimators_val=rf_estimators,
        rf_max_depth_val=rf_max_depth,
        dt_max_depth_val=dt_depth,
        knn_k_val=knn_k
    )
    st.session_state["cmp_df"] = cmp_df

# Show current / last comparison
cmp_df = st.session_state.get("cmp_df", None)

if cmp_df is None or cmp_df.empty:
    st.info("No comparison results yet. Click **🔁 Run Comparison** to populate the table.")
    st.markdown("### Comparison Table (Test Set)")
    st.markdown("> Fill this table using the app or `python train.py` (generates `model_comparison.csv`).")
    st.markdown(
        "| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |\n"
        "|---------------------|---------:|------:|----------:|------:|------:|------:|\n"
        "| Logistic Regression |          |       |           |       |       |       |\n"
        "| Decision Tree       |          |       |           |       |       |       |\n"
        "| kNN                 |          |       |           |       |       |       |\n"
        "| Naive Bayes         |          |       |           |       |       |       |\n"
        "| Random Forest       |          |       |           |       |       |       |\n"
        "| XGBoost             |          |       |           |       |       |       |"
    )
else:
    st.dataframe(
        cmp_df.style.format({
            "Accuracy": "{:.3f}",
            "AUC": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1": "{:.3f}",
            "MCC": "{:.3f}",
        }),
        use_container_width=True
    )

    st.markdown("#### Comparison Table (Test Set)")
    header = "| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |\n|---------------------|---------:|------:|----------:|------:|------:|------:|"
    lines = [header]
    for _, r in cmp_df.iterrows():
        lines.append(
            f"| {r['ML Model Name']:<19} | {r['Accuracy']:.3f} | {r['AUC']:.3f} | "
            f"{r['Precision']:.3f} | {r['Recall']:.3f} | {r['F1']:.3f} | {r['MCC']:.3f} |"
        )
    st.markdown("\n".join(lines))

    st.download_button(
        "⬇️ Download model_comparison.csv",
        data=cmp_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    st.markdown("#### Insights (F1 / AUC / MCC & Bias–Variance)")
    try:
        top_f1 = cmp_df.loc[cmp_df["F1"].idxmax()]
        top_auc = cmp_df.loc[cmp_df["AUC"].idxmax()]
        top_mcc = cmp_df.loc[cmp_df["MCC"].idxmax()]

        bullets = []
        bullets.append(f"**F1 winner:** `{top_f1['ML Model Name']}` with F1 = **{top_f1['F1']:.3f}** (balanced precision/recall).")
        bullets.append(f"**AUC winner:** `{top_auc['ML Model Name']}` with AUC = **{top_auc['AUC']:.3f}** (best rank separation).")
        bullets.append(f"**MCC winner:** `{top_mcc['ML Model Name']}` with MCC = **{top_mcc['MCC']:.3f}** (robust to class imbalance).")

        names = set(cmp_df["ML Model Name"])
        if {"Decision Tree", "Random Forest"}.issubset(names):
            dt_row = cmp_df[cmp_df["ML Model Name"] == "Decision Tree"].iloc[0]
            rf_row = cmp_df[cmp_df["ML Model Name"] == "Random Forest"].iloc[0]
            bullets.append(
                f"**Variance reduction:** Random Forest improves over Decision Tree "
                f"(ΔF1 = {rf_row['F1']-dt_row['F1']:+.3f}, ΔAUC = {rf_row['AUC']-dt_row['AUC']:+.3f}, ΔMCC = {rf_row['MCC']-dt_row['MCC']:+.3f})."
            )
        if {"Logistic Regression", "XGBoost"}.issubset(names):
            lr_row = cmp_df[cmp_df["ML Model Name"] == "Logistic Regression"].iloc[0]
            xg_row = cmp_df[cmp_df["ML Model Name"] == "XGBoost"].iloc[0]
            trend = "outperforms" if xg_row["F1"] >= lr_row["F1"] else "does not outperform"
            bullets.append(
                f"**Bias vs flexibility:** Logistic Regression (linear, higher bias) {trend} XGBoost (non-linear, lower bias) on F1 "
                f"({xg_row['F1']:.3f} vs {lr_row['F1']:.3f})."
            )
        if "kNN" in names:
            bullets.append("**kNN** uses distance weighting; small *k* may increase variance—tune *k* for stability.")
        if "Naive Bayes" in names:
            bullets.append("**Naive Bayes** (high-bias independence assumption) is fast but may underfit mixed tabular data.")

        st.markdown("\n".join([f"- {b}" for b in bullets[:5]]))
    except Exception as e:
        st.caption(f"Insight generation skipped: {e}")


# -----------------------------
# Inference (Single Prediction)
# -----------------------------
st.subheader("5) Predict on a Single Example")
st.markdown("Use the sidebar form to construct a single input and predict.")

with st.sidebar.expander("🧪 Single Prediction Input", expanded=False):
    single_input = {}
    for col in X.columns:
        if col in num_cols:
            default_val = float(np.nanmedian(pd.to_numeric(X[col], errors="coerce")))
            val = st.number_input(f"{col}", value=float(default_val) if np.isfinite(default_val) else 0.0, step=1.0, format="%.3f")
            single_input[col] = val
        else:
            opts = sorted(list(map(str, X[col].dropna().unique())))
            val = st.selectbox(f"{col}", options=opts if opts else [""], index=0)
            single_input[col] = val
    do_predict = st.button("🔮 Predict (Positive vs Negative)")

if do_predict:
    if "model" not in st.session_state:
        with st.spinner("No trained model found. Training quickly with current settings..."):
            try:
                quick_pipe = build_any_model(
                    model_name=model_choice,
                    preprocessor=preprocessor,
                    lr_C=lr_C,
                    rf_estimators=rf_estimators,
                    rf_max_depth=rf_max_depth,
                    dt_max_depth=dt_depth,
                    knn_k=knn_k,
                )
            except ImportError as e:
                st.error(str(e))
                st.stop()
            quick_pipe.fit(X_train, y_train)
            st.session_state["model"] = quick_pipe

    pipe = st.session_state["model"]
    single_df = pd.DataFrame([single_input], columns=X.columns)
    try:
        proba = pipe.predict_proba(single_df)[:, 1][0]
        pred = int(proba >= 0.5)
        st.write(f"**Predicted class:** {'yes' if pred == 1 else 'no'}  •  **Probability (positive):** {proba:.3f}")
    except Exception:
        try:
            pred = pipe.predict(single_df)[0]
            st.write(f"**Predicted class:** {'yes' if int(pred) == 1 else 'no'}")
        except Exception as e2:
            st.error(f"Prediction failed: {e2}")


# Persist the trained model in session (if trained above)
if st.button("💾 Save current model in session"):
    try:
        if single_pipe is None:
            single_pipe = build_any_model(
                model_name=model_choice,
                preprocessor=preprocessor,
                lr_C=lr_C,
                rf_estimators=rf_estimators,
                rf_max_depth=rf_max_depth,
                dt_max_depth=dt_depth,
                knn_k=knn_k,
            )
        single_pipe.fit(X_train, y_train)
        st.session_state["model"] = single_pipe
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
    "- For profiling, install `ydata-profiling` (modern replacement for pandas_profiling).\n"
    "- Target column `y` is mapped to 1 (yes) / 0 (no).\n"
    f"- Running on Python: {sys.version.split()[0]}"
)
