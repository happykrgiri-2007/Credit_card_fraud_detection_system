from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "creditcard.csv"

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

st.title("💳 Credit Card Fraud Detection")
st.caption("Machine Learning Internship Project | SAM AI Technologies")

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)

@st.cache_resource
def train_models(path, modified_time):
    """Train both models once per dataset version and cache the result."""
    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")
    if df["Class"].isna().any():
        raise ValueError("The Class column contains missing values.")
    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain 0 (legitimate) and 1 (fraud).")

    original_rows = len(df)
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "predictions": predictions,
            "probabilities": probabilities,
            "y_test": y_test
        }

        file_name = (
            "logistic_regression.pkl"
            if name == "Logistic Regression"
            else "random_forest.pkl"
        )
        joblib.dump(model, MODEL_DIR / file_name)

    best_name = max(results, key=lambda name: results[name]["f1_score"])
    best_model = models[best_name]
    joblib.dump(best_model, MODEL_DIR / "fraud_model.pkl")

    # Save metadata (without arrays used only for plotting).
    metadata = {
        "best_model": best_name,
        "feature_columns": feature_columns,
        "metrics": {
            name: {
                key: value for key, value in metrics.items()
                if key not in {"predictions", "probabilities", "y_test"}
            }
            for name, metrics in results.items()
        },
        "rows_before_deduplication": original_rows,
        "duplicate_rows_removed": duplicate_count,
        "rows_after_deduplication": len(df),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "fraud_count": int((y == 1).sum()),
        "legitimate_count": int((y == 0).sum())
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save selected-model confusion matrix.
    selected_cm = np.array(
    results[best_name]["confusion_matrix"]
)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=selected_cm,
        display_labels=["Legitimate", "Fraud"]
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Save fraud-class distribution chart.
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Class", ax=ax)
    ax.set_title("Legitimate vs Fraud Transactions")
    ax.set_xticks([0, 1], ["Legitimate (0)", "Fraud (1)"])
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    return df, models, results, metadata, best_model

# -------------------- CHECK DATASET --------------------
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Create a folder named 'dataset' beside app.py and put your "
        "creditcard.csv file inside it."
    )
    st.stop()

try:
    modified_time = DATA_PATH.stat().st_mtime
    with st.spinner("Loading dataset and training models. First run may take a few minutes..."):
        df, models, results, metadata, best_model = train_models(
            str(DATA_PATH), modified_time
        )
except Exception as exc:
    st.error(f"Could not load/train the dataset: {exc}")
    st.stop()

# -------------------- TABS --------------------
modified_time = DATA_PATH.stat().st_mtime

with st.spinner("Training models... please wait"):
    df, models, results, metadata, best_model = train_models(
        str(DATA_PATH), modified_time
    )
overview_tab, analysis_tab, performance_tab, prediction_tab = st.tabs([
    "Overview", "Data Analysis", "Model Performance", "Fraud Prediction"
])

with overview_tab:
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions", f'{len(df):,}')
    col2.metric("Legitimate", f'{metadata["legitimate_count"]:,}')
    col3.metric("Fraud", f'{metadata["fraud_count"]:,}')
    fraud_pct = metadata["fraud_count"] / len(df) * 100
    col4.metric("Fraud Percentage", f"{fraud_pct:.4f}%")

    st.write(f"Dataset columns: **{df.shape[1]}**")
    st.write(f"Exact duplicate rows removed before splitting: **{metadata['duplicate_rows_removed']:,}**")
    st.write("Preview of the dataset:")
    st.dataframe(df.head(10), use_container_width=True)

with analysis_tab:
    st.subheader("Exploratory Data Analysis")
    st.write("Class distribution")
    st.bar_chart(df["Class"].value_counts().sort_index().rename(index={
        0: "Legitimate", 1: "Fraud"
    }))
    st.write("Transaction amount distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(data=df, x="Amount", bins=50, ax=ax)
    ax.set_title("Transaction Amount Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.write("Missing values by column")
    missing = df.isna().sum()
    st.dataframe(
        missing[missing > 0].rename("Missing Values").to_frame()
        if (missing > 0).any()
        else pd.DataFrame({"Missing Values": [0]}, index=["No missing values"]),
        use_container_width=True
    )

with performance_tab:
    st.subheader("Model Comparison on Held-Out Test Data")
    comparison = pd.DataFrame([
        {
            "Model": name,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1-score": values["f1_score"],
            "ROC-AUC": values["roc_auc"],
            "Average Precision": values["average_precision"]
        }
        for name, values in metadata["metrics"].items()
    ]).set_index("Model")
    st.dataframe(comparison.round(4), use_container_width=True)
    st.success(f"Selected model by F1-score: **{metadata['best_model']}**")

    selected = metadata["metrics"][metadata["best_model"]]
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(
        selected["confusion_matrix"],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    ), use_container_width=True)
    st.caption(
        "Because the dataset is imbalanced, do not rely on accuracy alone. "
        "Review precision and recall together."
    )

with prediction_tab:
    st.subheader("Predict Transactions from a CSV")
    st.write(
        "Upload a CSV containing the same input feature columns as the training dataset. "
        "The Class column is optional and will be ignored."
    )
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            original = pd.read_csv(uploaded_file)
            incoming = original.drop(columns=["Class"], errors="ignore")
            required_columns = metadata["feature_columns"]
            missing_columns = [c for c in required_columns if c not in incoming.columns]

            if missing_columns:
                st.error("Missing required columns: " + ", ".join(missing_columns))
            elif original.empty:
                st.warning("The uploaded CSV contains no rows.")
            else:
                incoming = incoming[required_columns].apply(pd.to_numeric, errors="coerce")
                if incoming.isna().any().any():
                    st.error("Required feature values contain blanks or non-numeric values.")
                else:
                    predictions = best_model.predict(incoming)
                    probabilities = best_model.predict_proba(incoming)[:, 1]

                    output = original.copy()
                    output["Predicted_Class"] = predictions
                    output["Fraud_Probability"] = probabilities.round(6)
                    output["Prediction_Label"] = [
                        "Potential Fraud" if pred == 1 else "Likely Legitimate"
                        for pred in predictions
                    ]

                    st.metric("Potential fraud predictions", int((predictions == 1).sum()))
                    st.dataframe(output.head(100), use_container_width=True)
                    st.download_button(
                        "Download prediction results",
                        data=output.to_csv(index=False).encode("utf-8"),
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                    st.warning(
                        "Predictions are estimates, not proof of fraud. "
                        "This is an educational project, not a production banking system."
                    )
        except Exception as exc:
            st.error(f"Could not process uploaded CSV: {exc}")

st.divider()
st.caption("Models and charts are saved automatically in the models/ and reports/ folders.")
