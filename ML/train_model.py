
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project paths
ML_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ML_DIR.parent

DATA_PATH = ML_DIR / "training_data.csv"
MODEL_PATH = ML_DIR / "decision_tree.joblib"
REPORT_DIR = PROJECT_DIR / "reports"

# Features used by the model
FEATURES = [
    "sql_keyword_count",
    "script_pattern_count",
    "login_frequency",
    "inter_arrival_ms",
    "concurrent_requests",
    "duplicate_request_count",
    "session_reuse",
]

TARGET = "label"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in training_data.csv: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Prepare features and labels
    df = df[required].copy()

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature], errors="raise"
        )

    if df[required].isnull().any().any():
        raise ValueError(
            "Dataset contains empty values. Clean them before training."
        )

    X = df[FEATURES]
    y = df[TARGET].astype(str)

    if y.nunique() < 2:
        raise ValueError(
            "At least two different attack/normal labels are needed."
        )

    # Split data while preserving class proportions where possible
    class_counts = y.value_counts()
    stratify_labels = (
        y if class_counts.min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_labels,
    )

    # Train Decision Tree
    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n===== MODEL EVALUATION =====")
    print(f"Rows in dataset: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Labels: {sorted(y.unique())}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Save model and feature names in the format app.py expects
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    # Save reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_,
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(report).transpose().to_csv(
        REPORT_DIR / "model_evaluation_report.csv"
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Confusion matrix saved in: {REPORT_DIR}")
    print("Training completed.")


if __name__ == "__main__":
    main()
