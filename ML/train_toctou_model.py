
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

ML_DIR = Path(__file__).resolve().parent

DATA_PATH = ML_DIR / "toctou_training_data.csv"
MODEL_PATH = ML_DIR / "toctou_model.joblib"
REPORT_PATH = ML_DIR.parent / "reports"

FEATURES = [
    "check_duration_ms",
    "check_use_delay_ms",
    "use_duration_ms",
    "concurrent_requests",
    "duplicate_request_count",
]

TARGET = "label"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run toctou_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== TOCTOU MODEL EVALUATION ===")
    print("Total records:", len(df))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "TOCTOU"],
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        confusion_matrix(y_test, predictions),
        index=["Actual Normal", "Actual TOCTOU"],
        columns=["Predicted Normal", "Predicted TOCTOU"],
    ).to_csv(REPORT_PATH / "toctou_confusion_matrix.csv")

    print("\nModel saved:", MODEL_PATH)
    print("Evaluation report saved in:", REPORT_PATH)


if __name__ == "__main__":
    main()
