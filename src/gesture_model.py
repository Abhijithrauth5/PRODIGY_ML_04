"""Train and evaluate a hand-gesture classifier from landmark features."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "hand_gesture_landmarks.csv"
TARGET_COLUMN = "gesture"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Landmark dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a {TARGET_COLUMN!r} column")

    features = data.drop(columns=[TARGET_COLUMN])
    labels = data[TARGET_COLUMN]
    if features.empty or labels.nunique() < 2:
        raise ValueError("Dataset must contain features and at least two gesture classes")

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_valid)

    classes = sorted(labels.unique())
    matrix = confusion_matrix(y_valid, predictions, labels=classes)
    print(f"Validation accuracy: {accuracy_score(y_valid, predictions):.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=classes, columns=classes).to_string())


if __name__ == "__main__":
    main()
