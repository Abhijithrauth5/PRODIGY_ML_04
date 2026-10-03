#!/usr/bin/env python3
"""Create a runnable synthetic hand-gesture recognition project."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


GESTURES = ("Thumbs Up", "Peace", "Open Palm", "Fist", "OK Sign")
INSTANCES_PER_GESTURE = 30
LANDMARK_COUNT = 21


MODEL_SOURCE = '''"""Train and evaluate a hand-gesture classifier from landmark features."""

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
    print("\\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=classes, columns=classes).to_string())


if __name__ == "__main__":
    main()
'''


README = '''# Hand Gesture Recognition Model

## Objective

Developing an intuitive gesture recognition pipeline that classifies hand landmarks for system controls. This Task 4 project demonstrates the complete classification workflow with a compact synthetic dataset, so it runs immediately without downloading large video datasets.

## Project architecture

```text
PRODIGY_ML_04/
├── dataset/
│   └── hand_gesture_landmarks.csv  # 150 labeled synthetic landmark samples
├── src/
│   └── gesture_model.py            # Feature scaling, training, and validation
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── setup_gesture_project.py        # Recreates the project scaffold and sample data
```

## Dataset and feature extraction

- **Instances:** 150 synthetic samples (30 per gesture): Thumbs Up, Peace, Open Palm, Fist, and OK Sign.
- **Landmarks:** 21 hand landmarks per sample, represented by normalized `x`, `y`, and relative `z` coordinates.
- **Features:** 63 flattened values per sample (`21 landmarks × 3 coordinates`), plus a `gesture` label.
- **Input file:** `dataset/hand_gesture_landmarks.csv`.

The mock coordinates follow a MediaPipe-style hand landmark ordering and include small deterministic coordinate variations. Replace this CSV with real extracted landmarks to train on real observations.

## Model workflow

`src/gesture_model.py` loads the CSV, uses a stratified 80/20 training/validation split, scales features with `StandardScaler`, trains a class-balanced Random Forest, and prints validation accuracy and a labeled confusion matrix.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/gesture_model.py
```

The script prints the held-out validation accuracy and a confusion matrix whose rows are actual gestures and columns are predicted gestures.
'''


REQUIREMENTS = """pandas
numpy
scikit-learn
matplotlib
seaborn
"""


def build_landmarks(gesture: str, rng: random.Random) -> list[tuple[float, float, float]]:
    """Build 21 MediaPipe-ordered synthetic landmarks for one gesture."""
    points: list[tuple[float, float, float]] = [(0.50, 0.92, 0.0)]

    if gesture == "Thumbs Up":
        thumb = [(0.38, 0.70), (0.29, 0.61), (0.23, 0.46), (0.21, 0.28)]
    elif gesture == "OK Sign":
        thumb = [(0.37, 0.70), (0.34, 0.61), (0.40, 0.56), (0.45, 0.59)]
    elif gesture in ("Peace", "Fist"):
        thumb = [(0.38, 0.70), (0.43, 0.73), (0.46, 0.68), (0.42, 0.66)]
    else:
        thumb = [(0.38, 0.70), (0.29, 0.64), (0.20, 0.56), (0.12, 0.48)]
    points.extend((x, y, 0.0) for x, y in thumb)

    finger_bases = (0.40, 0.47, 0.54, 0.61)
    for finger_index, base_x in enumerate(finger_bases):
        is_extended = gesture == "Open Palm"
        if gesture == "Peace" and finger_index in (0, 1):
            is_extended = True
        if gesture == "OK Sign" and finger_index in (1, 2, 3):
            is_extended = True

        if gesture == "OK Sign" and finger_index == 0:
            joints = [
                (base_x, 0.68),
                (base_x + 0.045, 0.59),
                (base_x + 0.065, 0.53),
                (base_x + 0.015, 0.56),
            ]
        elif is_extended:
            spread = (finger_index - 1.5) * 0.018
            joints = [
                (base_x, 0.67),
                (base_x + spread * 0.35, 0.48),
                (base_x + spread * 0.70, 0.30),
                (base_x + spread, 0.13 if finger_index != 3 else 0.18),
            ]
        else:
            curl = (finger_index - 1.5) * 0.012
            joints = [
                (base_x, 0.67),
                (base_x + curl, 0.78),
                (base_x + curl * 1.6, 0.72),
                (base_x + curl * 2.0, 0.80),
            ]
        points.extend((x, y, 0.0) for x, y in joints)

    noisy_points = []
    for x, y, z in points:
        noisy_points.append(
            (
                min(1.0, max(0.0, x + rng.gauss(0, 0.012))),
                min(1.0, max(0.0, y + rng.gauss(0, 0.012))),
                min(0.25, max(-0.25, z + rng.gauss(0, 0.008))),
            )
        )
    return noisy_points


def dataset_rows() -> tuple[list[str], list[dict[str, object]]]:
    rng = random.Random(42)
    feature_names = [
        f"landmark_{landmark:02d}_{axis}"
        for landmark in range(LANDMARK_COUNT)
        for axis in ("x", "y", "z")
    ]
    rows: list[dict[str, object]] = []
    for gesture in GESTURES:
        for _ in range(INSTANCES_PER_GESTURE):
            coordinates = build_landmarks(gesture, rng)
            row: dict[str, object] = {"gesture": gesture}
            row.update(
                (name, round(value, 6))
                for name, value in zip(feature_names, (value for point in coordinates for value in point))
            )
            rows.append(row)
    rng.shuffle(rows)
    return ["gesture", *feature_names], rows


def render_dataset() -> str:
    fieldnames, rows = dataset_rows()
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def run_git(project_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(project_dir), *args],
        check=check,
        text=True,
        capture_output=True,
    )


def ensure_git_remote(project_dir: Path, remote_url: str | None) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("Git is required for the default setup; install Git or use --skip-git.")

    run_git(project_dir, "init")
    result = run_git(project_dir, "remote", "get-url", "origin", check=False)
    if result.returncode == 0:
        existing_url = result.stdout.strip()
        if remote_url and remote_url != existing_url:
            raise RuntimeError(
                f"origin already points to {existing_url!r}, not the supplied URL {remote_url!r}."
            )
        return
    if not remote_url:
        raise RuntimeError(
            "No origin remote is configured. Clone the empty GitHub repository first, "
            "or provide --remote <repository-url>."
        )


def setup_git(project_dir: Path, remote_url: str | None) -> None:
    result = run_git(project_dir, "remote", "get-url", "origin", check=False)
    if result.returncode != 0:
        assert remote_url is not None
        run_git(project_dir, "remote", "add", "origin", remote_url)

    run_git(project_dir, "add", "--all")
    staged_diff = run_git(project_dir, "diff", "--cached", "--quiet", check=False)
    if staged_diff.returncode == 1:
        run_git(
            project_dir,
            "-c",
            "user.name=Abijeed Rauth",
            "-c",
            "user.email=rauthabhijith@gmail.com",
            "commit",
            "-m",
            "Initial commit: hand gesture recognition",
        )
    elif staged_diff.returncode != 0:
        raise RuntimeError("Unable to inspect staged changes before committing.")
    else:
        head = run_git(project_dir, "rev-parse", "--verify", "HEAD", check=False)
        if head.returncode != 0:
            raise RuntimeError("No staged files are available for the required initial commit.")

    run_git(project_dir, "branch", "-M", "main")
    run_git(project_dir, "push", "-u", "origin", "main")


def write_project(project_dir: Path, overwrite: bool) -> None:
    content_by_path = {
        "dataset/hand_gesture_landmarks.csv": render_dataset(),
        "src/gesture_model.py": MODEL_SOURCE,
        "requirements.txt": REQUIREMENTS,
        "README.md": README,
    }

    conflicts = [
        relative_path
        for relative_path, content in content_by_path.items()
        if (project_dir / relative_path).exists()
        and (project_dir / relative_path).read_text(encoding="utf-8") != content
    ]
    if conflicts and not overwrite:
        formatted = ", ".join(conflicts)
        raise FileExistsError(
            f"Refusing to overwrite existing generated files: {formatted}. Use --force to replace them."
        )

    for relative_path, content in content_by_path.items():
        target = project_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic hand-gesture project and push its initial commit to origin/main."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="project root (defaults to the directory containing this script)",
    )
    parser.add_argument(
        "--remote",
        help="origin URL to configure if the project has no origin remote",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="generate the project files without initializing, committing, or pushing Git",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite generated project files that already exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_dir = args.project_dir.expanduser().resolve()
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_git:
            ensure_git_remote(project_dir, args.remote)
        write_project(project_dir, overwrite=args.force)
        if not args.skip_git:
            setup_git(project_dir, args.remote)
    except (FileExistsError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        return 1

    print(f"Project files created in {project_dir}")
    if args.skip_git:
        print("Git initialization, commit, and push were skipped.")
    else:
        print("Committed with the requested author identity and pushed to origin/main.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
