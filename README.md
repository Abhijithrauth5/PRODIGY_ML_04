# Hand Gesture Recognition Model

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
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/gesture_model.py
```

The script prints the held-out validation accuracy and a confusion matrix whose rows are actual gestures and columns are predicted gestures.
