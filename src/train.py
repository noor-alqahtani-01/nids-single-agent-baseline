"""Train the static detector for the single-agent NIDS baseline.

Reads data/processed/train.csv and test.csv (made by prepare_data.py),
trains one multi-class Random Forest, and writes:
    models/detector.joblib   (model + feature names + class names)
    models/metrics.json      (test-set scores)

The model predicts the attack type. The binary verdict is derived from it:
anything that is not BENIGN counts as an attack.

Usage:
    python src/train.py
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, recall_score

SEED = 42
BENIGN = "BENIGN"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
NON_FEATURES = ["attack_type", "label"]


def load(name: str):
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python src/prepare_data.py")
    df = pd.read_csv(path)
    X = df.drop(columns=NON_FEATURES)
    y = df["attack_type"].to_numpy(dtype=object)
    return X, y


def main() -> None:
    print("Loading data ...")
    X_train, y_train = load("train.csv")
    X_test, y_test = load("test.csv")
    features = list(X_train.columns)
    print(f"Train rows: {len(X_train):,}   Test rows: {len(X_test):,}   Features: {len(features)}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=25,
        min_samples_leaf=2,
        class_weight="balanced",   # benign traffic far outnumbers attacks
        random_state=SEED,
        n_jobs=-1,
    )
    print("Training (this can take a few minutes) ...")
    start = time.time()
    model.fit(X_train, y_train)
    train_seconds = round(time.time() - start, 1)

    pred = model.predict(X_test[features])

    # Binary view: attack (1) vs benign (0).
    true_bin = (y_test != BENIGN).astype(int)
    pred_bin = (pred != BENIGN).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_bin, pred_bin, average="binary", zero_division=0
    )

    # Recall per attack type: shows where the baseline is weak.
    classes = sorted(set(y_test))
    per_class = recall_score(y_test, pred, labels=classes, average=None, zero_division=0)
    per_class_recall = {
        c: {"recall": round(float(r), 4), "test_rows": int((y_test == c).sum())}
        for c, r in zip(classes, per_class)
    }

    metrics = {
        "model": "RandomForestClassifier",
        "params": {"n_estimators": 100, "max_depth": 25, "min_samples_leaf": 2,
                   "class_weight": "balanced", "random_state": SEED},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_seconds": train_seconds,
        "binary": {"precision": round(float(precision), 4),
                   "recall": round(float(recall), 4),
                   "f1": round(float(f1), 4)},
        "multiclass_accuracy": round(float((pred == y_test).mean()), 4),
        "per_class_recall": per_class_recall,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "detector.joblib"
    joblib.dump(
        {"model": model, "features": features, "classes": list(model.classes_)},
        model_path,
        compress=3,
    )
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    size_mb = model_path.stat().st_size / 1e6
    print(f"\nTraining time: {train_seconds} s")
    print("Attack vs benign on the test set:")
    print(f"  precision {precision:.4f}   recall {recall:.4f}   F1 {f1:.4f}")
    print("\nRecall per class:")
    for c, v in per_class_recall.items():
        print(f"  {c:<20} {v['recall']:.4f}   ({v['test_rows']:,} test rows)")
    print(f"\nSaved {model_path.name} ({size_mb:.1f} MB) and metrics.json in models/")
    if size_mb > 90:
        print("WARNING: the model is over 90 MB; GitHub rejects files above 100 MB.")


if __name__ == "__main__":
    main()
