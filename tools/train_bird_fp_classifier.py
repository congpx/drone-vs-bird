from pathlib import Path
import argparse
import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "conf",
    "area",
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity",
    "compactness",
    "rectangularity",
    "eccentricity",
    "vertex_count",
    "bbox_w",
    "bbox_h",
]

def build_model(model_name: str):
    if model_name == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42
            ))
        ])
    elif model_name == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
    else:
        raise ValueError(model_name)


def main(args):
    df = pd.read_csv(args.input_csv)

    # Chỉ dùng 2 nhóm:
    # positive = bird_fp
    # negative = true_drone
    df = df[(df["label_bird_fp"] == 1) | (df["label_true_drone"] == 1)].copy()
    df["target"] = df["label_bird_fp"].astype(int)

    X = df[FEATURES].copy()
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = build_model(args.model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred).tolist()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / f"bird_fp_filter_{args.model_type}.joblib"
    joblib.dump(model, model_path)

    summary = {
        "model_type": args.model_type,
        "features": FEATURES,
        "samples_total": int(len(df)),
        "positive_bird_fp": int(df["target"].sum()),
        "negative_true_drone": int((df["target"] == 0).sum()),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "bird_fp_binary_metrics": {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        },
        "confusion_matrix": cm,
        "classification_report": report,
    }

    (outdir / f"bird_fp_filter_{args.model_type}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("[DONE] Saved:")
    print(" -", model_path)
    print(" -", outdir / f"bird_fp_filter_{args.model_type}_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model-type", choices=["logreg", "rf"], default="rf")
    args = ap.parse_args()
    main(args)