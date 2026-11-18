"""
Train & Save the Final Model (for the web app)
------------------------------------------------
classify.py compares six models but does not save any of them. This script
trains ONE model (Gradient Boosting by default, our best) and saves everything
the web app needs into model.joblib:
  - the trained model and the scaler
  - the list of feature names (so the app builds features in the same order)
  - average feature values for human vs AI texts (used for the explanation table)
  - a few held-out test examples (texts the model never saw) for the demo

It uses the SAME train/test split as classify.py (same test_size and
random_state), so the accuracy printed here matches your classify.py results.

Usage:
    python train_final_model.py --input features.csv --dataset dataset.csv
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser(description="Train and save the final model for the web app.")
    parser.add_argument("--input", required=True, help="features.csv from feature_engineering.py")
    parser.add_argument("--dataset", default=None, help="dataset.csv (optional) - used to save demo examples")
    parser.add_argument("--output", default="model.joblib", help="Where to save the model (default: model.joblib)")
    parser.add_argument("--model", choices=["gb", "svm"], default="gb", help="gb = Gradient Boosting (default), svm = SVM")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    groups = df["qid"] if "qid" in df.columns else None
    X = df.drop(columns=[c for c in ["label", "qid"] if c in df.columns])
    y = df["label"]

    # Same split as classify.py
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=args.test_size, random_state=args.random_state, stratify=y
        )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if args.model == "svm":
        model = SVC(kernel="rbf", C=3, gamma=0.01, probability=True, random_state=args.random_state)
        model_name = "SVM"
    else:
        model = HistGradientBoostingClassifier(random_state=args.random_state)
        model_name = "Gradient Boosting"
    model.fit(X_train_scaled, y_train)

    probs = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
    auc = roc_auc_score(y_test, probs)
    print(f"{model_name}: test accuracy {acc:.3f}, ROC-AUC {auc:.3f} "
          f"({len(X_train)} train / {len(X_test)} test samples)")

    # Average feature value per class (raw, unscaled) - used for the explanation table in the app
    class_means = {
        "human": X_train[y_train == 0].mean().to_dict(),
        "ai": X_train[y_train == 1].mean().to_dict(),
    }

    # A few held-out examples the model got right, for a smooth live demo
    examples = []
    if args.dataset:
        ds = pd.read_csv(args.dataset)
        if len(ds) == len(df):
            test_part = ds.iloc[test_idx].reset_index(drop=True)
            test_part["prob_ai"] = probs
            test_part["true_label"] = y_test.to_numpy()
            humans = test_part[(test_part["true_label"] == 0) & (test_part["prob_ai"] < 0.2)].head(3)
            ais = test_part[(test_part["true_label"] == 1) & (test_part["prob_ai"] > 0.8)].head(3)
            for name, part in [("Human", humans), ("AI", ais)]:
                for _, row in part.iterrows():
                    examples.append({"label_name": name, "text": row["text"], "prob_ai": float(row["prob_ai"])})
        else:
            print("Warning: dataset.csv and features.csv have different lengths - skipping examples.")

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "model_name": model_name,
        "uncertain_low": 0.40,
        "uncertain_high": 0.60,
        "test_accuracy": float(acc),
        "test_auc": float(auc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_means": class_means,
        "examples": examples,
    }
    joblib.dump(bundle, args.output)
    print(f"Saved {args.output} (with {len(examples)} demo examples)")


if __name__ == "__main__":
    main()