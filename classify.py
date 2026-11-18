"""
Model Training & Comparison — AI-generated vs Human Text Detection
---------------------------------------------------------------------
Takes the features.csv produced by feature_engineering.py and trains/
compares baseline classifiers (Naive Bayes, Logistic Regression, SVM)
against ensemble methods (Bagging, Random Forest) and boosting methods
(AdaBoost, Gradient Boosting).

Expected input: features.csv with numeric feature columns + a 'label'
column (0 = human, 1 = AI-generated).

Usage:
    python classify.py --input features.csv
"""

import argparse #Reading command-line options like --input, --tune

import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, classification_report,
)


def load_data(path: str, label_col: str = "label"):
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Expected a '{label_col}' column in {path}, found: {list(df.columns)}")
    groups = df["qid"] if "qid" in df.columns else None
    X = df.drop(columns=[c for c in [label_col, "qid"] if c in df.columns])
    y = df[label_col]
    return X, y, groups


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    # Some models (e.g. SVC without probability=True) won't have predict_proba
    try:
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
    except AttributeError:
        auc = float("nan")

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def uncertainty_report(model, X_test, y_test, low, high):
    """
    The "Flag Uncertain" step: instead of forcing every prediction into
    human/AI, mark predictions whose AI-probability falls in the grey
    zone [low, high] as 'Uncertain'. Returns the share flagged, the
    accuracy on the remaining confident predictions, and per-sample verdicts.
    """
    probs = model.predict_proba(X_test)[:, 1]
    uncertain = (probs >= low) & (probs <= high)
    confident = ~uncertain
    y_arr = y_test.to_numpy()
    if confident.any():
        conf_acc = accuracy_score(y_arr[confident], (probs[confident] >= 0.5).astype(int))
    else:
        conf_acc = float("nan")
    verdicts = ["Uncertain" if u else ("AI-generated" if p >= 0.5 else "Human")
                for u, p in zip(uncertain, probs)]
    return uncertain.mean(), conf_acc, probs, verdicts


def main():
    parser = argparse.ArgumentParser(description="Train and compare classifiers for AI-text detection.")
    parser.add_argument("--input", required=True, help="Path to features.csv (from feature_engineering.py)")
    parser.add_argument("--label_col", default="label", help="Name of the label column (default: 'label')")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--drop", nargs="*", default=[], help="Feature columns to exclude, e.g. --drop num_words num_sentences")
    parser.add_argument("--out_prefix", default="", help="Prefix for the saved result files, so runs don't overwrite each other")
    parser.add_argument("--tune", action="store_true", help="Tune SVM / Random Forest / Gradient Boosting settings on the training data only (slower)")
    parser.add_argument("--uncertain_low", type=float, default=0.40, help="Lower bound of the 'uncertain' probability zone")
    parser.add_argument("--uncertain_high", type=float, default=0.60, help="Upper bound of the 'uncertain' probability zone")
    args = parser.parse_args()

    X, y, groups = load_data(args.input, args.label_col)
    if args.drop:
        missing = [c for c in args.drop if c not in X.columns]
        if missing:
            raise ValueError(f"--drop columns not found: {missing}. Available: {list(X.columns)}")
        X = X.drop(columns=args.drop)
        print(f"Dropped features: {args.drop}")
    print(f"Using features: {list(X.columns)}\n")

    if groups is not None:
        # Keep all answers to the same question on the same side of the split,
        # so the model can't "recognise" a question it already saw in training.
        splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        print("Split by question (no question appears in both train and test).")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        groups_train = None
        print("No 'qid' column found: using a plain random split.")

    # Naive Bayes and SVM are distance/probability based, so features
    # need to be on a comparable scale. Tree-based ensembles don't need
    # this, but scaling doesn't hurt them either, so we apply it to all
    # for a fair, simple pipeline.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Naive Bayes": GaussianNB(),
        # Linear baseline: a weighted sum of the features passed through a sigmoid,
        # giving the probability that a text is AI-generated.
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=args.random_state),
        "SVM": SVC(kernel="rbf", probability=True, random_state=args.random_state),
        "Bagging": BaggingClassifier(n_estimators=50, random_state=args.random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=args.random_state, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=args.random_state),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=args.random_state),
    }

    if args.tune:
        # Settings are chosen using ONLY the training data (never the test set).
        grids = {
            "SVM": {"C": [1, 3, 10], "gamma": ["scale", 0.01, 0.03]},
            "Random Forest": {"max_depth": [None, 20], "min_samples_leaf": [1, 3]},
            "Gradient Boosting": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
        }
        cv = GroupKFold(n_splits=3) if groups_train is not None else 3
        for name, grid in grids.items():
            print(f"Tuning {name} ...")
            search = GridSearchCV(models[name], grid, cv=cv, scoring="accuracy", n_jobs=-1)
            search.fit(X_train_scaled, y_train, **({"groups": groups_train} if groups_train is not None else {}))
            print(f"  best settings: {search.best_params_}")
            models[name] = search.best_estimator_

    results = []
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(name, model, X_test_scaled, y_test)
        results.append(metrics)

        print(f"--- {name} ---")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1-score:  {metrics['f1']:.3f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
        print(f"Confusion matrix: {metrics['confusion_matrix']}")

        flagged, conf_acc, _, _ = uncertainty_report(
            model, X_test_scaled, y_test, args.uncertain_low, args.uncertain_high
        )
        metrics["flagged_uncertain"] = flagged
        metrics["accuracy_confident_only"] = conf_acc
        print(f"Flagged uncertain ({args.uncertain_low:.0%}-{args.uncertain_high:.0%} zone): {flagged:.1%} of samples")
        print(f"Accuracy on confident predictions only: {conf_acc:.3f}")
        print()

    # Summary comparison table
    summary_df = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "flagged_uncertain", "accuracy_confident_only"]]
    print("=== Model Comparison ===")
    print(summary_df.to_string(index=False))

    # Feature importance from Random Forest — useful for your "why it
    # helps" / interpretability angle in the review
    rf = models["Random Forest"]
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n=== Random Forest Feature Importance (top 15) ===")
    print(importance_df.head(15).to_string(index=False))

    # Per-sample verdicts from Random Forest, including the Uncertain flag
    _, _, rf_probs, rf_verdicts = uncertainty_report(
        rf, X_test_scaled, y_test, args.uncertain_low, args.uncertain_high
    )
    pred_df = pd.DataFrame({
        "true_label": y_test.to_numpy(),
        "ai_probability": rf_probs,
        "verdict": rf_verdicts,
    })
    pred_df.to_csv(args.out_prefix + "predictions_with_uncertainty.csv", index=False)

    summary_df.to_csv(args.out_prefix + "model_comparison_results.csv", index=False)
    importance_df.to_csv(args.out_prefix + "feature_importance.csv", index=False)
    print(f"\nSaved {args.out_prefix}model_comparison_results.csv, {args.out_prefix}feature_importance.csv and {args.out_prefix}predictions_with_uncertainty.csv")


if __name__ == "__main__":
    main()