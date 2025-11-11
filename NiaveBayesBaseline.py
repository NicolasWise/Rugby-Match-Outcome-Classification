# baseline_naive_bayes.py
# Baseline classifier for Tier 1 rugby test results using Naïve Bayes (CategoricalNB)
# ------------------------------------------------------------
# Requirements:
#   pip install scikit-learn pandas numpy
# ------------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main(csv_path: str = "results.csv", test_size: float = 0.2, random_state: int = 42):
    # --------------------------
    # 1) Load data
    # --------------------------
    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find dataset at '{csv_path}'. "
              f"Place results.csv next to this script or pass a path argument.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Basic sanity check for required columns
    required = ["home_team", "away_team", "competition", "country",
                "neutral", "world_cup", "home_score", "away_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: The dataset is missing required columns: {missing}")
        sys.exit(1)

    # --------------------------
    # 2) Create target variable
    # --------------------------
    df["result"] = np.where(df["home_score"] > df["away_score"], "HomeWin",
                     np.where(df["home_score"] < df["away_score"], "HomeLoss", "Draw"))

    # --------------------------
    # 3) Select baseline features
    # --------------------------
    X = df[["home_team", "away_team", "competition", "country", "neutral", "world_cup"]].copy()
    y = df["result"].copy()

    # Handle missing competition (there are a few NaNs)
    X["competition"] = X["competition"].fillna("Unknown")

    # Ensure booleans are booleans
    for b in ["neutral", "world_cup"]:
        if X[b].dtype != bool:
            X[b] = X[b].astype(bool)

    # --------------------------
    # 4) Train / test split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --------------------------
    # 5) Preprocessing + model
    # --------------------------
    cat_features  = ["home_team", "away_team", "competition", "country"]
    bool_features = ["neutral", "world_cup"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("bool", "passthrough", bool_features),
        ],
        remainder="drop"
    )

    nb = CategoricalNB(alpha=1.0)  # Laplace smoothing

    pipeline = Pipeline(steps=[
        ("prep", preprocess),
        ("clf", nb),
    ])

    # --------------------------
    # 6) Tiny hyperparam search
    # --------------------------
    param_grid = {"clf__alpha": [0.5, 1.0, 2.0]}
    grid = GridSearchCV(
        pipeline, param_grid=param_grid, scoring="accuracy", cv=5, n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    print(f"[Baseline NB] Best params: {grid.best_params_}")

    # --------------------------
    # 7) Evaluation (with fixed label order)
    # --------------------------
    labels = ["Draw", "HomeLoss", "HomeWin"]  # <-- explicit, consistent order

    preds = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    print("Classification Report (macro-averaged metrics shown; ordered labels):")
    print(classification_report(
        y_test, preds, labels=labels, target_names=labels, digits=3, zero_division=0
    ))

    print("Confusion Matrix [rows = true, cols = pred] (ordered labels):")
    cm = confusion_matrix(y_test, preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())

    # Optional: preview predicted probabilities (aligned to sklearn's class order)
    # prob_cols = list(best_model.named_steps["clf"].classes_)  # typically alphabetical
    # preview = pd.DataFrame(proba, columns=prob_cols)
    # print("\nSample predicted probabilities:")
    # print(preview.head())

    # --------------------------
    # 8) (Optional) Save model
    # --------------------------
    # from joblib import dump
    # dump(best_model, "baseline_nb.joblib")
    # print("Saved model to baseline_nb.joblib")

if __name__ == "__main__":
    # Allow optional path argument: python baseline_naive_bayes.py [path_to_csv]
    csv = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    main(csv_path=csv)
