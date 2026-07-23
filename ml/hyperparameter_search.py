"""
StrideIQ — Hyperparameter Search
Grid search over the same three candidates train.py cross-validates
(Random Forest, SVM, MLP), for the tuning step train.py explicitly
defers to this file ("that's what hyperparameter_search.py is for").

Kept separate from train.py on purpose: train.py's job is "pick the
best model type and ship a bundle" every time you retrain, which needs
to be fast. Grid search is expensive and only worth re-running when
you've added a meaningful amount of new data — not on every retrain.

Usage:
    python -m ml.hyperparameter_search              # search all 3
    python -m ml.hyperparameter_search --model svm  # just one
"""

import argparse
import logging
import json
from pathlib import Path 
import joblib 
import time  

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from backend import config
from backend.database_manager import DatabaseManager
from ml.train import load_training_data

logger = logging.getLogger("strideiq.ml.hyperparameter_search")
logging.basicConfig(level=config.LOG_LEVEL)


# Grid choices are deliberately modest — this dataset will be small
# (self-collected + a handful of public sessions), and a huge grid on
# a small dataset mostly chases CV noise rather than finding anything
# real. Widen these once you have hundreds of labeled windows.
PARAM_GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [4, 8, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [
        "sqrt",
        "log2",
        None,
    ],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", 0.01, 0.1],
        "kernel": ["rbf", "linear"],
    },
    "mlp": {
        "hidden_layer_sizes": [(16,), (32, 16), (64, 32)], 
        "alpha": [1e-4, 1e-3, 1e-2],
    },
}

BASE_ESTIMATORS = {
    "random_forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "svm": SVC(probability=True, class_weight="balanced", random_state=42),
    "mlp": MLPClassifier(max_iter=2000, early_stopping=False, random_state=42),
}


def search(model_name: str, X, y):
    """
    Runs GridSearchCV for one model type. Returns the fitted
    GridSearchCV object — callers can pull .best_params_,
    .best_score_, and .best_estimator_ off it.
    """
    if model_name not in BASE_ESTIMATORS:
        raise ValueError(f"Unknown model '{model_name}', choose from {list(BASE_ESTIMATORS)}")

    n_splits = min(5, min(np.unique(y, return_counts=True)[1]))
    if n_splits < 2:
        raise ValueError(
            "At least one class has fewer than 2 samples — cannot cross-validate. "
            "Collect more sessions for the underrepresented label(s) first."
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = RandomizedSearchCV(
        estimator=BASE_ESTIMATORS[model_name],
        param_distributions=PARAM_GRIDS[model_name],
        n_iter=20,
        scoring="f1_macro",
        cv=cv, 
        random_state=42,
        n_jobs=-1,
        refit=True,
        ) 
    start = time.time()
    grid.fit(X, y)
    elapsed = time.time() - start
    logger.info(
    f"Search completed in {elapsed:.1f} seconds."
    )
    logger.info(
    f"Evaluated {len(grid.cv_results_['params'])} parameter combinations."
    )

    logger.info(
        f"{model_name}: best F1-macro = {grid.best_score_:.3f}, params = {grid.best_params_}"
    )
    return grid


def search_all(db: DatabaseManager = None, model_names: list = None) -> dict:
    """
    Runs the search across every requested model type (default: all
    three), on the same imputed/scaled features train.py uses, so
    results are directly comparable to train.py's cross-validation
    numbers.
    """
    db = db or DatabaseManager()
    X, y, feature_names = load_training_data(db)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))

    model_names = model_names or list(BASE_ESTIMATORS)
    results = {}
    for name in model_names:
        grid = search(name, X_scaled, y)
        results[name] = {
            "best_score": float(grid.best_score_),
            "best_params": grid.best_params_,
            "best_estimator": grid.best_estimator_, 
        }
    results_path = Path(config.MODEL_DIR) / "hyperparameter_results.json"

    with open(results_path, "w") as f:
     json.dump(results, f, indent=2)
    logger.info(f"Saved search results to {results_path}")
    return results 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StrideIQ hyperparameter search")
    parser.add_argument(
        "--model", choices=list(BASE_ESTIMATORS), default=None,
        help="Search just one model type; omit to search all three.",
    )
    args = parser.parse_args()

    model_names = [args.model] if args.model else None
    results = search_all(model_names=model_names)

    print("\nHyperparameter search results (F1-macro, cross-validated):")
    for name, res in sorted(results.items(), key=lambda kv: -kv[1]["best_score"]):
        print(f"  {name}: {res['best_score']:.3f} - {res['best_params']}")