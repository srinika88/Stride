"""
StrideIQ — Model Training
Trains and compares Random Forest, SVM, and MLP on your labeled feature
data, picks the best via cross-validation (not a single train/test split,
which is unreliable on a small dataset), and saves it with joblib.

Designed for the retrain loop you planned: collect -> train -> collect
more -> retrain -> evaluate -> improve. Run this fresh each time you've
added more labeled sessions — it doesn't assume a fixed dataset size.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from backend import config
from backend.database_manager import DatabaseManager
from ml.features import FEATURE_COLUMNS

logger = logging.getLogger("strideiq.ml.train")
logging.basicConfig(level=config.LOG_LEVEL)

MIN_SAMPLES_PER_CLASS = 3  # below this, cross-validation folds aren't meaningful

# Written by evaluate.py when it creates a held-out test split, so train.py
# can exclude exactly those rows and the two files' numbers stay honest
# relative to each other (evaluate.py previously assumed a split that
# train.py never actually respected — it always trained on 100% of data).
HOLDOUT_SPLIT_PATH = os.path.join(config.DATA_DIR, "holdout_split.json")


def _load_holdout_keys() -> set[tuple[int, float]]:
    if not os.path.exists(HOLDOUT_SPLIT_PATH):
        return set()
    with open(HOLDOUT_SPLIT_PATH) as f:
        keys = json.load(f)["test_keys"]
    return {(k[0], k[1]) for k in keys}


def load_training_data(db: DatabaseManager, exclude_holdout: bool = True):
    """
    Pulls labeled feature rows via the join already built in
    database_manager.get_training_data(). Returns (X, y, feature_names) —
    X has NaN where a column wasn't available for that row (e.g. waist
    metrics on a feet-only session), which get imputed below, not silently
    dropped as whole rows.

    If exclude_holdout is True (the default) and evaluate.py has already
    written a holdout split (data/holdout_split.json), those (run_id,
    window_start) rows are excluded from training — so the held-out set
    evaluate.py reports against was never seen during fitting. Pass False
    only if you deliberately want to train on everything (e.g. no holdout
    exists yet, or you're producing the final "ship it" model after
    evaluation is done and want every labeled sample in the mix).
    """
    rows = db.get_training_data()
    if not rows:
        raise ValueError(
            "No labeled training data found. This needs real labeled sessions "
            "(via the MediaPipe labeling pipeline) before it can train anything — "
            "running this against an empty database would only produce a model "
            "that's memorized nothing, which isn't a real model."
        )

    rows = [dict(r) for r in rows]

    if exclude_holdout:
        holdout_keys = _load_holdout_keys()
        if holdout_keys:
            before = len(rows)
            rows = [r for r in rows if (r["run_id"], r["window_start"]) not in holdout_keys]
            logger.info(
                f"Excluded {before - len(rows)} held-out rows (from {HOLDOUT_SPLIT_PATH}) "
                f"— training on the remaining {len(rows)}."
            )

    X = np.array([[r.get(col) for col in FEATURE_COLUMNS] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows])

    logger.info(f"Loaded {len(rows)} labeled windows across classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, FEATURE_COLUMNS


def _check_class_balance(y):
    classes, counts = np.unique(y, return_counts=True)
    sparse = classes[counts < MIN_SAMPLES_PER_CLASS]
    if len(sparse) > 0:
        logger.warning(
            f"Classes with fewer than {MIN_SAMPLES_PER_CLASS} samples: {list(sparse)} "
            f"— cross-validation results for these will be unreliable. Prioritize "
            f"collecting more sessions covering these labels before trusting this model."
        )


def build_candidates():
    """
    One reasonable, not-wildly-overfit config per model type. These are
    sane defaults for a small dataset, not exhaustively tuned — that's
    what hyperparameter_search.py is for, once there's enough data to
    make tuning meaningful rather than noise-chasing.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        ),
        "svm": SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=2000,
            early_stopping= False, random_state=42,
        ),
    }


def train_and_select_best(X, y, feature_names):
    """
    Cross-validates every candidate in config.CANDIDATE_MODELS, picks the
    best by mean CV accuracy, then refits it on the FULL dataset (standard
    practice: CV is for model selection, the final model uses all
    available data once you've picked which one to trust).
    """
    _check_class_balance(y)

    n_splits = min(5, min(np.unique(y, return_counts=True)[1]))
    if n_splits < 2:
        raise ValueError(
            "At least one class has fewer than 2 samples — cannot cross-validate. "
            "Collect more sessions for the underrepresented label(s) before training."
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    candidates = build_candidates()
    results = {}

    # Imputer + scaler are fit INSIDE each CV fold via Pipeline, not once on
    # the whole dataset beforehand. Fitting them on all of X first (the
    # previous approach) leaks each fold's held-out rows into the median/
    # mean/std used to preprocess it — cross_val_score's fold-by-fold score
    # is only an honest estimate if every step that "learns" from data,
    # including imputation and scaling, is refit per fold.
    for name in config.CANDIDATE_MODELS:
        if name not in candidates:
            logger.warning(f"'{name}' in config.CANDIDATE_MODELS has no implementation, skipping")
            continue

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", candidates[name]),
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
        results[name] = {
            "mean_f1_macro": float(np.mean(scores)),
            "std_f1_macro": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        }
        logger.info(f"{name}: F1-macro = {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

    if not results:
        raise ValueError("No candidate models were trainable — check config.CANDIDATE_MODELS")

    best_name = max(results, key=lambda k: results[k]["mean_f1_macro"])

    # Refit the winning pipeline on the FULL dataset for the final artifact —
    # CV above was for model selection only, standard practice once a
    # winner is picked. The imputer/scaler here are fit fresh one last time
    # on all available data, same as before, just no longer leaking into
    # the CV scores that chose this model.
    final_imputer = SimpleImputer(strategy="median")
    final_scaler = StandardScaler()
    X_imputed = final_imputer.fit_transform(X)
    X_scaled = final_scaler.fit_transform(X_imputed)

    best_model = candidates[best_name]
    best_model.fit(X_scaled, y)

    logger.info(f"Best model: {best_name} (F1-macro = {results[best_name]['mean_f1_macro']:.3f})")

    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(feature_names, best_model.feature_importances_.tolist()))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: -x[1])
        logger.info(f"Top features: {sorted_importance[:5]}")

    return {
        "model": best_model,
        "model_name": best_name,
        "imputer": final_imputer,
        "scaler": final_scaler,
        "feature_names": feature_names,
        "cv_results": results,
        "feature_importance": feature_importance,
    }


def save_model(bundle: dict, version: str = None):
    """
    Saves model + imputer + scaler + feature names TOGETHER in one bundle —
    a model alone is useless without knowing which columns/order it expects
    and how they were scaled. Also writes a JSON metadata file alongside
    it, so model_version in the predictions table always has a traceable
    record of what that version actually was.
    """
    version = version or config.MODEL_VERSION
    Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)

    model_path = Path(config.MODEL_DIR) / f"model_{version}.joblib"
    joblib.dump(bundle, model_path)

    metadata = {
        "version": version,
        "model_name": bundle["model_name"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cv_results": bundle["cv_results"],
        "feature_importance": bundle["feature_importance"],
        "feature_names": bundle["feature_names"],
    }
    meta_path = Path(config.MODEL_DIR) / f"model_{version}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save as the "current" model, what predict.py loads by default
    current_path = Path(config.MODEL_PATH)
    joblib.dump(bundle, current_path)

    logger.info(f"Saved model to {model_path} and {current_path}")
    return model_path


if __name__ == "__main__":
    db = DatabaseManager()
    X, y, feature_names = load_training_data(db)
    bundle = train_and_select_best(X, y, feature_names)
    save_model(bundle)

    print(f"\nBest model: {bundle['model_name']}")
    for name, res in bundle["cv_results"].items():
        print(f"  {name}: F1-macro = {res['mean_f1_macro']:.3f} (+/- {res['std_f1_macro']:.3f})") 