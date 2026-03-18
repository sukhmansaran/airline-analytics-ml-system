import os
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

import warnings
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores"
)

import json
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from training.preprocess import preprocess

MODELS_DIR = "models"
N_TRIALS = 30
CV_FOLDS = 5
_CB_TEMP = os.path.join(os.environ.get("TEMP", "/tmp"), "catboost")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(_CB_TEMP, exist_ok=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _xgb_objective(trial, x, y, spw):
    model = XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=spw,
        max_depth=        trial.suggest_int("max_depth", 3, 10),
        n_estimators=     trial.suggest_int("n_estimators", 100, 500, step=50),
        learning_rate=    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=        trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree= trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight= trial.suggest_int("min_child_weight", 1, 10),
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    return cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()


def _lgbm_objective(trial, x, y, spw):
    model = LGBMClassifier(
        verbose=-1,
        scale_pos_weight=  spw,
        n_estimators=      trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=         trial.suggest_int("max_depth", 3, 12),
        learning_rate=     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=        trial.suggest_int("num_leaves", 20, 150),
        subsample=         trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    return cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()


def _catboost_objective(trial, x, y):
    # Unique train_dir per trial — avoids parallel file collisions
    train_dir = os.path.join(_CB_TEMP, f"trial_{trial.number}")
    os.makedirs(train_dir, exist_ok=True)
    model = CatBoostClassifier(
        loss_function="Logloss",
        auto_class_weights="Balanced",
        verbose=0,
        train_dir=train_dir,
        iterations=          trial.suggest_int("iterations", 100, 500, step=50),
        depth=               trial.suggest_int("depth", 3, 10),
        learning_rate=       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg=         trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        bagging_temperature= trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength=     trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    # n_jobs=1 — CatBoost writes to disk; parallel CV causes file collisions
    return cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=1).mean()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_all(x_train, y_train, x_test, y_test, n_trials: int = N_TRIALS):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = round(neg / pos) if pos else 1

    results = {}

    print("Tuning XGBoost...")
    study_xg = optuna.create_study(direction="maximize")
    study_xg.optimize(lambda t: _xgb_objective(t, x_train, y_train, spw), n_trials=n_trials)
    p = study_xg.best_params
    best_xg = XGBClassifier(
        tree_method="hist", eval_metric="logloss", scale_pos_weight=spw,
        max_depth=p["max_depth"], n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], min_child_weight=p["min_child_weight"],
    )
    best_xg.fit(x_train, y_train)
    results["XGBoost"] = (best_xg, best_xg.predict(x_test))
    print(f"  best ROC-AUC (CV): {study_xg.best_value:.4f}")

    print("Tuning LightGBM...")
    study_lgbm = optuna.create_study(direction="maximize")
    study_lgbm.optimize(lambda t: _lgbm_objective(t, x_train, y_train, spw), n_trials=n_trials)
    p = study_lgbm.best_params
    best_lgbm = LGBMClassifier(
        verbose=-1, scale_pos_weight=spw,
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        min_child_samples=p["min_child_samples"],
    )
    best_lgbm.fit(x_train, y_train)
    results["LightGBM"] = (best_lgbm, best_lgbm.predict(x_test))
    print(f"  best ROC-AUC (CV): {study_lgbm.best_value:.4f}")

    print("Tuning CatBoost...")
    study_cb = optuna.create_study(direction="maximize")
    study_cb.optimize(lambda t: _catboost_objective(t, x_train, y_train), n_trials=n_trials)
    p = study_cb.best_params
    best_cb = CatBoostClassifier(
        loss_function="Logloss", auto_class_weights="Balanced", verbose=0,
        train_dir=os.path.join(_CB_TEMP, "final"),
        iterations=p["iterations"], depth=p["depth"], learning_rate=p["learning_rate"],
        l2_leaf_reg=p["l2_leaf_reg"], bagging_temperature=p["bagging_temperature"],
        random_strength=p["random_strength"],
    )
    best_cb.fit(x_train, y_train)
    results["CatBoost"] = (best_cb, best_cb.predict(x_test))
    print(f"  best ROC-AUC (CV): {study_cb.best_value:.4f}")

    # Use predict_proba for ROC-AUC — class labels give incorrect AUC
    rows = []
    for name, (model, _) in results.items():
        pred       = model.predict(x_test)
        pred_proba = model.predict_proba(x_test)[:, 1]
        rows.append({
            "model":     name,
            "accuracy":  accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall":    recall_score(y_test, pred),
            "f1":        f1_score(y_test, pred),
            "roc_auc":   roc_auc_score(y_test, pred_proba),
        })

    scores_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\n" + scores_df.to_string(index=False))

    best_name = scores_df["model"].iloc[0]
    return results[best_name][0], best_name, scores_df


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str = "data/customer_booking.csv", n_trials: int = N_TRIALS):
    raw_df = pd.read_csv(data_path, encoding="ISO-8859-1")
    x, y, scaler, feature_cols, label_encoders = preprocess(raw_df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model, name, scores_df = train_all(x_train, y_train, x_test, y_test, n_trials=n_trials)

    joblib.dump(model, os.path.join(MODELS_DIR, "booking_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
    meta = {"all_features": feature_cols, "best_model": name}
    with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)

    scores_df.to_csv(os.path.join(MODELS_DIR, "scores.csv"), index=False)
    print(f"\nBest model: {name} — saved to {MODELS_DIR}/")
    return model, scaler, feature_cols


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the booking completion model.")
    parser.add_argument("--data", default="data/customer_booking.csv", help="Path to CSV")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Optuna trials per model")
    args = parser.parse_args()
    run_pipeline(data_path=args.data, n_trials=args.trials)
