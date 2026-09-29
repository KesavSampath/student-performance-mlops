"""
train.py
========
Automated MLOps training pipeline.

Pipeline stages:
  1. Load & validate data      (Pandera)
  2. Preprocess                (Scikit-learn Pipeline)
  3. Train models              (Linear Regression, Random Forest)
  4. Evaluate all models       (MAE, RMSE, R²)
  5. Select best model         (highest R²)
  6. Log to MLflow             (params, metrics, artifacts)
  7. Apply quality gate        (R² ≥ 0.75, MAE ≤ 8.0)
  8. Save model artifact       (Joblib)
  9. Save metadata             (JSON)

Usage:
    python src/train.py
    python src/train.py --data data/raw/students.csv
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_validation import validate_dataset
from src.preprocess import (
    FEATURES, TARGET, RANDOM_SEED, TEST_SIZE,
    build_pipeline, split_data, save_preprocessor
)
from src.evaluate import compute_metrics, quality_gate

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "raw", "students.csv")
MODELS_DIR = "models"
MLFLOW_EXPERIMENT = "student-performance-prediction"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # local SQLite (MLflow 3.x requires DB backend)

# Model definitions: (name, instance)
CANDIDATE_MODELS = [
    (
        "LinearRegression",
        LinearRegression(),
    ),
    (
        "RandomForestRegressor",
        RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_predictions(y_test, y_pred, model_name: str, save_path: str) -> None:
    """Actual vs Predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="white", linewidths=0.3, s=40)
    lims = [max(0, min(y_test.min(), y_pred.min()) - 2),
            min(100, max(y_test.max(), y_pred.max()) + 2)]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_residuals(y_test, y_pred, model_name: str, save_path: str) -> None:
    """Residual plot."""
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=30)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"Residuals — {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_feature_importance(model, feature_names, model_name: str, save_path: str) -> None:
    """Feature importance (Random Forest) or coefficients (Linear Regression)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        label = "Feature Importance"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        label = "|Coefficient|"
    else:
        plt.close(fig)
        return

    indices = np.argsort(importances)[::-1]
    ax.barh(
        [feature_names[i] for i in indices],
        [importances[i] for i in indices],
        color="steelblue",
    )
    ax.set_xlabel(label)
    ax.set_title(f"Feature Importance — {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(data_path: str = DATA_PATH) -> dict:
    """
    Execute the full training pipeline.

    Returns
    -------
    dict with keys: best_model_name, metrics, model_path
    """
    print("\n" + "=" * 60)
    print("  STUDENT PERFORMANCE MLOps — Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ────────────────────────────────────────────
    print(f"\n[train] Loading data from {data_path} ...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run generate_data.py first.")

    df = pd.read_csv(data_path)
    print(f"[train] Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── 2. Validate ──────────────────────────────────────────────
    print("\n[train] Validating data with Pandera ...")
    df = validate_dataset(df)

    # ── 3. Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"[train] Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── 4. Preprocess ─────────────────────────────────────────────
    print("[train] Fitting preprocessor (imputer + scaler) ...")
    preprocessor = build_pipeline()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    save_preprocessor(preprocessor)

    # ── 5. Configure MLflow ──────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    os.makedirs(MODELS_DIR, exist_ok=True)
    tmp_artifacts = os.path.join(MODELS_DIR, "tmp_plots")
    os.makedirs(tmp_artifacts, exist_ok=True)

    # ── 6. Train & evaluate each candidate ─────────────────────
    results = {}
    for model_name, model in CANDIDATE_MODELS:
        print(f"\n[train] ── Training: {model_name} ──")

        with mlflow.start_run(run_name=model_name):

            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_test", len(X_test))

            params = model.get_params()
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Train
            model.fit(X_train_t, y_train)

            # Predict (clip to valid score range)
            y_pred = model.predict(X_test_t).clip(0, 100)

            # Metrics
            metrics = compute_metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)

            print(f"  MAE  = {metrics['mae']:.4f}")
            print(f"  RMSE = {metrics['rmse']:.4f}")
            print(f"  R²   = {metrics['r2']:.4f}")

            # Plots
            pred_plot = os.path.join(tmp_artifacts, f"{model_name}_predictions.png")
            resid_plot = os.path.join(tmp_artifacts, f"{model_name}_residuals.png")
            feat_plot = os.path.join(tmp_artifacts, f"{model_name}_features.png")
            plot_predictions(y_test, y_pred, model_name, pred_plot)
            plot_residuals(y_test, y_pred, model_name, resid_plot)
            plot_feature_importance(model, FEATURES, model_name, feat_plot)
            mlflow.log_artifact(pred_plot, artifact_path="plots")
            mlflow.log_artifact(resid_plot, artifact_path="plots")
            if os.path.exists(feat_plot):
                mlflow.log_artifact(feat_plot, artifact_path="plots")

            # Log model
            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="cloudpickle")

            run_id = mlflow.active_run().info.run_id
            results[model_name] = {
                "model": model,
                "metrics": metrics,
                "run_id": run_id,
            }

    # ── 7. Select best model (highest R²) ───────────────────────
    best_name = max(results, key=lambda k: results[k]["metrics"]["r2"])
    best = results[best_name]
    best_metrics = best["metrics"]

    print(f"\n[train] Best model: {best_name}")
    print(f"  R² = {best_metrics['r2']:.4f}  MAE = {best_metrics['mae']:.4f}  RMSE = {best_metrics['rmse']:.4f}")

    # ── 8. Quality gate ──────────────────────────────────────────
    quality_gate(best_metrics, raise_on_fail=True)

    # ── 9. Save best model artifact ──────────────────────────────
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best["model"], model_path)
    print(f"[train] Best model saved -> {model_path}")

    # Save metadata (used by FastAPI and Streamlit)
    metadata = {
        "model_name": best_name,
        "model_version": "1.0.0",
        "mlflow_run_id": best["run_id"],
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "features": FEATURES,
        "target": TARGET,
        "metrics": best_metrics,
        "quality_gate": {
            "min_r2": 0.75,
            "max_mae": 8.0,
            "passed": True,
        },
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] Metadata saved -> {metadata_path}")

    # Log best model metadata to its MLflow run
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_artifact(metadata_path, artifact_path="metadata")
        mlflow.log_artifact(model_path, artifact_path="best_model")

    print("\n[train] ✓ Training pipeline complete.")
    print("=" * 60 + "\n")

    return {
        "best_model_name": best_name,
        "metrics": best_metrics,
        "model_path": model_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student performance models")
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help="Path to the raw CSV dataset",
    )
    args = parser.parse_args()

    result = train(data_path=args.data)
    print(f"Result: {json.dumps(result, indent=2)}")
