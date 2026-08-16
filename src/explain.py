"""
explain.py
==========
SHAP-based model interpretability.

Generates:
  - SHAP summary plot (bar + beeswarm)
  - Mean |SHAP| values per feature (logged to MLflow)

Usage:
    python src/explain.py

Requires the model and preprocessor to exist (run train.py first).
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import FEATURES, TARGET, load_preprocessor, split_data
from src.data_validation import validate_dataset

DATA_PATH = os.path.join("data", "raw", "students.csv")
MODEL_PATH = os.path.join("models", "best_model.joblib")
SHAP_PLOT_PATH = os.path.join("models", "shap_summary.png")
SHAP_BAR_PATH = os.path.join("models", "shap_bar.png")


def compute_shap(
    model,
    X_transformed: np.ndarray,
    feature_names: list,
    max_samples: int = 200,
):
    """
    Compute SHAP values using the appropriate explainer.

    For tree-based models (Random Forest) → TreeExplainer (fast, exact).
    For linear models → LinearExplainer.
    Fallback → KernelExplainer on a sample.

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
    explainer   : the SHAP explainer object
    """
    model_type = type(model).__name__

    # Use a subsample for speed
    n = min(max_samples, X_transformed.shape[0])
    X_sample = X_transformed[:n]

    if "Forest" in model_type or "Tree" in model_type or "Gradient" in model_type:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif "Linear" in model_type or "Ridge" in model_type or "Lasso" in model_type:
        explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_sample)
    else:
        background = shap.sample(X_transformed, 50)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample)

    return shap_values, explainer, X_sample


def plot_shap_summary(shap_values, X_sample, feature_names, save_path: str) -> None:
    """Beeswarm summary plot."""
    fig = plt.figure(figsize=(9, 5))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_size=(9, 5),
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"[explain] SHAP beeswarm saved -> {save_path}")


def plot_shap_bar(shap_values, feature_names, save_path: str) -> None:
    """Mean absolute SHAP bar chart."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in idx],
        [mean_abs[i] for i in idx],
        color="steelblue",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"[explain] SHAP bar chart saved -> {save_path}")


def run_explanation():
    """Load model and data, compute and save SHAP plots."""
    if not os.path.exists(MODEL_PATH):
        print(f"[explain] ERROR: {MODEL_PATH} not found. Run train.py first.")
        sys.exit(1)

    print("[explain] Loading model and data ...")
    model = joblib.load(MODEL_PATH)
    preprocessor = load_preprocessor()

    df = pd.read_csv(DATA_PATH)
    df = validate_dataset(df)

    from src.preprocess import split_data
    _, X_test, _, _ = split_data(df)
    X_test_t = preprocessor.transform(X_test)

    print(f"[explain] Computing SHAP values for {type(model).__name__} ...")
    shap_values, _, X_sample = compute_shap(model, X_test_t, FEATURES)

    plot_shap_summary(shap_values, X_sample, FEATURES, SHAP_PLOT_PATH)
    plot_shap_bar(shap_values, FEATURES, SHAP_BAR_PATH)

    mean_abs = np.abs(shap_values).mean(axis=0)
    print("\n[explain] Mean |SHAP| per feature:")
    for feat, val in sorted(zip(FEATURES, mean_abs), key=lambda x: -x[1]):
        print(f"  {feat:<40} {val:.4f}")

    return shap_values


if __name__ == "__main__":
    run_explanation()
