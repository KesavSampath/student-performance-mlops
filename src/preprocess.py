"""
preprocess.py
=============
Reusable preprocessing for the student performance dataset.

Design decisions:
  - Uses a Scikit-learn Pipeline (imputer → scaler) so the EXACT same
    transformation is applied at training time and inference time.
  - Pipeline is saved alongside the model to prevent data leakage.
  - Fixed random seed ensures reproducible train/test splits.

Usage:
    from src.preprocess import build_pipeline, split_data, FEATURES, TARGET

    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_pipeline()
    X_train_t = pipeline.fit_transform(X_train)
    X_test_t  = pipeline.transform(X_test)
"""

import joblib
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Feature / Target definitions (single source of truth)
# ---------------------------------------------------------------------------
FEATURES = [
    "study_hours",
    "attendance_percentage",
    "previous_exam_score",
    "assignment_completion_percentage",
    "sleep_hours",
    "extracurricular_hours",
]

TARGET = "final_exam_score"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
PREPROCESSOR_PATH = os.path.join("models", "preprocessor.joblib")


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """
    Build the preprocessing pipeline.

    Steps:
      1. SimpleImputer  — fills missing values with column median
      2. StandardScaler — standardises to zero mean, unit variance

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified-free train/test split.

    Parameters
    ----------
    df          : full validated dataset
    test_size   : fraction for test set (default 0.20)
    random_state: reproducibility seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Convenience: save / load preprocessor
# ---------------------------------------------------------------------------
def save_preprocessor(pipeline: Pipeline, path: str = PREPROCESSOR_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"[preprocess] Preprocessor saved -> {path}")


def load_preprocessor(path: str = PREPROCESSOR_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessor not found at {path}. Run training first.")
    return joblib.load(path)
