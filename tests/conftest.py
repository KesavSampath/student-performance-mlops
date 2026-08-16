"""
tests/conftest.py
=================
Shared fixtures for all test modules.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import FEATURES, TARGET


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_df():
    """A small valid dataset with correct schema and ranges."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "study_hours": np.random.uniform(1, 10, n),
        "attendance_percentage": np.random.uniform(40, 100, n),
        "previous_exam_score": np.random.uniform(30, 95, n),
        "assignment_completion_percentage": np.random.uniform(40, 100, n),
        "sleep_hours": np.random.uniform(4, 9, n),
        "extracurricular_hours": np.random.uniform(0, 6, n),
        "final_exam_score": np.random.uniform(30, 95, n),
    })


@pytest.fixture
def invalid_range_df():
    """A dataset with out-of-range values."""
    df = pd.DataFrame({
        "study_hours": [5.0],
        "attendance_percentage": [150.0],   # INVALID: > 100
        "previous_exam_score": [70.0],
        "assignment_completion_percentage": [-5.0],  # INVALID: < 0
        "sleep_hours": [7.0],
        "extracurricular_hours": [2.0],
        "final_exam_score": [65.0],
    })
    return df


@pytest.fixture
def missing_column_df():
    """A dataset with a required column removed."""
    return pd.DataFrame({
        "study_hours": [5.0, 6.0],
        "attendance_percentage": [80.0, 75.0],
        # "previous_exam_score" is missing
        "assignment_completion_percentage": [90.0, 85.0],
        "sleep_hours": [7.0, 6.5],
        "extracurricular_hours": [2.0, 3.0],
        "final_exam_score": [70.0, 68.0],
    })


@pytest.fixture
def sample_features():
    """Single prediction input as dict."""
    return {
        "study_hours": 5.0,
        "attendance_percentage": 85.0,
        "previous_exam_score": 72.0,
        "assignment_completion_percentage": 90.0,
        "sleep_hours": 7.0,
        "extracurricular_hours": 2.0,
    }
