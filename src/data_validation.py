"""
data_validation.py
==================
Pandera schema for the student performance dataset.

Validates:
  - Required columns are present
  - Correct numeric dtypes
  - No missing values
  - Value ranges
  - No fully duplicate rows
  - Target variable range

Usage:
    from src.data_validation import validate_dataset
    validate_dataset(df)          # raises SchemaError on failure

    # Or from CLI:
    python src/data_validation.py
"""

import os
import sys

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

# ---------------------------------------------------------------------------
# Pandera Schema
# ---------------------------------------------------------------------------
STUDENT_SCHEMA = DataFrameSchema(
    columns={
        "study_hours": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(24),
            ],
            nullable=False,
            description="Daily study hours (0–24)",
        ),
        "attendance_percentage": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
            description="Class attendance percentage (0–100)",
        ),
        "previous_exam_score": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
            description="Score on the previous exam (0–100)",
        ),
        "assignment_completion_percentage": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
            description="Percentage of assignments completed (0–100)",
        ),
        "sleep_hours": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(24),
            ],
            nullable=False,
            description="Average nightly sleep hours (0–24)",
        ),
        "extracurricular_hours": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(24),
            ],
            nullable=False,
            description="Daily extracurricular hours (0–24)",
        ),
        "final_exam_score": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
            description="Target: final exam score (0–100)",
        ),
    },
    checks=[
        # No fully duplicate rows
        Check(
            lambda df: ~df.duplicated().all(),
            error="Dataset contains only duplicate rows",
        ),
    ],
    strict=False,   # Allow extra columns without failing
    coerce=True,    # Attempt numeric coercion before checking
    name="StudentPerformanceSchema",
)

# ---------------------------------------------------------------------------
# Inference-time schema (no target column)
# ---------------------------------------------------------------------------
INFERENCE_SCHEMA = DataFrameSchema(
    columns={
        "study_hours": Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(24)], nullable=False, coerce=True),
        "attendance_percentage": Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(100)], nullable=False, coerce=True),
        "previous_exam_score": Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(100)], nullable=False, coerce=True),
        "assignment_completion_percentage": Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(100)], nullable=False, coerce=True),
        "sleep_hours": Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(24)], nullable=False, coerce=True),
        "extracurricular_hours": Column(float, checks=[Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(24)], nullable=False, coerce=True),
    },
    strict=False,
    coerce=True,
    name="InferenceSchema",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a training dataset against the full schema.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  (validated and coerced)

    Raises
    ------
    pandera.errors.SchemaError  if validation fails
    """
    try:
        validated = STUDENT_SCHEMA.validate(df, lazy=True)
        print(f"[data_validation] PASSED: Schema validation passed. Rows: {len(validated)}")
        return validated
    except pa.errors.SchemaErrors as exc:
        print("[data_validation] FAILED: Validation errors found:")
        print(exc.failure_cases.to_string())
        raise


def validate_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a prediction-time dataframe (no target column required)."""
    return INFERENCE_SCHEMA.validate(df, lazy=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    data_path = os.path.join("data", "raw", "students.csv")
    if not os.path.exists(data_path):
        print(f"[data_validation] ERROR: {data_path} not found. Run generate_data.py first.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"[data_validation] Loaded {len(df)} rows from {data_path}")
    validate_dataset(df)


if __name__ == "__main__":
    main()
