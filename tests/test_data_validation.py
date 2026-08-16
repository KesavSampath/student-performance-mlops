"""
tests/test_data_validation.py
==============================
Tests for Pandera data validation (src/data_validation.py).

Covers:
  - Valid dataset passes validation
  - Out-of-range values are rejected
  - Missing required column is rejected
  - Null/NaN values are rejected
"""

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data_validation import validate_dataset, validate_inference_input


class TestValidDataset:
    def test_valid_dataset_passes(self, valid_df):
        """A clean dataset should validate without errors."""
        validated = validate_dataset(valid_df)
        assert len(validated) == len(valid_df)

    def test_valid_dataset_returns_dataframe(self, valid_df):
        result = validate_dataset(valid_df)
        assert isinstance(result, pd.DataFrame)

    def test_valid_inference_input_passes(self, sample_features):
        df = pd.DataFrame([sample_features])
        validated = validate_inference_input(df)
        assert len(validated) == 1


class TestInvalidRanges:
    def test_attendance_above_100_rejected(self):
        df = pd.DataFrame({
            "study_hours": [5.0],
            "attendance_percentage": [150.0],  # INVALID
            "previous_exam_score": [70.0],
            "assignment_completion_percentage": [80.0],
            "sleep_hours": [7.0],
            "extracurricular_hours": [2.0],
            "final_exam_score": [65.0],
        })
        with pytest.raises(pa.errors.SchemaErrors):
            validate_dataset(df)

    def test_negative_study_hours_rejected(self):
        df = pd.DataFrame({
            "study_hours": [-1.0],  # INVALID
            "attendance_percentage": [80.0],
            "previous_exam_score": [70.0],
            "assignment_completion_percentage": [80.0],
            "sleep_hours": [7.0],
            "extracurricular_hours": [2.0],
            "final_exam_score": [65.0],
        })
        with pytest.raises(pa.errors.SchemaErrors):
            validate_dataset(df)

    def test_final_score_above_100_rejected(self):
        df = pd.DataFrame({
            "study_hours": [5.0],
            "attendance_percentage": [80.0],
            "previous_exam_score": [70.0],
            "assignment_completion_percentage": [80.0],
            "sleep_hours": [7.0],
            "extracurricular_hours": [2.0],
            "final_exam_score": [110.0],  # INVALID
        })
        with pytest.raises(pa.errors.SchemaErrors):
            validate_dataset(df)

    def test_invalid_range_df_rejected(self, invalid_range_df):
        with pytest.raises(pa.errors.SchemaErrors):
            validate_dataset(invalid_range_df)


class TestMissingColumns:
    def test_missing_column_raises_error(self, missing_column_df):
        """Missing a required column should raise SchemaErrors."""
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            validate_dataset(missing_column_df)


class TestNullValues:
    def test_null_in_attendance_rejected(self):
        df = pd.DataFrame({
            "study_hours": [5.0, np.nan],
            "attendance_percentage": [80.0, 75.0],
            "previous_exam_score": [70.0, 65.0],
            "assignment_completion_percentage": [80.0, 90.0],
            "sleep_hours": [7.0, 6.0],
            "extracurricular_hours": [2.0, 1.0],
            "final_exam_score": [65.0, 60.0],
        })
        with pytest.raises((pa.errors.SchemaErrors, pa.errors.SchemaError)):
            validate_dataset(df)
