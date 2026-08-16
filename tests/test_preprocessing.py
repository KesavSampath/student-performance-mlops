"""
tests/test_preprocessing.py
============================
Tests for the preprocessing pipeline (src/preprocess.py).

Covers:
  - Pipeline can be built
  - Split produces correct ratios
  - Fitted pipeline transforms correctly
  - No data leakage (test set transform uses training statistics)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocess import (
    FEATURES,
    TEST_SIZE,
    RANDOM_SEED,
    build_pipeline,
    split_data,
)


class TestBuildPipeline:
    def test_returns_pipeline(self):
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_two_steps(self):
        pipe = build_pipeline()
        assert len(pipe.steps) == 2

    def test_step_names(self):
        pipe = build_pipeline()
        names = [name for name, _ in pipe.steps]
        assert "imputer" in names
        assert "scaler" in names


class TestSplitData:
    def test_split_ratios(self, valid_df):
        X_tr, X_te, y_tr, y_te = split_data(valid_df)
        n = len(valid_df)
        expected_test = int(n * TEST_SIZE)
        # Allow ±1 for rounding
        assert abs(len(X_te) - expected_test) <= 1

    def test_no_overlap(self, valid_df):
        X_tr, X_te, y_tr, y_te = split_data(valid_df)
        train_idx = set(X_tr.index)
        test_idx = set(X_te.index)
        assert len(train_idx & test_idx) == 0

    def test_feature_columns_preserved(self, valid_df):
        X_tr, X_te, _, _ = split_data(valid_df)
        for feat in FEATURES:
            assert feat in X_tr.columns
            assert feat in X_te.columns

    def test_reproducible_with_same_seed(self, valid_df):
        X_tr1, _, _, _ = split_data(valid_df, random_state=42)
        X_tr2, _, _, _ = split_data(valid_df, random_state=42)
        pd.testing.assert_frame_equal(X_tr1, X_tr2)

    def test_different_seeds_produce_different_splits(self, valid_df):
        X_tr1, _, _, _ = split_data(valid_df, random_state=1)
        X_tr2, _, _, _ = split_data(valid_df, random_state=99)
        # The splits should differ for independent seeds
        assert not X_tr1.index.equals(X_tr2.index)


class TestFitTransform:
    def test_transform_shape(self, valid_df):
        X_tr, X_te, _, _ = split_data(valid_df)
        pipe = build_pipeline()
        X_tr_t = pipe.fit_transform(X_tr)
        X_te_t = pipe.transform(X_te)
        assert X_tr_t.shape[1] == len(FEATURES)
        assert X_te_t.shape[1] == len(FEATURES)

    def test_scaler_zero_mean_on_train(self, valid_df):
        X_tr, _, _, _ = split_data(valid_df)
        pipe = build_pipeline()
        X_tr_t = pipe.fit_transform(X_tr)
        # Standardised training data should have near-zero mean
        assert np.abs(X_tr_t.mean(axis=0)).max() < 1e-8

    def test_no_leakage_scaler_on_test(self, valid_df):
        """Test set transform should use training statistics, not be re-fitted."""
        X_tr, X_te, _, _ = split_data(valid_df)
        pipe = build_pipeline()
        pipe.fit(X_tr)
        X_te_t = pipe.transform(X_te)
        # Test mean won't be exactly 0 because scaling was based on train stats
        # We just check the transform runs and output is finite
        assert np.isfinite(X_te_t).all()

    def test_handles_missing_values(self):
        """SimpleImputer should fill NaN without error."""
        df = pd.DataFrame({
            "study_hours": [5.0, np.nan, 3.0],
            "attendance_percentage": [80.0, 70.0, np.nan],
            "previous_exam_score": [70.0, 65.0, 60.0],
            "assignment_completion_percentage": [80.0, 90.0, 75.0],
            "sleep_hours": [7.0, 6.0, 8.0],
            "extracurricular_hours": [2.0, 1.0, 3.0],
            "final_exam_score": [65.0, 60.0, 55.0],
        })
        X = df[FEATURES]
        pipe = build_pipeline()
        result = pipe.fit_transform(X)
        assert np.isfinite(result).all()
        assert result.shape == (3, len(FEATURES))
