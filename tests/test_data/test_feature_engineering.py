"""Tests pour le feature engineering."""

import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    def test_fit_transform(self, sample_training_data, feature_columns):
        fe = FeatureEngineer()
        X, y = fe.fit_transform(
            sample_training_data,
            target_column="is_churned",
            numerical_columns=feature_columns,
            categorical_columns=[],
        )

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(sample_training_data)
        assert "is_churned" not in X.columns
        assert fe._is_fitted is True

    def test_transform_requires_fit(self):
        fe = FeatureEngineer()
        with pytest.raises(RuntimeError, match="n'est pas fit"):
            fe.transform(pd.DataFrame({"a": [1, 2, 3]}))

    def test_save_and_load(self, sample_training_data, feature_columns):
        fe = FeatureEngineer()
        fe.fit_transform(
            sample_training_data,
            target_column="is_churned",
            numerical_columns=feature_columns,
            categorical_columns=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fe.save(tmpdir)
            loaded_fe = FeatureEngineer.load(tmpdir)

            assert loaded_fe._is_fitted is True
            assert loaded_fe.feature_columns == fe.feature_columns

    def test_derived_features_created(self, sample_training_data, feature_columns):
        fe = FeatureEngineer()
        X, _ = fe.fit_transform(
            sample_training_data,
            target_column="is_churned",
            numerical_columns=feature_columns,
            categorical_columns=[],
        )

        # Les features derivees doivent etre creees
        assert "spend_per_transaction" in X.columns
        assert "spend_acceleration" in X.columns
        assert "activity_ratio" in X.columns

    def test_missing_values_handled(self, sample_training_data, feature_columns):
        df = sample_training_data.copy()
        df.loc[0, "total_spend"] = np.nan
        df.loc[1, "avg_spend"] = np.nan

        fe = FeatureEngineer()
        X, _ = fe.fit_transform(
            df,
            target_column="is_churned",
            numerical_columns=feature_columns,
            categorical_columns=[],
        )

        # Pas de NaN apres transformation
        assert X.isnull().sum().sum() == 0
