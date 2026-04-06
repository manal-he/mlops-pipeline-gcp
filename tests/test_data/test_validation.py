"""Tests pour la validation des donnees."""

import numpy as np
import pandas as pd

from src.data.validation import DataValidator, ValidationResult


class TestDataValidator:
    def test_valid_data(self, sample_training_data, validation_config):
        validator = DataValidator(validation_config)
        result = validator.validate(sample_training_data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.passed_checks == result.total_checks

    def test_missing_columns(self, sample_training_data, validation_config):
        df = sample_training_data.drop(columns=["total_spend"])
        validator = DataValidator(validation_config)
        result = validator.validate(df)

        assert result.is_valid is False
        assert any(c["check"] == "schema" for c in result.failed_checks)

    def test_insufficient_volume(self, validation_config):
        df = pd.DataFrame({
            "total_transactions": [1, 2, 3],
            "total_spend": [10.0, 20.0, 30.0],
            "avg_spend": [10.0, 20.0, 30.0],
            "is_churned": [0, 1, 0],
        })
        validator = DataValidator(validation_config)
        result = validator.validate(df)

        assert result.is_valid is False
        assert any(c["check"] == "volume" for c in result.failed_checks)

    def test_high_null_ratio(self, sample_training_data, validation_config):
        df = sample_training_data.copy()
        # Mettre 50% de nulls dans une colonne
        mask = np.random.choice([True, False], size=len(df), p=[0.5, 0.5])
        df.loc[mask, "total_spend"] = np.nan

        validator = DataValidator(validation_config)
        result = validator.validate(df)

        assert result.is_valid is False
        assert any(c["check"] == "nulls" for c in result.failed_checks)

    def test_out_of_range_values(self, sample_training_data, validation_config):
        df = sample_training_data.copy()
        df.loc[0, "total_spend"] = -100  # Valeur negative hors plage

        validator = DataValidator(validation_config)
        result = validator.validate(df)

        assert result.is_valid is False
        assert any(c["check"] == "ranges" for c in result.failed_checks)

    def test_statistics_computed(self, sample_training_data, validation_config):
        validator = DataValidator(validation_config)
        result = validator.validate(sample_training_data)

        assert "num_rows" in result.statistics
        assert result.statistics["num_rows"] == len(sample_training_data)
