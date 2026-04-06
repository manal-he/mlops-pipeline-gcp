"""Tests pour le module d'entrainement."""

import tempfile

import numpy as np
import pytest

from src.training.trainer import ModelTrainer


class TestModelTrainer:
    def test_train_xgboost(self, sample_features, sample_target):
        trainer = ModelTrainer(model_type="xgboost", task="classification")
        metrics = trainer.train(sample_features, sample_target)

        assert "train_f1" in metrics
        assert "train_accuracy" in metrics
        assert "cv_mean" in metrics
        assert trainer.model is not None

    def test_train_random_forest(self, sample_features, sample_target):
        trainer = ModelTrainer(model_type="random_forest", task="classification")
        metrics = trainer.train(sample_features, sample_target)

        assert "train_f1" in metrics
        assert trainer.model is not None

    def test_train_with_validation(self, sample_features, sample_target):
        n = len(sample_features)
        split = int(n * 0.8)

        X_train = sample_features.iloc[:split]
        y_train = sample_target.iloc[:split]
        X_val = sample_features.iloc[split:]
        y_val = sample_target.iloc[split:]

        trainer = ModelTrainer(model_type="xgboost")
        metrics = trainer.train(X_train, y_train, X_val, y_val)

        assert "val_f1" in metrics
        assert "val_accuracy" in metrics

    def test_predict(self, sample_features, sample_target):
        trainer = ModelTrainer()
        trainer.train(sample_features, sample_target)

        predictions = trainer.predict(sample_features)
        assert len(predictions) == len(sample_features)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, sample_features, sample_target):
        trainer = ModelTrainer()
        trainer.train(sample_features, sample_target)

        probas = trainer.predict_proba(sample_features)
        assert probas.shape == (len(sample_features), 2)
        assert all(0 <= p <= 1 for p in probas[:, 1])

    def test_save_and_load(self, sample_features, sample_target):
        trainer = ModelTrainer()
        trainer.train(sample_features, sample_target)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save(tmpdir)
            loaded = ModelTrainer.load(tmpdir)

            assert loaded.model is not None
            assert loaded.model_type == trainer.model_type

            # Predictions identiques
            pred_original = trainer.predict(sample_features)
            pred_loaded = loaded.predict(sample_features)
            np.testing.assert_array_equal(pred_original, pred_loaded)

    def test_training_metadata(self, sample_features, sample_target):
        trainer = ModelTrainer()
        trainer.train(sample_features, sample_target)

        meta = trainer.training_metadata
        assert "model_type" in meta
        assert "hyperparameters" in meta
        assert "feature_names" in meta
        assert "training_duration_seconds" in meta
        assert meta["n_train_samples"] == len(sample_features)

    def test_feature_importance(self, sample_features, sample_target):
        trainer = ModelTrainer()
        trainer.train(
            sample_features, sample_target,
            feature_names=list(sample_features.columns),
        )

        importance = trainer.training_metadata.get("feature_importance", {})
        assert len(importance) > 0

    def test_unsupported_model_type(self):
        trainer = ModelTrainer(model_type="unsupported")
        with pytest.raises(ValueError, match="non supporte"):
            trainer._create_model()
