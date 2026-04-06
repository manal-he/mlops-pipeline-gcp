"""Tests pour l'evaluateur de modele."""

import pytest

from src.evaluation.evaluator import EvaluationResult, ModelEvaluator
from src.training.trainer import ModelTrainer


@pytest.fixture
def trained_model(sample_features, sample_target):
    trainer = ModelTrainer(model_type="xgboost")
    trainer.train(sample_features, sample_target)
    return trainer.model


class TestModelEvaluator:
    def test_evaluate_classification(self, trained_model, sample_features, sample_target):
        evaluator = ModelEvaluator(task="classification")
        result = evaluator.evaluate(trained_model, sample_features, sample_target)

        assert isinstance(result, EvaluationResult)
        assert "f1" in result.metrics
        assert "accuracy" in result.metrics
        assert "auc_roc" in result.metrics
        assert isinstance(result.should_deploy, bool)

    def test_first_deployment_always_approved(self, trained_model, sample_features, sample_target):
        evaluator = ModelEvaluator(task="classification")
        result = evaluator.evaluate(
            trained_model, sample_features, sample_target,
            baseline_metrics=None,
        )

        # Sans baseline, c'est un premier deploiement
        assert result.is_better_than_baseline is True

    def test_better_than_baseline(self, trained_model, sample_features, sample_target):
        evaluator = ModelEvaluator(
            task="classification",
            primary_metric="f1",
            improvement_threshold=0.01,
        )

        # Baseline tres bas -> le nouveau modele devrait etre meilleur
        baseline = {"f1": 0.1, "accuracy": 0.1}
        result = evaluator.evaluate(
            trained_model, sample_features, sample_target,
            baseline_metrics=baseline,
        )

        assert result.is_better_than_baseline is True

    def test_worse_than_baseline(self, trained_model, sample_features, sample_target):
        evaluator = ModelEvaluator(
            task="classification",
            primary_metric="f1",
            improvement_threshold=0.01,
        )

        # Baseline tres haut -> le nouveau modele ne devrait pas etre meilleur
        baseline = {"f1": 0.999}
        result = evaluator.evaluate(
            trained_model, sample_features, sample_target,
            baseline_metrics=baseline,
        )

        assert result.is_better_than_baseline is False
        assert result.should_deploy is False

    def test_comparison_details(self, trained_model, sample_features, sample_target):
        evaluator = ModelEvaluator()
        baseline = {"f1": 0.5}
        result = evaluator.evaluate(
            trained_model, sample_features, sample_target,
            baseline_metrics=baseline,
        )

        assert "primary_metric" in result.comparison_details
        assert "improvement" in result.comparison_details
        assert "threshold" in result.comparison_details
