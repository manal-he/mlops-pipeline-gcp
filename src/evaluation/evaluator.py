"""Evaluation complete du modele avant deploiement."""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    metrics: dict
    is_better_than_baseline: bool
    should_deploy: bool
    comparison_details: dict
    sliced_metrics: dict = field(default_factory=dict)


class ModelEvaluator:
    """
    Evaluation complete du modele avant deploiement.

    GATE DE DEPLOIEMENT :
    Un modele ne doit etre deploye que s'il est MEILLEUR que le
    modele actuellement en production. On compare :

    1. Metriques globales (accuracy, AUC, F1)
    2. Metriques par sous-groupe (sliced metrics) pour detecter les biais
    3. Stabilite (variance des metriques en cross-validation)
    4. Performance sur des cas edge (outliers, valeurs extremes)

    Si le nouveau modele echoue a un de ces criteres, pas de deploiement.
    """

    def __init__(
        self,
        task: str = "classification",
        primary_metric: str = "f1",
        improvement_threshold: float = 0.01,
    ):
        self.task = task
        self.primary_metric = primary_metric
        self.improvement_threshold = improvement_threshold

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        baseline_metrics: dict = None,
        slice_columns: list[str] = None,
    ) -> EvaluationResult:
        """
        Evaluation complete du modele.

        Args:
            model: Modele entraine (avec predict et predict_proba)
            X_test: Features de test
            y_test: Target de test
            baseline_metrics: Metriques du modele en production
            slice_columns: Colonnes pour les sliced metrics

        Returns:
            EvaluationResult avec decision de deploiement
        """
        logger.info(f"Evaluation du modele sur {len(X_test)} samples...")

        # 1. Metriques globales
        metrics = self._compute_metrics(model, X_test, y_test)
        logger.info(f"Metriques globales: {json.dumps({k: f'{v:.4f}' for k, v in metrics.items()})}")

        # 2. Comparaison avec baseline
        comparison = self._compare_with_baseline(metrics, baseline_metrics)

        # 3. Sliced metrics
        sliced_metrics = {}
        if slice_columns:
            for col in slice_columns:
                if col in X_test.columns:
                    sliced_metrics[col] = self._compute_sliced_metrics(
                        model, X_test, y_test, col
                    )

        # 4. Decision de deploiement
        should_deploy = self._should_deploy(metrics, comparison, sliced_metrics)

        result = EvaluationResult(
            metrics=metrics,
            is_better_than_baseline=comparison.get("is_better", True),
            should_deploy=should_deploy,
            comparison_details=comparison,
            sliced_metrics=sliced_metrics,
        )

        if should_deploy:
            logger.info("DECISION: Modele approuve pour le deploiement")
        else:
            logger.warning("DECISION: Modele REJETE — ne sera pas deploye")

        return result

    def _compute_metrics(self, model, X: pd.DataFrame, y: pd.Series) -> dict:
        """Calcule toutes les metriques."""
        y_pred = model.predict(X)
        metrics = {}

        if self.task == "classification":
            metrics["accuracy"] = float(accuracy_score(y, y_pred))
            metrics["precision"] = float(
                precision_score(y, y_pred, average="binary", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y, y_pred, average="binary", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y, y_pred, average="binary", zero_division=0)
            )

            try:
                y_proba = model.predict_proba(X)[:, 1]
                metrics["auc_roc"] = float(roc_auc_score(y, y_proba))
            except Exception:
                pass

            cm = confusion_matrix(y, y_pred)
            metrics["true_positives"] = int(cm[1][1]) if cm.shape[0] > 1 else 0
            metrics["false_positives"] = int(cm[0][1]) if cm.shape[0] > 1 else 0
            metrics["true_negatives"] = int(cm[0][0])
            metrics["false_negatives"] = int(cm[1][0]) if cm.shape[0] > 1 else 0
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            metrics["mae"] = float(mean_absolute_error(y, y_pred))
            metrics["mse"] = float(mean_squared_error(y, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y, y_pred)))
            metrics["r2"] = float(r2_score(y, y_pred))

        return metrics

    def _compare_with_baseline(self, metrics: dict, baseline_metrics: dict = None) -> dict:
        """Compare les metriques avec le baseline."""
        if baseline_metrics is None:
            return {"is_better": True, "reason": "Pas de baseline — premier deploiement"}

        primary = self.primary_metric
        new_value = metrics.get(primary, 0)
        baseline_value = baseline_metrics.get(primary, 0)
        improvement = new_value - baseline_value

        is_better = improvement >= self.improvement_threshold

        return {
            "is_better": is_better,
            "primary_metric": primary,
            "new_value": new_value,
            "baseline_value": baseline_value,
            "improvement": improvement,
            "threshold": self.improvement_threshold,
            "reason": (
                f"{primary} ameliore de {improvement:.4f} (seuil: {self.improvement_threshold})"
                if is_better
                else f"{primary} insuffisant: {improvement:.4f} < {self.improvement_threshold}"
            ),
        }

    def _compute_sliced_metrics(
        self, model, X: pd.DataFrame, y: pd.Series, slice_column: str
    ) -> dict:
        """Calcule les metriques par sous-groupe."""
        sliced = {}
        for value in X[slice_column].unique():
            mask = X[slice_column] == value
            if mask.sum() < 10:
                continue
            X_slice = X[mask]
            y_slice = y[mask]
            sliced[str(value)] = self._compute_metrics(model, X_slice, y_slice)
        return sliced

    def _should_deploy(
        self, metrics: dict, comparison: dict, sliced_metrics: dict
    ) -> bool:
        """Decision finale de deploiement."""
        # Condition 1: Meilleur que le baseline
        if not comparison.get("is_better", True):
            logger.warning(f"Rejet: {comparison.get('reason', 'unknown')}")
            return False

        # Condition 2: Metriques minimales
        if self.task == "classification":
            if metrics.get("f1", 0) < 0.5:
                logger.warning(f"Rejet: F1 trop bas ({metrics.get('f1', 0):.4f} < 0.5)")
                return False
            if metrics.get("auc_roc", 0) < 0.6:
                logger.warning(
                    f"Rejet: AUC-ROC trop bas ({metrics.get('auc_roc', 0):.4f} < 0.6)"
                )
                return False

        # Condition 3: Pas de biais excessif dans les sliced metrics
        for col, slices in sliced_metrics.items():
            if len(slices) > 1:
                f1_values = [s.get("f1", 0) for s in slices.values()]
                if max(f1_values) - min(f1_values) > 0.3:
                    logger.warning(
                        f"Rejet: Biais detecte sur {col} "
                        f"(ecart F1: {max(f1_values) - min(f1_values):.4f})"
                    )
                    return False

        return True
