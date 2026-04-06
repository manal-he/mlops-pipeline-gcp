"""Integration Cloud Monitoring pour les alertes."""

import time

from google.cloud import monitoring_v3
from loguru import logger


class AlertManager:
    """
    Gestion des alertes via Google Cloud Monitoring.

    Envoie des metriques custom et configure des alertes pour :
    - Drift detecte
    - Performance du modele degradee
    - Latence de l'API trop elevee
    - Erreurs de prediction
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.project_name = f"projects/{project_id}"
        self.client = monitoring_v3.MetricServiceClient()

    def write_custom_metric(
        self,
        metric_type: str,
        value: float,
        labels: dict = None,
    ) -> None:
        """
        Ecrit une metrique custom dans Cloud Monitoring.

        Args:
            metric_type: Type de metrique (ex: "model/drift_score")
            value: Valeur de la metrique
            labels: Labels additionnels
        """
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/mlops/{metric_type}"

        if labels:
            for key, val in labels.items():
                series.metric.labels[key] = str(val)

        series.resource.type = "global"
        series.resource.labels["project_id"] = self.project_id

        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)

        interval = monitoring_v3.TimeInterval(
            end_time={"seconds": seconds, "nanos": nanos}
        )
        point = monitoring_v3.Point(
            interval=interval,
            value={"double_value": value},
        )
        series.points = [point]

        self.client.create_time_series(
            request={"name": self.project_name, "time_series": [series]}
        )
        logger.debug(f"Metrique ecrite: {metric_type}={value}")

    def report_drift_metrics(self, drift_results: dict) -> None:
        """Envoie les metriques de drift a Cloud Monitoring."""
        # Metrique globale
        self.write_custom_metric(
            "model/drift_percentage",
            drift_results.get("drift_percentage", 0),
        )

        self.write_custom_metric(
            "model/overall_drift",
            1.0 if drift_results.get("overall_drift", False) else 0.0,
        )

        # Metriques par feature
        for feature in drift_results.get("feature_results", []):
            self.write_custom_metric(
                "model/feature_drift_score",
                feature.get("drift_score", 0),
                labels={"feature_name": feature.get("feature_name", "unknown")},
            )

    def report_prediction_metrics(
        self,
        latency_ms: float,
        prediction: float,
        model_version: str = "unknown",
    ) -> None:
        """Envoie les metriques de prediction."""
        self.write_custom_metric(
            "serving/prediction_latency_ms",
            latency_ms,
            labels={"model_version": model_version},
        )

        self.write_custom_metric(
            "serving/prediction_value",
            prediction,
            labels={"model_version": model_version},
        )

    def report_model_performance(self, metrics: dict) -> None:
        """Envoie les metriques de performance du modele."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.write_custom_metric(
                    f"model/performance/{metric_name}",
                    float(value),
                )
