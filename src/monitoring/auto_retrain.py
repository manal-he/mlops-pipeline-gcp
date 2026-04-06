"""Declenchement de reentrainement automatique."""

from datetime import datetime

from google.cloud import aiplatform
from loguru import logger


class AutoRetrainer:
    """
    Declenche un reentrainement automatique quand un drift est detecte.

    BOUCLE DE FEEDBACK :
    Production -> Monitoring -> Drift detecte -> Pipeline relance
         ^                                           |
         +---- Nouveau modele deploye <--------------+

    Declencheurs de reentrainement :
    1. Drift detecte (score PSI > seuil)
    2. Performance degradee (metrique en production < seuil)
    3. Planifie (hebdomadaire, mensuel)
    4. Volume de nouvelles donnees suffisant
    """

    def __init__(
        self,
        project_id: str,
        region: str,
        pipeline_name: str,
    ):
        self.project_id = project_id
        self.region = region
        self.pipeline_name = pipeline_name
        aiplatform.init(project=project_id, location=region)

    def should_retrain(
        self, drift_results: dict, performance_metrics: dict = None
    ) -> tuple[bool, str]:
        """
        Decide si un reentrainement est necessaire.

        Returns:
            (should_retrain: bool, reason: str)
        """
        # Condition 1 : Drift significatif
        if drift_results.get("overall_drift", False):
            drift_pct = drift_results.get("drift_percentage", 0)
            return True, f"Data drift detecte ({drift_pct:.0%} des features)"

        # Condition 2 : Performance degradee
        if performance_metrics:
            primary_metric = performance_metrics.get("f1", 1.0)
            if primary_metric < 0.6:
                return True, f"Performance degradee (F1={primary_metric:.4f})"

        # Condition 3 : Trop de features avec drift modere
        feature_results = drift_results.get("feature_results", [])
        moderate_drift = sum(
            1
            for f in feature_results
            if f.get("drift_score", 0) > 0.05
        )
        if feature_results and moderate_drift > len(feature_results) * 0.3:
            return True, (
                f"Drift modere sur {moderate_drift}/{len(feature_results)} features"
            )

        return False, "Pas de reentrainement necessaire"

    def trigger_retrain(
        self,
        pipeline_template_path: str,
        pipeline_root: str,
        parameters: dict = None,
    ) -> str:
        """
        Lance le pipeline de reentrainement sur Vertex AI.

        Args:
            pipeline_template_path: Chemin GCS vers le pipeline compile
            pipeline_root: Racine GCS pour les artefacts du pipeline
            parameters: Parametres du pipeline

        Returns:
            ID du job pipeline
        """
        logger.info("Declenchement du reentrainement automatique...")

        default_params = {
            "project_id": self.project_id,
            "region": self.region,
            "trigger_reason": "auto_retrain",
            "triggered_at": datetime.now().isoformat(),
        }

        if parameters:
            default_params.update(parameters)

        job = aiplatform.PipelineJob(
            display_name=f"{self.pipeline_name}-retrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            template_path=pipeline_template_path,
            pipeline_root=pipeline_root,
            parameter_values=default_params,
            enable_caching=False,  # Pas de cache pour le retrain
        )

        sa_email = f"mlops-pipeline-sa@{self.project_id}.iam.gserviceaccount.com"
        job.submit(service_account=sa_email)

        logger.info(f"Pipeline de reentrainement soumis: {job.resource_name}")
        return job.resource_name

    def check_and_retrain(
        self,
        drift_results: dict,
        performance_metrics: dict = None,
        pipeline_template_path: str = None,
        pipeline_root: str = None,
        parameters: dict = None,
    ) -> dict:
        """
        Verifie le drift et declenche le retrain si necessaire.

        Returns:
            {
                "should_retrain": bool,
                "reason": str,
                "pipeline_job_id": str or None,
            }
        """
        should, reason = self.should_retrain(drift_results, performance_metrics)

        result = {
            "should_retrain": should,
            "reason": reason,
            "pipeline_job_id": None,
            "checked_at": datetime.now().isoformat(),
        }

        if should and pipeline_template_path and pipeline_root:
            try:
                job_id = self.trigger_retrain(
                    pipeline_template_path, pipeline_root, parameters
                )
                result["pipeline_job_id"] = job_id
                logger.info(f"Retrain declenche: {reason}")
            except Exception as e:
                logger.error(f"Erreur lors du declenchement du retrain: {e}")
                result["error"] = str(e)
        elif should:
            logger.warning(
                f"Retrain necessaire ({reason}) mais pas de template configure"
            )

        return result
