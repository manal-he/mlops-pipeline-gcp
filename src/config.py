"""Configuration centralisee du projet MLOps."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GCPConfig:
    project_id: str = os.getenv("PROJECT_ID", "mlops-pipeline-prod")
    region: str = os.getenv("REGION", "europe-west1")
    dataset_id: str = os.getenv("DATASET_ID", "ml_data")
    artifacts_bucket: str = os.getenv(
        "ARTIFACTS_BUCKET", f"{os.getenv('PROJECT_ID', 'mlops-pipeline-prod')}-mlops-artifacts"
    )
    sa_email: str = os.getenv(
        "SA_EMAIL",
        f"mlops-pipeline-sa@{os.getenv('PROJECT_ID', 'mlops-pipeline-prod')}.iam.gserviceaccount.com",
    )


@dataclass
class ModelConfig:
    model_type: str = os.getenv("MODEL_TYPE", "xgboost")
    task: str = "classification"
    target_column: str = os.getenv("TARGET_COLUMN", "is_churned")
    baseline_f1: float = float(os.getenv("BASELINE_F1", "0.75"))
    improvement_threshold: float = float(os.getenv("IMPROVEMENT_THRESHOLD", "0.01"))
    numerical_columns: list[str] = field(
        default_factory=lambda: [
            "total_transactions",
            "active_days",
            "days_since_last",
            "customer_lifetime",
            "total_spend",
            "avg_spend",
            "std_spend",
            "max_spend",
            "min_spend",
            "category_diversity",
            "merchant_diversity",
            "spend_last_30d",
            "spend_prev_30d",
            "spend_trend_ratio",
            "spend_per_active_day",
        ]
    )
    categorical_columns: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    pipeline_name: str = os.getenv("PIPELINE_NAME", "mlops-training-pipeline")
    pipeline_root: str = os.getenv("PIPELINE_ROOT", "")
    service_name: str = os.getenv("SERVICE_NAME", "ml-prediction-api")
    model_uri: str = os.getenv("MODEL_URI", "")


@dataclass
class Config:
    gcp: GCPConfig = field(default_factory=GCPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self):
        if not self.pipeline.pipeline_root:
            self.pipeline.pipeline_root = (
                f"gs://{self.gcp.artifacts_bucket}/pipeline-root/"
            )
        if not self.pipeline.model_uri:
            self.pipeline.model_uri = (
                f"gs://{self.gcp.artifacts_bucket}/models/latest/"
            )


config = Config()
