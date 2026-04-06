"""Compilation et lancement du pipeline Vertex AI."""

import click
from kfp import compiler
from google.cloud import aiplatform


def compile_pipeline():
    """Compile le pipeline en fichier YAML."""
    from src.pipeline.pipeline import mlops_pipeline

    compiler.Compiler().compile(
        pipeline_func=mlops_pipeline,
        package_path="pipeline.yaml",
    )
    print("Pipeline compile -> pipeline.yaml")


def run_pipeline(
    project_id: str,
    region: str = "europe-west1",
    pipeline_root: str = None,
):
    """Lance le pipeline sur Vertex AI."""
    aiplatform.init(project=project_id, location=region)

    if pipeline_root is None:
        pipeline_root = f"gs://{project_id}-mlops-artifacts/pipeline-root/"

    job = aiplatform.PipelineJob(
        display_name="mlops-pipeline-run",
        template_path="pipeline.yaml",
        pipeline_root=pipeline_root,
        parameter_values={
            "project_id": project_id,
            "region": region,
            "dataset_id": "ml_data",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_column": "is_churned",
            "baseline_f1": 0.75,
            "improvement_threshold": 0.01,
            "output_bucket": f"{project_id}-mlops-artifacts",
        },
        enable_caching=True,
    )

    job.submit(
        service_account=f"mlops-pipeline-sa@{project_id}.iam.gserviceaccount.com"
    )

    print(f"Pipeline soumis: {job.resource_name}")
    print("Console: https://console.cloud.google.com/vertex-ai/pipelines")

    return job


@click.command()
@click.argument("action", type=click.Choice(["compile", "run"]))
@click.option("--project", default=None, help="GCP Project ID")
@click.option("--region", default="europe-west1", help="GCP Region")
def main(action: str, project: str, region: str):
    """CLI pour le pipeline MLOps."""
    if action == "compile":
        compile_pipeline()
    elif action == "run":
        if not project:
            raise click.UsageError("--project est requis pour 'run'")
        run_pipeline(project, region)


if __name__ == "__main__":
    main()
