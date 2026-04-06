output "service_account_email" {
  value = google_service_account.mlops_sa.email
}

output "artifacts_bucket" {
  value = google_storage_bucket.mlops_artifacts.name
}

output "prediction_api_url" {
  value = google_cloud_run_v2_service.prediction_api.uri
}

output "bigquery_dataset" {
  value = google_bigquery_dataset.ml_data.dataset_id
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.mlops_docker.repository_id}"
}
