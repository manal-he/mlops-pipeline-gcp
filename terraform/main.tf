# ===========================
# APIs
# ===========================
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "secretmanager.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}

# ===========================
# SERVICE ACCOUNT
# ===========================
resource "google_service_account" "mlops_sa" {
  account_id   = "mlops-pipeline-sa"
  display_name = "MLOps Pipeline Service Account"
}

resource "google_project_iam_member" "mlops_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataEditor",
    "roles/bigquery.jobUser",
    "roles/storage.objectAdmin",
    "roles/run.admin",
    "roles/artifactregistry.writer",
    "roles/monitoring.editor",
    "roles/logging.logWriter",
    "roles/iam.serviceAccountUser",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.mlops_sa.email}"
}

# ===========================
# CLOUD STORAGE
# ===========================
resource "google_storage_bucket" "mlops_artifacts" {
  name          = "${var.project_id}-mlops-artifacts"
  location      = var.region
  force_destroy = var.environment != "prod"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  uniform_bucket_level_access = true
}

# ===========================
# BIGQUERY
# ===========================
resource "google_bigquery_dataset" "ml_data" {
  dataset_id    = "ml_data"
  friendly_name = "ML Data"
  description   = "Dataset pour les donnees ML"
  location      = var.region

  default_table_expiration_ms = null

  access {
    role          = "OWNER"
    special_group = "projectOwners"
  }

  access {
    role          = "WRITER"
    user_by_email = google_service_account.mlops_sa.email
  }
}

# ===========================
# ARTIFACT REGISTRY
# ===========================
resource "google_artifact_registry_repository" "mlops_docker" {
  location      = var.region
  repository_id = "mlops-docker"
  description   = "Docker images for MLOps pipeline"
  format        = "DOCKER"

  cleanup_policy_dry_run = false
}

# ===========================
# CLOUD RUN
# ===========================
resource "google_cloud_run_v2_service" "prediction_api" {
  name     = "ml-prediction-api"
  location = var.region

  template {
    scaling {
      min_instance_count = var.environment == "prod" ? 1 : 0
      max_instance_count = 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/mlops-docker/ml-prediction-api:latest"

      ports {
        container_port = 8080
      }

      env {
        name  = "MODEL_URI"
        value = "gs://${google_storage_bucket.mlops_artifacts.name}/models/latest/"
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }

    service_account = google_service_account.mlops_sa.email
  }

  depends_on = [google_project_service.apis]
}

# Autoriser l'acces public (dev/staging)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  count    = var.environment != "prod" ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.prediction_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ===========================
# MONITORING
# ===========================
resource "google_monitoring_alert_policy" "model_drift" {
  display_name = "MLOps - Model Drift Detected"
  combiner     = "OR"

  conditions {
    display_name = "Drift percentage > 30%"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/mlops/model/drift_percentage\" AND resource.type=\"global\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0.3
      duration        = "300s"
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = []

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "api_latency" {
  display_name = "MLOps - High API Latency"
  combiner     = "OR"

  conditions {
    display_name = "Latency > 2s"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/mlops/serving/prediction_latency_ms\" AND resource.type=\"global\""
      comparison      = "COMPARISON_GT"
      threshold_value = 2000
      duration        = "300s"
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = []

  alert_strategy {
    auto_close = "1800s"
  }
}
