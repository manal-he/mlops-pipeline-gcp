#!/bin/bash
set -e

# Configuration
export PROJECT_ID=${PROJECT_ID:-"mlops-pipeline-prod"}
export REGION=${REGION:-"europe-west1"}

echo "=== Configuration GCP pour MLOps Pipeline ==="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"

# 1. Se connecter a GCP
echo "=== Authentification..."
gcloud auth login
gcloud auth application-default login

# 2. Configurer le projet
echo "=== Configuration du projet..."
gcloud config set project ${PROJECT_ID}

# 3. Activer les APIs necessaires
echo "=== Activation des APIs..."
gcloud services enable \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    storage.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    container.googleapis.com \
    iam.googleapis.com \
    secretmanager.googleapis.com

# 4. Creer un Service Account pour le pipeline
echo "=== Creation du Service Account..."
gcloud iam service-accounts create mlops-pipeline-sa \
    --display-name="MLOps Pipeline Service Account" \
    2>/dev/null || echo "Service account deja existant"

# 5. Attribuer les roles necessaires
SA_EMAIL="mlops-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com"
echo "=== Attribution des roles a ${SA_EMAIL}..."

for ROLE in \
    roles/aiplatform.user \
    roles/bigquery.dataEditor \
    roles/bigquery.jobUser \
    roles/storage.objectAdmin \
    roles/run.admin \
    roles/artifactregistry.writer \
    roles/monitoring.editor \
    roles/logging.logWriter \
    roles/iam.serviceAccountUser; do
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="${ROLE}" \
        --quiet
done

# 6. Creer un bucket GCS pour les artefacts
echo "=== Creation du bucket GCS..."
gsutil mb -l ${REGION} gs://${PROJECT_ID}-mlops-artifacts/ 2>/dev/null || echo "Bucket deja existant"

# 7. Creer un Artifact Registry pour Docker
echo "=== Creation de l'Artifact Registry..."
gcloud artifacts repositories create mlops-docker \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker images for MLOps pipeline" \
    2>/dev/null || echo "Registry deja existant"

# 8. Configurer Docker pour le registry
echo "=== Configuration Docker..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev

echo ""
echo "=== Setup termine! ==="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service Account: ${SA_EMAIL}"
echo "Artifacts Bucket: gs://${PROJECT_ID}-mlops-artifacts/"
echo "Docker Registry: ${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-docker/"
