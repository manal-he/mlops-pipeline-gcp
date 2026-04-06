#!/bin/bash
set -e

PROJECT_ID=${PROJECT_ID:-"mlops-pipeline-prod"}
REGION=${REGION:-"europe-west1"}
SERVICE_NAME="ml-prediction-api"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/mlops-docker/${SERVICE_NAME}"
MODEL_URI="gs://${PROJECT_ID}-mlops-artifacts/models/latest/"

echo "=== Building Docker image..."
docker build -t ${IMAGE_NAME}:latest -f src/serving/Dockerfile .

echo "=== Pushing to Artifact Registry..."
docker push ${IMAGE_NAME}:latest

echo "=== Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --region=${REGION} \
    --platform=managed \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --concurrency=80 \
    --timeout=60 \
    --set-env-vars="MODEL_URI=${MODEL_URI},PROJECT_ID=${PROJECT_ID}" \
    --service-account="mlops-pipeline-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --allow-unauthenticated

echo "=== Deploiement termine!"
gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)'
