"""API de serving FastAPI pour les predictions ML."""

import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

# Variables globales pour le modele (charge au demarrage)
model = None
preprocessor = None
model_metadata = None


def load_model_from_gcs(model_uri: str) -> tuple:
    """Charge le modele et ses artefacts depuis GCS."""
    from google.cloud import storage

    # Parse GCS URI
    # Format: gs://bucket-name/path/to/model/
    parts = model_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Telecharger dans un repertoire temporaire
    local_dir = tempfile.mkdtemp()

    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_path = blob.name[len(prefix):].lstrip("/")
        local_path = Path(local_dir) / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    # Charger le modele
    loaded_model = joblib.load(Path(local_dir) / "model.joblib")

    # Charger les metadonnees
    metadata_path = Path(local_dir) / "metadata.json"
    loaded_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            loaded_metadata = json.load(f)

    # Charger le preprocesseur
    from src.serving.preprocessor import ServingPreprocessor

    loaded_preprocessor = None
    fe_path = Path(local_dir) / "feature_engineer.joblib"
    if fe_path.exists():
        loaded_preprocessor = ServingPreprocessor(local_dir)

    return loaded_model, loaded_preprocessor, loaded_metadata


def load_model_local(model_dir: str) -> tuple:
    """Charge le modele depuis un repertoire local."""
    model_path = Path(model_dir)
    loaded_model = joblib.load(model_path / "model.joblib")

    metadata_path = model_path / "metadata.json"
    loaded_metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            loaded_metadata = json.load(f)

    from src.serving.preprocessor import ServingPreprocessor

    loaded_preprocessor = None
    fe_path = model_path / "feature_engineer.joblib"
    if fe_path.exists():
        loaded_preprocessor = ServingPreprocessor(str(model_path))

    return loaded_model, loaded_preprocessor, loaded_metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modele au demarrage du conteneur."""
    global model, preprocessor, model_metadata

    model_uri = os.getenv("MODEL_URI", "")
    model_dir = os.getenv("MODEL_DIR", "")

    if model_uri.startswith("gs://"):
        logger.info(f"Chargement du modele depuis GCS: {model_uri}")
        model, preprocessor, model_metadata = load_model_from_gcs(model_uri)
    elif model_dir:
        logger.info(f"Chargement du modele depuis: {model_dir}")
        model, preprocessor, model_metadata = load_model_local(model_dir)
    else:
        logger.warning("Aucun modele configure (MODEL_URI ou MODEL_DIR)")
        model_metadata = {}

    if model is not None:
        logger.info(f"Modele charge: {model_metadata.get('model_type', 'unknown')}")

    yield
    logger.info("Arret du serveur de prediction")


app = FastAPI(
    title="MLOps Prediction API",
    description="API de prediction pour le pipeline MLOps sur GCP",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """Schema de requete de prediction."""

    features: dict = Field(
        ...,
        description="Dictionnaire des features",
        examples=[
            {
                "total_transactions": 25,
                "total_spend": 1500.0,
                "days_since_last": 5,
                "category_diversity": 4,
                "avg_spend": 60.0,
                "std_spend": 25.0,
                "max_spend": 150.0,
                "min_spend": 10.0,
                "active_days": 20,
                "customer_lifetime": 180,
                "merchant_diversity": 8,
                "spend_last_30d": 300.0,
                "spend_prev_30d": 350.0,
            }
        ],
    )


class PredictionResponse(BaseModel):
    """Schema de reponse de prediction."""

    prediction: int | float
    probability: float | None = None
    model_version: str = "unknown"
    latency_ms: float


class HealthResponse(BaseModel):
    """Schema de reponse health check."""

    status: str
    model_loaded: bool
    model_type: str | None = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de health check."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type=model_metadata.get("model_type") if model_metadata else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint de prediction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    start_time = time.time()

    try:
        # Preprocesser les features
        if preprocessor is not None:
            df = preprocessor.preprocess(request.features)
        else:
            df = pd.DataFrame([request.features])

        # Prediction
        prediction = model.predict(df)[0]

        # Probabilite (classification uniquement)
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)[0]
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                pass

        latency_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=int(prediction) if isinstance(prediction, (np.integer,)) else float(prediction),
            probability=probability,
            model_version=model_metadata.get("version", "unknown"),
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error(f"Erreur de prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Informations sur le modele en production."""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    return {
        "model_type": model_metadata.get("model_type"),
        "training_date": model_metadata.get("training_date"),
        "metrics": model_metadata.get("metrics", {}),
        "n_features": model_metadata.get("n_features"),
        "feature_names": model_metadata.get("feature_names", []),
    }


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Prediction en batch."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    start_time = time.time()
    results = []

    for req in requests:
        if preprocessor is not None:
            df = preprocessor.preprocess(req.features)
        else:
            df = pd.DataFrame([req.features])

        prediction = model.predict(df)[0]
        probability = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(df)[0]
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            except Exception:
                pass

        results.append({
            "prediction": int(prediction) if isinstance(prediction, (np.integer,)) else float(prediction),
            "probability": probability,
        })

    latency_ms = (time.time() - start_time) * 1000

    return {
        "predictions": results,
        "count": len(results),
        "latency_ms": round(latency_ms, 2),
    }
