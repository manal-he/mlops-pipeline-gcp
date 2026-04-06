"""Tests pour l'API de serving."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.serving.app as app_module
from src.serving.app import app


@pytest.fixture
def mock_model():
    """Mock du modele ML."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    return model


@pytest.fixture
def client(mock_model):
    """Client de test FastAPI avec modele mocke."""
    app_module.model = mock_model
    app_module.preprocessor = None
    app_module.model_metadata = {
        "model_type": "xgboost",
        "version": "v1",
        "training_date": "2024-01-01",
        "metrics": {"f1": 0.85},
        "n_features": 15,
        "feature_names": ["total_transactions", "total_spend"],
    }
    return TestClient(app)


class TestServingAPI:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_check_no_model(self):
        app_module.model = None
        app_module.model_metadata = {}
        test_client = TestClient(app)
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_predict(self, client):
        response = client.post("/predict", json={
            "features": {
                "total_transactions": 25,
                "total_spend": 1500.0,
                "days_since_last": 5,
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "latency_ms" in data

    def test_predict_no_model(self):
        app_module.model = None
        test_client = TestClient(app)
        response = test_client.post("/predict", json={
            "features": {"total_transactions": 25}
        })
        assert response.status_code == 503

    def test_model_info(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "xgboost"
        assert "metrics" in data

    def test_predict_batch(self, client, mock_model):
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        response = client.post("/predict/batch", json=[
            {"features": {"total_transactions": 25}},
            {"features": {"total_transactions": 10}},
        ])
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
