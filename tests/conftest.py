"""Fixtures communes pour les tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_training_data():
    """Genere un DataFrame de donnees d'entrainement synthetiques."""
    np.random.seed(42)
    n = 1000

    data = {
        "user_id": [f"U{i:04d}" for i in range(n)],
        "total_transactions": np.random.randint(1, 100, n),
        "active_days": np.random.randint(1, 365, n),
        "days_since_last": np.random.randint(0, 90, n),
        "customer_lifetime": np.random.randint(30, 730, n),
        "total_spend": np.random.uniform(10, 10000, n),
        "avg_spend": np.random.uniform(5, 500, n),
        "std_spend": np.random.uniform(0, 200, n),
        "max_spend": np.random.uniform(50, 2000, n),
        "min_spend": np.random.uniform(1, 50, n),
        "category_diversity": np.random.randint(1, 15, n),
        "merchant_diversity": np.random.randint(1, 30, n),
        "spend_last_30d": np.random.uniform(0, 3000, n),
        "spend_prev_30d": np.random.uniform(0, 3000, n),
        "spend_trend_ratio": np.random.uniform(0, 3, n),
        "spend_per_active_day": np.random.uniform(0, 100, n),
        "is_churned": np.random.randint(0, 2, n),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features(sample_training_data):
    """Retourne les features sans la target."""
    return sample_training_data.drop(columns=["is_churned", "user_id"])


@pytest.fixture
def sample_target(sample_training_data):
    """Retourne la target."""
    return sample_training_data["is_churned"]


@pytest.fixture
def feature_columns():
    """Liste des colonnes de features numeriques."""
    return [
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


@pytest.fixture
def validation_config():
    """Configuration de validation des donnees."""
    return {
        "expected_columns": [
            "total_transactions",
            "total_spend",
            "avg_spend",
            "is_churned",
        ],
        "max_null_ratio": 0.1,
        "min_rows": 100,
        "numerical_ranges": {
            "total_spend": (0, 100000),
            "total_transactions": (0, 10000),
        },
    }
