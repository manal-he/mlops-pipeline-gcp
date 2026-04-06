"""Tests pour la detection de drift."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift_detector import DriftDetector


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 1000),
        "feature_b": np.random.normal(10, 2, 1000),
        "feature_c": np.random.uniform(0, 100, 1000),
    })


@pytest.fixture
def no_drift_data():
    np.random.seed(123)
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 1000),
        "feature_b": np.random.normal(10, 2, 1000),
        "feature_c": np.random.uniform(0, 100, 1000),
    })


@pytest.fixture
def drifted_data():
    np.random.seed(99)
    return pd.DataFrame({
        "feature_a": np.random.normal(5, 3, 1000),  # Moyenne deplacee
        "feature_b": np.random.normal(20, 5, 1000),  # Moyenne et std changees
        "feature_c": np.random.uniform(50, 200, 1000),  # Plage changee
    })


class TestDriftDetector:
    def test_no_drift_ks(self, reference_data, no_drift_data):
        detector = DriftDetector(method="ks")
        results = detector.detect_drift(reference_data, no_drift_data)

        assert results["overall_drift"] is False
        assert results["n_features_analyzed"] == 3

    def test_drift_detected_ks(self, reference_data, drifted_data):
        detector = DriftDetector(method="ks")
        results = detector.detect_drift(reference_data, drifted_data)

        assert results["overall_drift"] is True
        assert results["n_features_drifted"] > 0

    def test_drift_psi(self, reference_data, drifted_data):
        detector = DriftDetector(method="psi", threshold=0.1)
        results = detector.detect_drift(reference_data, drifted_data)

        assert results["n_features_analyzed"] == 3
        # Au moins certaines features doivent avoir du drift
        assert results["n_features_drifted"] > 0

    def test_drift_js(self, reference_data, drifted_data):
        detector = DriftDetector(method="js", threshold=0.05)
        results = detector.detect_drift(reference_data, drifted_data)

        assert results["n_features_analyzed"] == 3

    def test_feature_results_structure(self, reference_data, drifted_data):
        detector = DriftDetector(method="ks")
        results = detector.detect_drift(reference_data, drifted_data)

        for feature_result in results["feature_results"]:
            assert "feature_name" in feature_result
            assert "drift_detected" in feature_result
            assert "drift_score" in feature_result
            assert "method" in feature_result

    def test_specific_columns(self, reference_data, drifted_data):
        detector = DriftDetector(method="ks")
        results = detector.detect_drift(
            reference_data, drifted_data,
            feature_columns=["feature_a"],
        )

        assert results["n_features_analyzed"] == 1

    def test_psi_computation(self):
        ref = np.random.normal(0, 1, 10000)
        cur = np.random.normal(0, 1, 10000)
        psi = DriftDetector._compute_psi(ref, cur)

        # Memes distributions -> PSI faible
        assert psi < 0.1

        cur_shifted = np.random.normal(5, 1, 10000)
        psi_shifted = DriftDetector._compute_psi(ref, cur_shifted)

        # Distributions differentes -> PSI eleve
        assert psi_shifted > 0.1
