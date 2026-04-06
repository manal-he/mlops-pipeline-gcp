"""Detection de data drift et concept drift."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


@dataclass
class DriftResult:
    feature_name: str
    drift_detected: bool
    drift_score: float  # 0 = pas de drift, 1 = drift total
    method: str
    p_value: Optional[float]
    threshold: float
    details: dict


class DriftDetector:
    """
    Detection de data drift et concept drift.

    Types de drift :

    1. DATA DRIFT (covariate shift) :
       La distribution des features d'entree change.

    2. CONCEPT DRIFT :
       La relation entre les features et la target change.

    3. PREDICTION DRIFT :
       La distribution des predictions change.

    Methodes de detection :
    - KS Test (Kolmogorov-Smirnov) : Compare 2 distributions continues
    - PSI (Population Stability Index) : Mesure le changement de distribution
    - Chi2 Test : Compare 2 distributions categorielles
    - Jensen-Shannon Divergence : Mesure la distance entre distributions
    """

    def __init__(
        self,
        method: str = "ks",
        threshold: float = 0.1,
        significance_level: float = 0.05,
    ):
        self.method = method
        self.threshold = threshold
        self.significance_level = significance_level

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_columns: list[str] = None,
    ) -> dict:
        """
        Detecte le drift entre les donnees de reference et les donnees courantes.

        Args:
            reference_data: Donnees de training (reference)
            current_data: Donnees de production (courantes)
            feature_columns: Colonnes a analyser

        Returns:
            {
                "overall_drift": bool,
                "drift_percentage": float,
                "feature_results": [DriftResult, ...]
            }
        """
        if feature_columns is None:
            feature_columns = [
                c for c in reference_data.columns
                if reference_data[c].dtype in ["float64", "int64", "float32", "int32"]
            ]

        results = []
        for col in feature_columns:
            if col not in reference_data.columns or col not in current_data.columns:
                continue

            ref_values = reference_data[col].dropna().values
            cur_values = current_data[col].dropna().values

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            if self.method == "ks":
                result = self._ks_test(col, ref_values, cur_values)
            elif self.method == "psi":
                result = self._psi_test(col, ref_values, cur_values)
            elif self.method == "js":
                result = self._js_divergence(col, ref_values, cur_values)
            else:
                result = self._ks_test(col, ref_values, cur_values)

            results.append(result)

        # Resultat global
        drifted_features = [r for r in results if r.drift_detected]
        drift_percentage = len(drifted_features) / len(results) if results else 0

        # Drift global si >30% des features ont du drift
        overall_drift = drift_percentage > 0.3

        if overall_drift:
            logger.warning(
                f"DRIFT DETECTE: {len(drifted_features)}/{len(results)} features "
                f"({drift_percentage:.0%})"
            )
        else:
            logger.info(
                f"Pas de drift significatif: {len(drifted_features)}/{len(results)} features "
                f"({drift_percentage:.0%})"
            )

        return {
            "overall_drift": overall_drift,
            "drift_percentage": drift_percentage,
            "n_features_analyzed": len(results),
            "n_features_drifted": len(drifted_features),
            "feature_results": [
                {
                    "feature_name": r.feature_name,
                    "drift_detected": r.drift_detected,
                    "drift_score": r.drift_score,
                    "method": r.method,
                    "p_value": r.p_value,
                    "threshold": r.threshold,
                }
                for r in results
            ],
        }

    def _ks_test(self, feature_name: str, ref: np.ndarray, cur: np.ndarray) -> DriftResult:
        """Test de Kolmogorov-Smirnov."""
        statistic, p_value = stats.ks_2samp(ref, cur)

        return DriftResult(
            feature_name=feature_name,
            drift_detected=p_value < self.significance_level,
            drift_score=float(statistic),
            method="ks",
            p_value=float(p_value),
            threshold=self.significance_level,
            details={
                "ks_statistic": float(statistic),
                "ref_mean": float(np.mean(ref)),
                "cur_mean": float(np.mean(cur)),
                "ref_std": float(np.std(ref)),
                "cur_std": float(np.std(cur)),
            },
        )

    def _psi_test(self, feature_name: str, ref: np.ndarray, cur: np.ndarray) -> DriftResult:
        """Population Stability Index."""
        psi_value = self._compute_psi(ref, cur)

        return DriftResult(
            feature_name=feature_name,
            drift_detected=psi_value > self.threshold,
            drift_score=float(psi_value),
            method="psi",
            p_value=None,
            threshold=self.threshold,
            details={
                "psi_value": float(psi_value),
                "interpretation": (
                    "No drift" if psi_value < 0.1
                    else "Moderate drift" if psi_value < 0.25
                    else "Significant drift"
                ),
            },
        )

    def _js_divergence(self, feature_name: str, ref: np.ndarray, cur: np.ndarray) -> DriftResult:
        """Jensen-Shannon Divergence."""
        # Creer des histogrammes normalises
        bins = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=50)
        ref_hist, _ = np.histogram(ref, bins=bins, density=True)
        cur_hist, _ = np.histogram(cur, bins=bins, density=True)

        # Normaliser pour obtenir des distributions de probabilite
        ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
        cur_hist = cur_hist / (cur_hist.sum() + 1e-10)

        # JS Divergence
        m = 0.5 * (ref_hist + cur_hist)
        js = 0.5 * (
            stats.entropy(ref_hist + 1e-10, m + 1e-10)
            + stats.entropy(cur_hist + 1e-10, m + 1e-10)
        )

        return DriftResult(
            feature_name=feature_name,
            drift_detected=js > self.threshold,
            drift_score=float(js),
            method="js_divergence",
            p_value=None,
            threshold=self.threshold,
            details={"js_divergence": float(js)},
        )

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Calcule le Population Stability Index (PSI)."""
        # Creer des bins basees sur les quantiles de la reference
        breakpoints = np.percentile(
            reference, np.linspace(0, 100, n_bins + 1)
        )
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Calculer les proportions dans chaque bin
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)

        # Eviter les log(0)
        ref_pct = np.clip(ref_pct, 1e-4, None)
        cur_pct = np.clip(cur_pct, 1e-4, None)

        # PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
