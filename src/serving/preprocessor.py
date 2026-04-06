"""Preprocesseur pour le serving — recharge les transformeurs sauvegardes."""

from pathlib import Path

import joblib
import pandas as pd
from loguru import logger


class ServingPreprocessor:
    """
    Applique les memes transformations que pendant le training.

    Charge les scalers et encoders sauvegardes pour eviter
    le train/serve skew.
    """

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Charge les artefacts de feature engineering."""
        fe_path = self.artifacts_dir / "feature_engineer.joblib"
        if fe_path.exists():
            self.artifacts = joblib.load(fe_path)
            logger.info(
                f"Artefacts charges: {len(self.artifacts.get('feature_columns', []))} features"
            )
        else:
            logger.warning(f"Artefacts non trouves: {fe_path}")
            self.artifacts = {
                "scalers": {},
                "encoders": {},
                "feature_columns": [],
            }

    def preprocess(self, features: dict) -> pd.DataFrame:
        """
        Preprocesse les features d'une requete de prediction.

        Args:
            features: Dictionnaire des features brutes

        Returns:
            DataFrame pret pour la prediction
        """
        df = pd.DataFrame([features])

        # Appliquer les encoders
        for col, encoder in self.artifacts.get("encoders", {}).items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda x, enc=encoder: (
                        enc.transform([x])[0] if x in enc.classes_ else -1
                    )
                )

        # Appliquer le scaler
        scaler = self.artifacts.get("scalers", {}).get("numerical")
        if scaler is not None:
            num_cols = [c for c in scaler.feature_names_in_ if c in df.columns]
            if num_cols:
                df[num_cols] = scaler.transform(df[num_cols])

        # S'assurer que toutes les colonnes attendues sont presentes
        for col in self.artifacts.get("feature_columns", []):
            if col not in df.columns:
                df[col] = 0

        expected_cols = self.artifacts.get("feature_columns", list(df.columns))
        return df[[c for c in expected_cols if c in df.columns]]
