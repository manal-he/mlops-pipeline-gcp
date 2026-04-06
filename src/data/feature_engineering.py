"""Transformation des donnees brutes en features pour le modele."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    """
    Transforme les donnees brutes en features pour le modele.

    CONCEPT CLE — Train/Serve Skew :

    Les transformations appliquees pendant le training DOIVENT etre
    identiques pendant le serving (prediction en production).

    Solution : Sauvegarder les transformeurs (scaler, encoders) apres le
    fit sur les donnees de training, et les recharger pendant le serving.
    """

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self._is_fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str,
        numerical_columns: list[str],
        categorical_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit les transformeurs sur les donnees de training ET transforme.
        Cette methode est appelee UNIQUEMENT pendant le training.
        """
        logger.info("Feature engineering — fit_transform")
        df = df.copy()

        # Separer target
        y = df[target_column].copy()
        df = df.drop(columns=[target_column])

        # 1. Traitement des valeurs manquantes
        df = self._handle_missing(df, numerical_columns, categorical_columns)

        # 2. Encoding des variables categorielles
        for col in categorical_columns:
            if col in df.columns:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder

        # 3. Scaling des variables numeriques
        valid_num_cols = [c for c in numerical_columns if c in df.columns]
        if valid_num_cols:
            scaler = StandardScaler()
            df[valid_num_cols] = scaler.fit_transform(df[valid_num_cols])
            self.scalers["numerical"] = scaler

        # 4. Creer des features additionnelles
        df = self._create_derived_features(df, valid_num_cols)

        self.feature_columns = list(df.columns)
        self._is_fitted = True

        logger.info(f"Features generees: {len(self.feature_columns)} colonnes")
        return df, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforme les donnees en utilisant les transformeurs deja fit.
        Cette methode est appelee pendant le serving.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureEngineer n'est pas fit. Appelez fit_transform d'abord."
            )

        logger.info("Feature engineering — transform (serving mode)")
        df = df.copy()

        # Meme pipeline que fit_transform mais sans fit
        numerical_columns = [
            c for c in df.columns if c in self.scalers.get("numerical", StandardScaler()).feature_names_in_
        ] if "numerical" in self.scalers else []

        categorical_columns = list(self.encoders.keys())

        df = self._handle_missing(df, numerical_columns, categorical_columns)

        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(
                    lambda x, enc=encoder: (
                        enc.transform([x])[0] if x in enc.classes_ else -1
                    )
                )

        if "numerical" in self.scalers and numerical_columns:
            df[numerical_columns] = self.scalers["numerical"].transform(
                df[numerical_columns]
            )

        df = self._create_derived_features(df, numerical_columns)

        # S'assurer que les memes colonnes sont presentes
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        return df[self.feature_columns]

    def save(self, output_dir: str) -> None:
        """Sauvegarde les transformeurs pour le serving."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "scalers": self.scalers,
            "encoders": self.encoders,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(artifacts, output_path / "feature_engineer.joblib")
        logger.info(f"Feature engineer sauvegarde dans {output_dir}")

    @classmethod
    def load(cls, input_dir: str) -> "FeatureEngineer":
        """Charge les transformeurs sauvegardes."""
        input_path = Path(input_dir)
        artifacts = joblib.load(input_path / "feature_engineer.joblib")

        fe = cls()
        fe.scalers = artifacts["scalers"]
        fe.encoders = artifacts["encoders"]
        fe.feature_columns = artifacts["feature_columns"]
        fe._is_fitted = True

        logger.info(f"Feature engineer charge depuis {input_dir}")
        return fe

    def _handle_missing(
        self,
        df: pd.DataFrame,
        numerical_columns: list[str],
        categorical_columns: list[str],
    ) -> pd.DataFrame:
        """Traite les valeurs manquantes."""
        for col in numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna("UNKNOWN")

        return df

    def _create_derived_features(
        self, df: pd.DataFrame, numerical_columns: list[str]
    ) -> pd.DataFrame:
        """Cree des features derivees."""
        if "total_spend" in df.columns and "total_transactions" in df.columns:
            df["spend_per_transaction"] = np.where(
                df["total_transactions"] > 0,
                df["total_spend"] / df["total_transactions"],
                0,
            )

        if "spend_last_30d" in df.columns and "spend_prev_30d" in df.columns:
            df["spend_acceleration"] = df["spend_last_30d"] - df["spend_prev_30d"]

        if "active_days" in df.columns and "customer_lifetime" in df.columns:
            df["activity_ratio"] = np.where(
                df["customer_lifetime"] > 0,
                df["active_days"] / df["customer_lifetime"],
                0,
            )

        return df
