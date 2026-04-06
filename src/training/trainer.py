"""Entrainement de modeles ML avec tracking des experiences."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


class ModelTrainer:
    """
    Entrainement de modeles ML avec tracking des experiences.

    Chaque entrainement produit des "artefacts" qu'il faut sauvegarder :
    - Le modele serialise (.joblib)
    - Les metriques de performance
    - Les hyperparametres utilises
    - Les features utilisees
    - Les transformeurs de donnees (scalers, encoders)
    - Les metadonnees (date, duree, version des donnees)
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        task: str = "classification",
        hyperparameters: dict = None,
    ):
        self.model_type = model_type
        self.task = task
        self.hyperparameters = hyperparameters or self._default_hyperparameters()
        self.model = None
        self.metrics = {}
        self.training_metadata = {}

    def _default_hyperparameters(self) -> dict:
        """Hyperparametres par defaut selon le type de modele."""
        defaults = {
            "xgboost": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
            },
            "random_forest": {
                "n_estimators": 300,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "n_jobs": -1,
            },
            "logistic": {
                "C": 1.0,
                "max_iter": 1000,
                "random_state": 42,
            },
        }
        return defaults.get(self.model_type, defaults["xgboost"])

    def _create_model(self):
        """Cree le modele selon le type et la tache."""
        if self.model_type == "xgboost":
            if self.task == "classification":
                return xgb.XGBClassifier(
                    **self.hyperparameters,
                    eval_metric="logloss",
                    use_label_encoder=False,
                )
            return xgb.XGBRegressor(**self.hyperparameters)

        if self.model_type == "random_forest":
            if self.task == "classification":
                return RandomForestClassifier(**self.hyperparameters)
            return RandomForestRegressor(**self.hyperparameters)

        if self.model_type == "logistic":
            return LogisticRegression(**self.hyperparameters)

        raise ValueError(f"Type de modele non supporte: {self.model_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        feature_names: list[str] = None,
    ) -> dict:
        """
        Entraine le modele et retourne les metriques.

        Args:
            X_train: Features d'entrainement
            y_train: Target d'entrainement
            X_val: Features de validation (optionnel)
            y_val: Target de validation (optionnel)
            feature_names: Noms des features

        Returns:
            Dictionnaire de metriques
        """
        start_time = datetime.now()
        logger.info(
            f"Entrainement {self.model_type} ({self.task}) — "
            f"{len(X_train)} samples, {len(X_train.columns)} features"
        )

        self.model = self._create_model()

        # Entrainement avec early stopping si validation disponible
        if X_val is not None and y_val is not None and self.model_type == "xgboost":
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)

        # Calculer les metriques
        train_metrics = self._compute_metrics(X_train, y_train, prefix="train")
        self.metrics.update(train_metrics)

        if X_val is not None and y_val is not None:
            val_metrics = self._compute_metrics(X_val, y_val, prefix="val")
            self.metrics.update(val_metrics)

        # Cross-validation sur le train set
        cv_scores = cross_val_score(
            self._create_model(),
            X_train,
            y_train,
            cv=5,
            scoring="f1" if self.task == "classification" else "neg_root_mean_squared_error",
        )
        self.metrics["cv_mean"] = float(cv_scores.mean())
        self.metrics["cv_std"] = float(cv_scores.std())

        # Feature importance
        feature_importance = self._get_feature_importance(feature_names or list(X_train.columns))

        # Metadata
        duration = (datetime.now() - start_time).total_seconds()
        self.training_metadata = {
            "model_type": self.model_type,
            "task": self.task,
            "hyperparameters": self.hyperparameters,
            "feature_names": feature_names or list(X_train.columns),
            "feature_importance": feature_importance,
            "n_train_samples": len(X_train),
            "n_features": len(X_train.columns),
            "training_duration_seconds": duration,
            "training_date": datetime.now().isoformat(),
            "metrics": self.metrics,
        }

        logger.info(f"Entrainement termine en {duration:.1f}s")
        logger.info(f"Metriques: {json.dumps({k: f'{v:.4f}' for k, v in self.metrics.items()})}")

        return self.metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction."""
        if self.model is None:
            raise RuntimeError("Le modele n'est pas entraine.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction de probabilites (classification uniquement)."""
        if self.model is None:
            raise RuntimeError("Le modele n'est pas entraine.")
        if self.task != "classification":
            raise ValueError("predict_proba n'est disponible qu'en classification.")
        return self.model.predict_proba(X)

    def save(self, output_dir: str) -> dict:
        """Sauvegarde le modele et ses metadonnees."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Modele
        model_path = output_path / "model.joblib"
        joblib.dump(self.model, model_path)

        # Metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2, default=str)

        logger.info(f"Modele sauvegarde dans {output_dir}")
        return {"model_path": str(model_path), "metadata_path": str(metadata_path)}

    @classmethod
    def load(cls, input_dir: str) -> "ModelTrainer":
        """Charge un modele sauvegarde."""
        input_path = Path(input_dir)

        with open(input_path / "metadata.json") as f:
            metadata = json.load(f)

        trainer = cls(
            model_type=metadata["model_type"],
            task=metadata["task"],
            hyperparameters=metadata["hyperparameters"],
        )
        trainer.model = joblib.load(input_path / "model.joblib")
        trainer.training_metadata = metadata
        trainer.metrics = metadata.get("metrics", {})

        logger.info(f"Modele charge depuis {input_dir}")
        return trainer

    def _compute_metrics(
        self, X: pd.DataFrame, y: pd.Series, prefix: str
    ) -> dict:
        """Calcule les metriques de performance."""
        y_pred = self.model.predict(X)
        metrics = {}

        if self.task == "classification":
            metrics[f"{prefix}_accuracy"] = float(accuracy_score(y, y_pred))
            metrics[f"{prefix}_precision"] = float(
                precision_score(y, y_pred, average="binary", zero_division=0)
            )
            metrics[f"{prefix}_recall"] = float(
                recall_score(y, y_pred, average="binary", zero_division=0)
            )
            metrics[f"{prefix}_f1"] = float(
                f1_score(y, y_pred, average="binary", zero_division=0)
            )
            try:
                y_proba = self.model.predict_proba(X)[:, 1]
                metrics[f"{prefix}_auc_roc"] = float(roc_auc_score(y, y_proba))
            except Exception:
                pass
        else:
            metrics[f"{prefix}_mae"] = float(mean_absolute_error(y, y_pred))
            metrics[f"{prefix}_mse"] = float(mean_squared_error(y, y_pred))
            metrics[f"{prefix}_rmse"] = float(np.sqrt(mean_squared_error(y, y_pred)))
            metrics[f"{prefix}_r2"] = float(r2_score(y, y_pred))

        return metrics

    def _get_feature_importance(self, feature_names: list[str]) -> dict:
        """Retourne l'importance des features."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            return {
                feature_names[i]: float(importances[i])
                for i in sorted_idx[:20]
            }
        return {}
