"""Recherche automatique des meilleurs hyperparametres."""

import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


class HyperparameterTuner:
    """
    Recherche automatique des meilleurs hyperparametres.

    Methodes de recherche :
    - Grid Search : Teste toutes les combinaisons (exhaustif mais lent)
    - Random Search : Echantillonne aleatoirement (plus rapide, souvent suffisant)
    - Bayesian Optimization : Utilise les resultats precedents pour guider
      la recherche (Optuna, Vertex AI Vizier)
    """

    XGBOOST_SEARCH_SPACE = {
        "n_estimators": [100, 200, 300, 500, 800, 1000],
        "max_depth": [3, 4, 5, 6, 7, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7, 10],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0, 5.0],
        "gamma": [0, 0.1, 0.3, 0.5, 1.0],
    }

    def __init__(
        self,
        task: str = "classification",
        n_iter: int = 50,
        cv_folds: int = 5,
        scoring: str = None,
        random_state: int = 42,
    ):
        self.task = task
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.scoring = scoring or (
            "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
        )
        self.random_state = random_state

    def tune(self, X_train, y_train) -> dict:
        """
        Lance la recherche d'hyperparametres.

        Returns:
            {
                "best_params": dict,
                "best_score": float,
                "cv_results": dict,
            }
        """
        logger.info(
            f"Hyperparameter tuning — {self.n_iter} iterations, "
            f"{self.cv_folds}-fold CV, scoring={self.scoring}"
        )

        if self.task == "classification":
            base_model = xgb.XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1,
            )
            cv = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
        else:
            base_model = xgb.XGBRegressor(
                random_state=self.random_state, n_jobs=-1
            )
            cv = self.cv_folds

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.XGBOOST_SEARCH_SPACE,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        best_score = search.best_score_

        logger.info(f"Meilleurs hyperparametres: {best_params}")
        logger.info(f"Meilleur score ({self.scoring}): {best_score:.4f}")

        # Top 5 des resultats
        results = []
        for i in range(min(5, len(search.cv_results_["mean_test_score"]))):
            idx = search.cv_results_["rank_test_score"].tolist().index(i + 1)
            results.append({
                "rank": i + 1,
                "mean_test_score": float(search.cv_results_["mean_test_score"][idx]),
                "std_test_score": float(search.cv_results_["std_test_score"][idx]),
                "params": {
                    k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
                    for k, v in search.cv_results_["params"][idx].items()
                },
            })

        return {
            "best_params": {
                k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in best_params.items()
            },
            "best_score": float(best_score),
            "top_results": results,
        }
