"""Pipeline MLOps complet orchestre avec Vertex AI Pipelines (Kubeflow)."""

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output


# ============================
# COMPOSANT 1 : DATA INGESTION
# ============================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-bigquery",
        "google-cloud-storage",
        "pandas",
        "pyarrow",
        "loguru",
    ],
)
def data_ingestion_component(
    project_id: str,
    dataset_id: str,
    start_date: str,
    end_date: str,
    output_bucket: str,
    train_data: Output[Dataset],
    val_data: Output[Dataset],
    test_data: Output[Dataset],
    data_stats: Output[Artifact],
):
    """Composant d'ingestion et preparation des donnees."""
    import json

    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT * FROM `{project_id}.{dataset_id}.features`
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """
    df = client.query(query).to_dataframe()

    # Split temporel (70/15/15)
    df = df.sort_values("date") if "date" in df.columns else df
    n = len(df)
    train = df.iloc[: int(n * 0.7)]
    val = df.iloc[int(n * 0.7) : int(n * 0.85)]
    test = df.iloc[int(n * 0.85) :]

    # Sauvegarder
    train.to_csv(train_data.path, index=False)
    val.to_csv(val_data.path, index=False)
    test.to_csv(test_data.path, index=False)

    # Stats
    stats = {
        "total_rows": len(df),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
    }
    with open(data_stats.path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


# ============================
# COMPOSANT 2 : TRAINING
# ============================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "loguru",
    ],
)
def training_component(
    train_data: Input[Dataset],
    val_data: Input[Dataset],
    target_column: str,
    model_type: str,
    trained_model: Output[Model],
    training_metrics: Output[Metrics],
):
    """Composant d'entrainement du modele."""
    import json

    import joblib
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    train = pd.read_csv(train_data.path)
    val = pd.read_csv(val_data.path)

    # Separer features et target
    feature_cols = [c for c in train.columns if c != target_column and c != "user_id" and c != "date"]
    X_train = train[feature_cols]
    y_train = train[target_column]
    X_val = val[feature_cols]
    y_val = val[target_column]

    # Entrainement XGBoost
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Metriques
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    metrics = {
        "val_f1": float(f1_score(y_val, y_pred)),
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "val_auc_roc": float(roc_auc_score(y_val, y_proba)),
        "n_features": len(feature_cols),
    }

    # Sauvegarder
    joblib.dump(model, trained_model.path)

    with open(training_metrics.path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Logger les metriques pour Vertex AI
    training_metrics.log_metric("val_f1", metrics["val_f1"])
    training_metrics.log_metric("val_accuracy", metrics["val_accuracy"])
    training_metrics.log_metric("val_auc_roc", metrics["val_auc_roc"])


# ============================
# COMPOSANT 3 : EVALUATION
# ============================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "joblib",
        "loguru",
    ],
)
def evaluation_component(
    trained_model: Input[Model],
    test_data: Input[Dataset],
    target_column: str,
    baseline_f1: float,
    improvement_threshold: float,
    eval_metrics: Output[Metrics],
    eval_report: Output[Artifact],
) -> bool:
    """Composant d'evaluation avec gate de deploiement."""
    import json

    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

    model = joblib.load(trained_model.path)
    test = pd.read_csv(test_data.path)

    feature_cols = [c for c in test.columns if c != target_column and c != "user_id" and c != "date"]
    X_test = test[feature_cols]
    y_test = test[target_column]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_auc_roc": float(roc_auc_score(y_test, y_proba)),
    }

    # Gate de deploiement
    improvement = metrics["test_f1"] - baseline_f1
    should_deploy = improvement >= improvement_threshold and metrics["test_f1"] >= 0.5

    report = {
        "metrics": metrics,
        "baseline_f1": baseline_f1,
        "improvement": improvement,
        "improvement_threshold": improvement_threshold,
        "should_deploy": should_deploy,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    eval_metrics.log_metric("test_f1", metrics["test_f1"])
    eval_metrics.log_metric("should_deploy", 1.0 if should_deploy else 0.0)

    with open(eval_report.path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return should_deploy


# ============================
# COMPOSANT 4 : DEPLOYMENT
# ============================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-storage",
        "google-cloud-run",
        "joblib",
        "loguru",
    ],
)
def deployment_component(
    trained_model: Input[Model],
    project_id: str,
    region: str,
    output_bucket: str,
    deployment_info: Output[Artifact],
):
    """Composant de deploiement du modele."""
    import json
    from datetime import datetime

    from google.cloud import storage

    # Upload du modele vers GCS
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(output_bucket)

    version = datetime.now().strftime("v%Y%m%d-%H%M%S")
    blob = bucket.blob(f"models/{version}/model.joblib")
    blob.upload_from_filename(trained_model.path)

    # Mettre a jour le latest
    bucket.blob("models/latest/model.joblib")
    bucket.copy_blob(blob, bucket, "models/latest/model.joblib")

    info = {
        "version": version,
        "gcs_uri": f"gs://{output_bucket}/models/{version}/",
        "deployed_at": datetime.now().isoformat(),
        "status": "deployed",
    }

    with open(deployment_info.path, "w") as f:
        json.dump(info, f, indent=2)


# ============================
# COMPOSANT 5 : MONITORING
# ============================
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas",
        "numpy",
        "scipy",
        "loguru",
    ],
)
def monitoring_component(
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    target_column: str,
    drift_report: Output[Artifact],
):
    """Composant de detection de drift."""
    import json

    import pandas as pd
    from scipy import stats

    train = pd.read_csv(train_data.path)
    test = pd.read_csv(test_data.path)

    feature_cols = [
        c for c in train.columns
        if c != target_column and c != "user_id" and c != "date"
        and train[c].dtype in ["float64", "int64"]
    ]

    drift_results = []
    for col in feature_cols:
        ref = train[col].dropna().values
        cur = test[col].dropna().values
        if len(ref) > 0 and len(cur) > 0:
            statistic, p_value = stats.ks_2samp(ref, cur)
            drift_results.append({
                "feature": col,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": p_value < 0.05,
            })

    n_drifted = sum(1 for r in drift_results if r["drift_detected"])
    report = {
        "n_features": len(drift_results),
        "n_drifted": n_drifted,
        "drift_percentage": n_drifted / len(drift_results) if drift_results else 0,
        "overall_drift": n_drifted > len(drift_results) * 0.3,
        "feature_results": drift_results,
    }

    with open(drift_report.path, "w") as f:
        json.dump(report, f, indent=2)


# ============================
# PIPELINE COMPLET
# ============================
@dsl.pipeline(
    name="mlops-training-pipeline",
    description="Pipeline MLOps end-to-end: data ingestion, training, evaluation, deployment, monitoring",
)
def mlops_pipeline(
    project_id: str,
    region: str,
    dataset_id: str,
    start_date: str,
    end_date: str,
    target_column: str = "is_churned",
    model_type: str = "xgboost",
    baseline_f1: float = 0.75,
    improvement_threshold: float = 0.01,
    output_bucket: str = "",
):
    """Pipeline MLOps complet."""

    # Etape 1: Data Ingestion
    ingestion = data_ingestion_component(
        project_id=project_id,
        dataset_id=dataset_id,
        start_date=start_date,
        end_date=end_date,
        output_bucket=output_bucket,
    )

    # Etape 2: Training
    training = training_component(
        train_data=ingestion.outputs["train_data"],
        val_data=ingestion.outputs["val_data"],
        target_column=target_column,
        model_type=model_type,
    )

    # Etape 3: Evaluation
    evaluation = evaluation_component(
        trained_model=training.outputs["trained_model"],
        test_data=ingestion.outputs["test_data"],
        target_column=target_column,
        baseline_f1=baseline_f1,
        improvement_threshold=improvement_threshold,
    )

    # Etape 4: Deployment (conditionnel)
    with dsl.If(evaluation.output == True):  # noqa: E712
        deployment_component(
            trained_model=training.outputs["trained_model"],
            project_id=project_id,
            region=region,
            output_bucket=output_bucket,
        )

    # Etape 5: Monitoring
    monitoring_component(
        train_data=ingestion.outputs["train_data"],
        test_data=ingestion.outputs["test_data"],
        target_column=target_column,
    )
