# MLOps Pipeline on GCP — Customer Churn Prediction

A production-grade **MLOps Level 2** pipeline for customer churn prediction, built on Google Cloud Platform. This project implements a fully automated ML lifecycle: data ingestion, training, evaluation, deployment, monitoring, and retraining.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![GCP](https://img.shields.io/badge/Cloud-GCP-4285F4)
![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-FF6600)

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   BigQuery   │───>│   Training   │───>│  Evaluation  │───>│  Cloud Run   │
│  (Raw Data)  │    │  (XGBoost)   │    │    (Gate)    │    │  (Serving)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                                       │                    │
       v                                       v                    v
┌──────────────┐                        ┌──────────────┐    ┌──────────────┐
│   Feature    │                        │    Model     │    │  Monitoring  │
│ Engineering  │                        │  Registry    │    │   & Drift    │
└──────────────┘                        │   (GCS)      │    │  Detection   │
                                        └──────────────┘    └──────────────┘
```

**Orchestration:** Vertex AI Pipelines (Kubeflow DSL v2)
**Infrastructure:** Terraform
**CI/CD:** GitHub Actions

---

## Project Structure

```
mlops-pipeline-gcp/
├── src/
│   ├── config.py                  # Centralized configuration
│   ├── data/
│   │   ├── ingestion.py           # BigQuery data extraction
│   │   ├── validation.py          # Data quality checks
│   │   ├── feature_engineering.py # Feature transforms (train/serve skew prevention)
│   │   └── queries/               # SQL templates
│   ├── training/
│   │   ├── trainer.py             # Model training (XGBoost, RF, Logistic)
│   │   ├── hyperparameter_tuning.py # Randomized search CV
│   │   └── model_registry.py      # GCS-based model versioning
│   ├── evaluation/
│   │   └── evaluator.py           # Metrics + deployment gate
│   ├── serving/
│   │   ├── app.py                 # FastAPI prediction API
│   │   ├── preprocessor.py        # Serving-time feature transforms
│   │   └── Dockerfile
│   ├── monitoring/
│   │   ├── drift_detector.py      # KS test, PSI, Jensen-Shannon
│   │   ├── auto_retrain.py        # Automatic retraining trigger
│   │   └── alerting.py            # Cloud Monitoring integration
│   └── pipeline/
│       └── pipeline.py            # Vertex AI Pipeline (Kubeflow DSL)
├── terraform/
│   ├── main.tf                    # GCP infrastructure
│   ├── modules/                   # BigQuery, Cloud Run, Storage, Monitoring
│   └── environments/              # dev / staging / prod tfvars
├── tests/                         # 41 unit tests
├── scripts/
│   ├── setup_gcp.sh               # GCP project bootstrap
│   ├── deploy_cloud_run.sh        # Build & deploy serving API
│   └── run_pipeline.py            # Compile & trigger pipeline
├── .github/workflows/             # CI (lint + test) & CD (pipeline + serving)
├── Makefile
└── requirements.txt
```

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Data Ingestion** | BigQuery SQL with temporal, monetary, and behavioral features |
| **Data Validation** | Schema, volume, null ratio, range, and duplicate checks |
| **Feature Engineering** | StandardScaler + LabelEncoder with saved artifacts (no train/serve skew) |
| **Model Training** | XGBoost, RandomForest, LogisticRegression with cross-validation |
| **Hyperparameter Tuning** | RandomizedSearchCV with configurable search spaces |
| **Evaluation Gate** | Auto-compares new model vs baseline on F1/AUC before deployment |
| **Model Registry** | GCS-based versioning with metadata and promotion workflow |
| **Serving API** | FastAPI with health check, single/batch prediction endpoints |
| **Drift Detection** | Kolmogorov-Smirnov, PSI, Jensen-Shannon divergence |
| **Auto-Retraining** | Triggers Vertex AI Pipeline when drift exceeds threshold |
| **Alerting** | Cloud Monitoring custom metrics and alert policies |
| **Infrastructure** | Terraform modules for all GCP resources |
| **CI/CD** | GitHub Actions for linting, testing, pipeline deployment, and serving |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Google Cloud SDK (for GCP deployment)
- Terraform (for infrastructure)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tests

```bash
make test
# or
pytest tests/ -v
```

### 3. Run the API locally

```bash
# Train a local model first
python -c "
import pandas as pd, numpy as np
from src.data.feature_engineering import FeatureEngineer
from src.training.trainer import ModelTrainer
import os

np.random.seed(42)
n = 1000
num_cols = ['total_transactions','total_spend','avg_transaction_amount',
            'days_since_last_transaction','unique_products',
            'avg_days_between_transactions','transaction_count_trend',
            'spend_trend','tenure_months']
cat_cols = ['region','segment']

data = pd.DataFrame({
    'customer_id': range(n),
    'total_transactions': np.random.randint(1, 100, n),
    'total_spend': np.random.uniform(10, 5000, n),
    'avg_transaction_amount': np.random.uniform(10, 200, n),
    'days_since_last_transaction': np.random.randint(0, 365, n),
    'unique_products': np.random.randint(1, 50, n),
    'avg_days_between_transactions': np.random.uniform(1, 60, n),
    'transaction_count_trend': np.random.uniform(-1, 1, n),
    'spend_trend': np.random.uniform(-1, 1, n),
    'region': np.random.choice(['US', 'EU', 'ASIA'], n),
    'segment': np.random.choice(['premium', 'standard', 'basic'], n),
    'tenure_months': np.random.randint(1, 60, n),
    'churn': np.random.choice([0, 1], n, p=[0.7, 0.3]),
})

fe = FeatureEngineer()
X, y = fe.fit_transform(data, 'churn', num_cols, cat_cols)
trainer = ModelTrainer(model_type='xgboost', task='classification')
trainer.train(X, y)
os.makedirs('local_model', exist_ok=True)
trainer.save('local_model')
fe.save('local_model')
"

# Start the API
MODEL_DIR=local_model uvicorn src.serving.app:app --port 8000
```

### 4. Test the API

Open **http://localhost:8000/docs** for the interactive Swagger UI, or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "customer_id": 1,
      "total_transactions": 45,
      "total_spend": 2500.0,
      "avg_transaction_amount": 55.5,
      "days_since_last_transaction": 30,
      "unique_products": 12,
      "avg_days_between_transactions": 8.5,
      "transaction_count_trend": 0.2,
      "spend_trend": -0.1,
      "region": "US",
      "segment": "premium",
      "tenure_months": 24
    }
  }'

# Model info
curl http://localhost:8000/model-info
```

---

## GCP Deployment

### 1. Bootstrap GCP project

```bash
cp .env.example .env
# Edit .env with your GCP project details

bash scripts/setup_gcp.sh
```

### 2. Deploy infrastructure

```bash
cd terraform
terraform init
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

### 3. Deploy serving API

```bash
bash scripts/deploy_cloud_run.sh
```

### 4. Run the ML pipeline

```bash
python scripts/run_pipeline.py compile
python scripts/run_pipeline.py run
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health + model status |
| `GET` | `/model-info` | Model metadata and metrics |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/predict` | Single customer churn prediction |
| `POST` | `/predict/batch` | Batch predictions (multiple customers) |

---

## CI/CD Pipelines

| Workflow | Trigger | Actions |
|----------|---------|---------|
| **CI** (`ci.yml`) | Push / PR | Ruff lint, mypy type check, pytest |
| **CD Pipeline** (`cd-pipeline.yml`) | Push to `main` (src/pipeline, training, data, evaluation) | Compile + upload + trigger Vertex AI Pipeline |
| **CD Serving** (`cd-serving.yml`) | Push to `main` (src/serving) | Build Docker + deploy to Cloud Run |

---

## Monitoring & Drift Detection

The pipeline monitors model health in production using three statistical tests:

- **Kolmogorov-Smirnov (KS) test** — Detects distribution shifts in numerical features
- **Population Stability Index (PSI)** — Measures population distribution changes
- **Jensen-Shannon Divergence** — Symmetric measure of distribution similarity

When drift is detected in >30% of features, the system automatically triggers a retraining pipeline via Vertex AI.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Warehouse | BigQuery |
| ML Training | XGBoost, scikit-learn |
| Orchestration | Vertex AI Pipelines (Kubeflow v2) |
| Model Serving | FastAPI on Cloud Run |
| Infrastructure | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | Cloud Monitoring + custom drift detection |
| Model Registry | GCS with versioning |

---

## License

MIT
