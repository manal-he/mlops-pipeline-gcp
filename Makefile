.PHONY: install test lint format clean deploy-serving compile-pipeline run-pipeline terraform-init terraform-plan terraform-apply

# Installation
install:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Lint
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Format
format:
	black src/ tests/
	ruff check --fix src/ tests/

# Clean
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov coverage.xml

# Serving
deploy-serving:
	bash scripts/deploy_cloud_run.sh

# Pipeline
compile-pipeline:
	python scripts/run_pipeline.py compile

run-pipeline:
	python scripts/run_pipeline.py run --project=$(PROJECT_ID)

# Terraform
terraform-init:
	cd terraform && terraform init

terraform-plan:
	cd terraform && terraform plan -var-file=environments/$(ENV).tfvars

terraform-apply:
	cd terraform && terraform apply -var-file=environments/$(ENV).tfvars

# Setup GCP
setup-gcp:
	bash scripts/setup_gcp.sh
