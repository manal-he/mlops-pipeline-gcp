FROM python:3.11-slim

WORKDIR /app

# Copier les requirements
COPY src/serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/

# Port Cloud Run
ENV PORT=8080

# Commande de lancement
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
