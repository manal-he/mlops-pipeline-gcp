"""Versioning et gestion des modeles sur GCS."""

import json
from datetime import datetime
from pathlib import Path

from google.cloud import storage
from loguru import logger


class ModelRegistry:
    """
    Gestion du versioning des modeles sur Google Cloud Storage.

    Chaque modele est stocke avec :
    - Un numero de version (v1, v2, ...)
    - Ses metadonnees (metriques, hyperparametres, date)
    - Un lien symbolique "latest" vers la derniere version deployee
    """

    def __init__(self, project_id: str, bucket_name: str, prefix: str = "models"):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)

    def register_model(
        self,
        local_model_dir: str,
        metrics: dict,
        metadata: dict = None,
        version: str = None,
    ) -> str:
        """
        Enregistre un nouveau modele dans le registry.

        Args:
            local_model_dir: Repertoire local contenant le modele
            metrics: Metriques de performance
            metadata: Metadonnees additionnelles
            version: Version explicite (auto-incrementee si None)

        Returns:
            GCS URI du modele enregistre
        """
        if version is None:
            version = self._next_version()

        gcs_prefix = f"{self.prefix}/{version}"
        logger.info(f"Enregistrement du modele version {version}...")

        # Upload tous les fichiers du repertoire
        local_path = Path(local_model_dir)
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                blob_name = f"{gcs_prefix}/{relative_path}"
                blob = self.bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))

        # Sauvegarder les infos de version
        version_info = {
            "version": version,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {},
            "gcs_uri": f"gs://{self.bucket_name}/{gcs_prefix}/",
        }

        version_blob = self.bucket.blob(f"{gcs_prefix}/version_info.json")
        version_blob.upload_from_string(
            json.dumps(version_info, indent=2, default=str),
            content_type="application/json",
        )

        gcs_uri = f"gs://{self.bucket_name}/{gcs_prefix}/"
        logger.info(f"Modele enregistre: {gcs_uri}")
        return gcs_uri

    def promote_to_latest(self, version: str) -> str:
        """Promouvoir une version comme 'latest'."""
        source_prefix = f"{self.prefix}/{version}"
        latest_prefix = f"{self.prefix}/latest"

        # Supprimer l'ancien latest
        blobs = list(self.bucket.list_blobs(prefix=f"{latest_prefix}/"))
        for blob in blobs:
            blob.delete()

        # Copier la version vers latest
        source_blobs = list(self.bucket.list_blobs(prefix=f"{source_prefix}/"))
        for blob in source_blobs:
            new_name = blob.name.replace(source_prefix, latest_prefix, 1)
            self.bucket.copy_blob(blob, self.bucket, new_name)

        latest_uri = f"gs://{self.bucket_name}/{latest_prefix}/"
        logger.info(f"Version {version} promue comme latest: {latest_uri}")
        return latest_uri

    def list_versions(self) -> list[dict]:
        """Liste toutes les versions de modeles."""
        versions = []
        blobs = self.bucket.list_blobs(prefix=f"{self.prefix}/")

        seen_versions = set()
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 2:
                version = parts[1]
                if version not in seen_versions and version != "latest":
                    seen_versions.add(version)
                    # Charger les infos de version
                    info_blob = self.bucket.blob(
                        f"{self.prefix}/{version}/version_info.json"
                    )
                    if info_blob.exists():
                        info = json.loads(info_blob.download_as_text())
                        versions.append(info)

        return sorted(versions, key=lambda x: x.get("registered_at", ""), reverse=True)

    def get_latest_metrics(self) -> dict:
        """Retourne les metriques du modele latest."""
        info_blob = self.bucket.blob(f"{self.prefix}/latest/version_info.json")
        if info_blob.exists():
            return json.loads(info_blob.download_as_text()).get("metrics", {})
        return {}

    def _next_version(self) -> str:
        """Calcule le prochain numero de version."""
        versions = self.list_versions()
        if not versions:
            return "v1"

        max_version = 0
        for v in versions:
            ver_str = v.get("version", "v0")
            try:
                num = int(ver_str.replace("v", ""))
                max_version = max(max_version, num)
            except ValueError:
                continue

        return f"v{max_version + 1}"
