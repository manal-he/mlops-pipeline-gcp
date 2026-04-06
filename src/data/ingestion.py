"""Extraction de donnees depuis BigQuery."""

from pathlib import Path

import pandas as pd
from google.cloud import bigquery, storage
from loguru import logger


class BigQueryDataSource:
    """
    Extraction de donnees depuis BigQuery.

    BigQuery est un data warehouse serverless. On ecrit des requetes SQL
    pour extraire et transformer les donnees. BigQuery est tres rapide
    pour scanner des TB de donnees car il utilise un format de stockage
    en colonnes (Capacitor) et une execution distribuee.

    Bonnes pratiques :
    - Toujours partitionner les tables par date (cout reduit)
    - Utiliser des tables de staging pour les transformations complexes
    - Versionner les requetes SQL dans le code source
    """

    def __init__(self, project_id: str, dataset_id: str):
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id

    def extract_training_data(
        self,
        query_path: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        Extrait les donnees d'entrainement depuis BigQuery.

        Args:
            query_path: Chemin vers le fichier SQL
            start_date: Date de debut (format YYYY-MM-DD)
            end_date: Date de fin (format YYYY-MM-DD)

        Returns:
            DataFrame avec les donnees extraites
        """
        if query_path:
            query = Path(query_path).read_text()
        else:
            query = self._default_extraction_query()

        # Remplacer les parametres dans la requete
        if start_date and end_date:
            query = query.replace("@start_date", f"'{start_date}'")
            query = query.replace("@end_date", f"'{end_date}'")

        query = query.replace("{project_id}", self.project_id)
        query = query.replace("{dataset_id}", self.dataset_id)

        logger.info(f"Execution de la requete BigQuery ({self.dataset_id})...")

        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE,
        )

        df = self.client.query(query, job_config=job_config).to_dataframe()
        logger.info(f"Donnees extraites: {len(df)} lignes, {len(df.columns)} colonnes")

        return df

    def extract_with_feature_engineering(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        """Extrait les donnees avec feature engineering SQL."""
        query_path = str(
            Path(__file__).parent / "queries" / "feature_engineering.sql"
        )
        return self.extract_training_data(
            query_path=query_path,
            start_date=start_date,
            end_date=end_date,
        )

    def save_to_gcs(
        self,
        df: pd.DataFrame,
        bucket_name: str,
        destination_path: str,
    ) -> str:
        """Sauvegarde un DataFrame en CSV sur GCS."""
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_path)

        blob.upload_from_string(
            df.to_csv(index=False),
            content_type="text/csv",
        )

        gcs_uri = f"gs://{bucket_name}/{destination_path}"
        logger.info(f"Donnees sauvegardees sur GCS: {gcs_uri}")
        return gcs_uri

    def _default_extraction_query(self) -> str:
        """Requete d'extraction par defaut."""
        return f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.features`
        WHERE date BETWEEN @start_date AND @end_date
        """

    def load_from_gcs(self, bucket_name: str, source_path: str) -> pd.DataFrame:
        """Charge un DataFrame depuis un CSV sur GCS."""
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_path)

        csv_data = blob.download_as_text()
        df = pd.read_csv(pd.io.common.StringIO(csv_data))
        logger.info(f"Donnees chargees depuis GCS: {len(df)} lignes")
        return df

    def get_table_info(self, table_id: str) -> dict:
        """Retourne les informations d'une table BigQuery."""
        table_ref = f"{self.project_id}.{self.dataset_id}.{table_id}"
        table = self.client.get_table(table_ref)
        return {
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes,
            "schema": [
                {"name": field.name, "type": field.field_type}
                for field in table.schema
            ],
            "created": table.created.isoformat(),
            "modified": table.modified.isoformat(),
        }
