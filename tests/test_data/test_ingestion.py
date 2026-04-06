"""Tests pour l'ingestion de donnees (sans acces reel a BigQuery)."""

from pathlib import Path

from src.data.ingestion import BigQueryDataSource


class TestBigQueryDataSource:
    def test_default_extraction_query(self):
        source = BigQueryDataSource.__new__(BigQueryDataSource)
        source.project_id = "test-project"
        source.dataset_id = "test_dataset"

        query = source._default_extraction_query()
        assert "test-project" in query
        assert "test_dataset" in query
        assert "@start_date" in query
        assert "@end_date" in query

    def test_sql_files_exist(self):
        queries_dir = Path(__file__).parent.parent.parent / "src" / "data" / "queries"
        assert (queries_dir / "extract_raw_data.sql").exists()
        assert (queries_dir / "feature_engineering.sql").exists()
        assert (queries_dir / "data_quality_checks.sql").exists()

    def test_sql_files_have_content(self):
        queries_dir = Path(__file__).parent.parent.parent / "src" / "data" / "queries"
        for sql_file in queries_dir.glob("*.sql"):
            content = sql_file.read_text()
            assert len(content) > 0
            assert "SELECT" in content.upper()
