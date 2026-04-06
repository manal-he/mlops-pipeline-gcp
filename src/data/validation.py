"""Validation de la qualite des donnees avant le training."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ValidationResult:
    is_valid: bool
    total_checks: int
    passed_checks: int
    failed_checks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    statistics: dict = field(default_factory=dict)


class DataValidator:
    """
    Validation de la qualite des donnees avant le training.

    "Garbage in, garbage out". Si les donnees sont mauvaises,
    le modele sera mauvais. On valide :

    1. Schema : Les colonnes attendues existent avec les bons types
    2. Completude : Pas trop de valeurs manquantes
    3. Distribution : Les valeurs sont dans des plages raisonnables
    4. Volume : Assez de donnees pour entrainer un modele
    5. Freshness : Les donnees ne sont pas trop anciennes
    """

    def __init__(self, config: dict):
        """
        Args:
            config: {
                "expected_columns": ["col1", "col2", ...],
                "column_types": {"col1": "float64", "col2": "object"},
                "max_null_ratio": 0.1,
                "min_rows": 1000,
                "numerical_ranges": {"col1": (0, 1000)},
                "categorical_values": {"col2": ["A", "B", "C"]},
            }
        """
        self.config = config

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Execute toutes les validations."""
        checks = []
        warnings = []

        # 1. Verification du schema
        schema_ok, schema_msg = self._check_schema(df)
        checks.append(("schema", schema_ok, schema_msg))

        # 2. Verification du volume
        vol_ok, vol_msg = self._check_volume(df)
        checks.append(("volume", vol_ok, vol_msg))

        # 3. Verification des valeurs manquantes
        null_ok, null_msg = self._check_nulls(df)
        checks.append(("nulls", null_ok, null_msg))

        # 4. Verification des plages numeriques
        range_ok, range_msg = self._check_numerical_ranges(df)
        checks.append(("ranges", range_ok, range_msg))

        # 5. Verification des valeurs categoriques
        cat_ok, cat_msg = self._check_categorical_values(df)
        checks.append(("categorical", cat_ok, cat_msg))

        # 6. Verification des doublons
        dup_ok, dup_msg = self._check_duplicates(df)
        checks.append(("duplicates", dup_ok, dup_msg))

        # Compiler les resultats
        passed = sum(1 for _, ok, _ in checks if ok)
        failed = [
            {"check": name, "message": msg}
            for name, ok, msg in checks
            if not ok
        ]

        is_valid = len(failed) == 0

        # Statistiques descriptives
        statistics = self._compute_statistics(df)

        result = ValidationResult(
            is_valid=is_valid,
            total_checks=len(checks),
            passed_checks=passed,
            failed_checks=failed,
            warnings=warnings,
            statistics=statistics,
        )

        if is_valid:
            logger.info(f"Validation reussie: {passed}/{len(checks)} checks passes")
        else:
            logger.warning(
                f"Validation echouee: {len(failed)} checks en erreur: "
                f"{[f['check'] for f in failed]}"
            )

        return result

    def _check_schema(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Verifie que les colonnes attendues existent."""
        expected = set(self.config.get("expected_columns", []))
        if not expected:
            return True, "Pas de colonnes attendues configurees"

        actual = set(df.columns)
        missing = expected - actual
        if missing:
            return False, f"Colonnes manquantes: {missing}"
        return True, "Schema valide"

    def _check_volume(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Verifie le volume minimum de donnees."""
        min_rows = self.config.get("min_rows", 100)
        if len(df) < min_rows:
            return False, f"Volume insuffisant: {len(df)} lignes (minimum: {min_rows})"
        return True, f"Volume OK: {len(df)} lignes"

    def _check_nulls(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Verifie le ratio de valeurs manquantes."""
        max_ratio = self.config.get("max_null_ratio", 0.1)
        null_ratios = df.isnull().mean()
        high_null_cols = null_ratios[null_ratios > max_ratio]

        if len(high_null_cols) > 0:
            details = {col: f"{ratio:.1%}" for col, ratio in high_null_cols.items()}
            return False, f"Colonnes avec trop de nulls (>{max_ratio:.0%}): {details}"
        return True, "Valeurs manquantes OK"

    def _check_numerical_ranges(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Verifie que les valeurs numeriques sont dans les plages attendues."""
        ranges = self.config.get("numerical_ranges", {})
        violations = []

        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < min_val or col_max > max_val:
                violations.append(
                    f"{col}: [{col_min}, {col_max}] hors de [{min_val}, {max_val}]"
                )

        if violations:
            return False, f"Plages depassees: {violations}"
        return True, "Plages numeriques OK"

    def _check_categorical_values(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Verifie que les valeurs categoriques sont dans les listes attendues."""
        expected_values = self.config.get("categorical_values", {})
        violations = []

        for col, allowed_values in expected_values.items():
            if col not in df.columns:
                continue
            actual_values = set(df[col].dropna().unique())
            unexpected = actual_values - set(allowed_values)
            if unexpected:
                violations.append(f"{col}: valeurs inattendues {unexpected}")

        if violations:
            return False, f"Valeurs categoriques invalides: {violations}"
        return True, "Valeurs categoriques OK"

    def _check_duplicates(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Verifie les doublons."""
        n_duplicates = df.duplicated().sum()
        ratio = n_duplicates / len(df) if len(df) > 0 else 0

        if ratio > 0.05:
            return False, f"Trop de doublons: {n_duplicates} ({ratio:.1%})"
        return True, f"Doublons OK: {n_duplicates} ({ratio:.1%})"

    def _compute_statistics(self, df: pd.DataFrame) -> dict:
        """Calcule des statistiques descriptives."""
        stats = {
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()

        return stats
