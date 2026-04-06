-- Checks de qualite des donnees dans BigQuery

-- Check 1: Volume minimum de donnees
SELECT
    'volume_check' AS check_name,
    COUNT(*) AS row_count,
    CASE WHEN COUNT(*) >= 1000 THEN 'PASS' ELSE 'FAIL' END AS status
FROM `{project_id}.{dataset_id}.transactions`
WHERE transaction_date BETWEEN @start_date AND @end_date;

-- Check 2: Ratio de valeurs nulles
SELECT
    'null_check' AS check_name,
    COUNTIF(amount IS NULL) / COUNT(*) AS null_ratio,
    CASE WHEN COUNTIF(amount IS NULL) / COUNT(*) <= 0.1 THEN 'PASS' ELSE 'FAIL' END AS status
FROM `{project_id}.{dataset_id}.transactions`
WHERE transaction_date BETWEEN @start_date AND @end_date;

-- Check 3: Plage de valeurs
SELECT
    'range_check' AS check_name,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    CASE WHEN MIN(amount) >= 0 AND MAX(amount) <= 100000 THEN 'PASS' ELSE 'FAIL' END AS status
FROM `{project_id}.{dataset_id}.transactions`
WHERE transaction_date BETWEEN @start_date AND @end_date;

-- Check 4: Doublons
SELECT
    'duplicate_check' AS check_name,
    COUNT(*) - COUNT(DISTINCT CONCAT(user_id, CAST(transaction_date AS STRING), CAST(amount AS STRING))) AS duplicates,
    CASE
        WHEN (COUNT(*) - COUNT(DISTINCT CONCAT(user_id, CAST(transaction_date AS STRING), CAST(amount AS STRING)))) / COUNT(*) <= 0.05
        THEN 'PASS' ELSE 'FAIL'
    END AS status
FROM `{project_id}.{dataset_id}.transactions`
WHERE transaction_date BETWEEN @start_date AND @end_date;

-- Check 5: Freshness des donnees
SELECT
    'freshness_check' AS check_name,
    DATE_DIFF(CURRENT_DATE(), MAX(DATE(transaction_date)), DAY) AS days_since_latest,
    CASE
        WHEN DATE_DIFF(CURRENT_DATE(), MAX(DATE(transaction_date)), DAY) <= 7
        THEN 'PASS' ELSE 'WARN'
    END AS status
FROM `{project_id}.{dataset_id}.transactions`;
