-- Feature engineering directement dans BigQuery (plus rapide pour de gros volumes)
WITH user_features AS (
    SELECT
        user_id,

        -- Features temporelles
        COUNT(*) AS total_transactions,
        COUNT(DISTINCT DATE(transaction_date)) AS active_days,
        DATE_DIFF(CURRENT_DATE(), MAX(DATE(transaction_date)), DAY) AS days_since_last,
        DATE_DIFF(MAX(DATE(transaction_date)), MIN(DATE(transaction_date)), DAY) AS customer_lifetime,

        -- Features monetaires
        SUM(amount) AS total_spend,
        AVG(amount) AS avg_spend,
        STDDEV(amount) AS std_spend,
        MAX(amount) AS max_spend,
        MIN(amount) AS min_spend,

        -- Features de diversite
        COUNT(DISTINCT category) AS category_diversity,
        COUNT(DISTINCT merchant_id) AS merchant_diversity,

        -- Features de tendance (30 derniers jours vs 30 jours precedents)
        SUM(CASE WHEN transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            THEN amount ELSE 0 END) AS spend_last_30d,
        SUM(CASE WHEN transaction_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)
                                        AND DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            THEN amount ELSE 0 END) AS spend_prev_30d,

        -- Target: churn (pas d'activite depuis 30 jours)
        CASE
            WHEN DATE_DIFF(CURRENT_DATE(), MAX(DATE(transaction_date)), DAY) > 30
            THEN 1 ELSE 0
        END AS is_churned

    FROM `{project_id}.{dataset_id}.transactions`
    WHERE transaction_date BETWEEN @start_date AND @end_date
    GROUP BY user_id
)

SELECT
    *,
    -- Feature derivee : tendance de depenses
    SAFE_DIVIDE(spend_last_30d, NULLIF(spend_prev_30d, 0)) AS spend_trend_ratio,

    -- Feature derivee : depense par jour actif
    SAFE_DIVIDE(total_spend, NULLIF(active_days, 0)) AS spend_per_active_day

FROM user_features
WHERE total_transactions >= 3  -- Filtre: au moins 3 transactions
