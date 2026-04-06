-- Extraction des donnees brutes de transactions
SELECT
    user_id,
    transaction_date,
    amount,
    category,
    merchant_id,
    payment_method,
    is_fraud
FROM `{project_id}.{dataset_id}.transactions`
WHERE transaction_date BETWEEN @start_date AND @end_date
ORDER BY transaction_date
