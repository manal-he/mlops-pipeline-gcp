resource "google_bigquery_dataset" "dataset" {
  dataset_id    = var.dataset_id
  friendly_name = var.friendly_name
  description   = var.description
  location      = var.location
}

resource "google_bigquery_table" "transactions" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "transactions"

  time_partitioning {
    type  = "DAY"
    field = "transaction_date"
  }

  schema = jsonencode([
    { name = "user_id", type = "STRING", mode = "REQUIRED" },
    { name = "transaction_date", type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "amount", type = "FLOAT64", mode = "REQUIRED" },
    { name = "category", type = "STRING", mode = "NULLABLE" },
    { name = "merchant_id", type = "STRING", mode = "NULLABLE" },
    { name = "payment_method", type = "STRING", mode = "NULLABLE" },
  ])
}

resource "google_bigquery_table" "features" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "features"

  schema = jsonencode([
    { name = "user_id", type = "STRING", mode = "REQUIRED" },
    { name = "date", type = "DATE", mode = "NULLABLE" },
    { name = "total_transactions", type = "INT64", mode = "NULLABLE" },
    { name = "active_days", type = "INT64", mode = "NULLABLE" },
    { name = "days_since_last", type = "INT64", mode = "NULLABLE" },
    { name = "customer_lifetime", type = "INT64", mode = "NULLABLE" },
    { name = "total_spend", type = "FLOAT64", mode = "NULLABLE" },
    { name = "avg_spend", type = "FLOAT64", mode = "NULLABLE" },
    { name = "std_spend", type = "FLOAT64", mode = "NULLABLE" },
    { name = "max_spend", type = "FLOAT64", mode = "NULLABLE" },
    { name = "min_spend", type = "FLOAT64", mode = "NULLABLE" },
    { name = "category_diversity", type = "INT64", mode = "NULLABLE" },
    { name = "merchant_diversity", type = "INT64", mode = "NULLABLE" },
    { name = "spend_last_30d", type = "FLOAT64", mode = "NULLABLE" },
    { name = "spend_prev_30d", type = "FLOAT64", mode = "NULLABLE" },
    { name = "spend_trend_ratio", type = "FLOAT64", mode = "NULLABLE" },
    { name = "spend_per_active_day", type = "FLOAT64", mode = "NULLABLE" },
    { name = "is_churned", type = "INT64", mode = "NULLABLE" },
  ])
}
