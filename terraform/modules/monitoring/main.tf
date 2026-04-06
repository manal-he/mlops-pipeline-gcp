resource "google_monitoring_alert_policy" "alerts" {
  for_each = var.alert_policies

  display_name = each.value.display_name
  combiner     = "OR"

  conditions {
    display_name = each.value.condition_display_name
    condition_threshold {
      filter          = each.value.filter
      comparison      = each.value.comparison
      threshold_value = each.value.threshold_value
      duration        = each.value.duration
      aggregations {
        alignment_period   = "300s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = var.notification_channels

  alert_strategy {
    auto_close = "1800s"
  }
}
