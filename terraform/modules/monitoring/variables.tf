variable "alert_policies" {
  description = "Map of alert policies"
  type = map(object({
    display_name           = string
    condition_display_name = string
    filter                 = string
    comparison             = string
    threshold_value        = number
    duration               = string
  }))
  default = {}
}

variable "notification_channels" {
  description = "Notification channel IDs"
  type        = list(string)
  default     = []
}
