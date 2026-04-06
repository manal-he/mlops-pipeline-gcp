resource "google_storage_bucket" "bucket" {
  name          = var.bucket_name
  location      = var.location
  force_destroy = var.force_destroy

  versioning {
    enabled = var.versioning
  }

  uniform_bucket_level_access = true

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_age > 0 ? [1] : []
    content {
      condition {
        age = var.lifecycle_age
      }
      action {
        type = "Delete"
      }
    }
  }
}
