variable "bucket_name" {
  description = "Name of the GCS bucket"
  type        = string
}

variable "location" {
  description = "Location of the bucket"
  type        = string
  default     = "europe-west1"
}

variable "force_destroy" {
  description = "Allow force destroy"
  type        = bool
  default     = false
}

variable "versioning" {
  description = "Enable versioning"
  type        = bool
  default     = true
}

variable "lifecycle_age" {
  description = "Delete objects older than N days (0 to disable)"
  type        = number
  default     = 0
}
