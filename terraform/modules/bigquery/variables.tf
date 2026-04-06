variable "dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
}

variable "friendly_name" {
  description = "Friendly name for the dataset"
  type        = string
  default     = "ML Data"
}

variable "description" {
  description = "Description of the dataset"
  type        = string
  default     = "Dataset for ML pipeline data"
}

variable "location" {
  description = "Location of the dataset"
  type        = string
  default     = "europe-west1"
}
