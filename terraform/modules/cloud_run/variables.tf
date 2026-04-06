variable "service_name" {
  description = "Cloud Run service name"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
}

variable "image" {
  description = "Docker image URI"
  type        = string
}

variable "service_account" {
  description = "Service account email"
  type        = string
}

variable "min_instances" {
  description = "Minimum number of instances"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of instances"
  type        = number
  default     = 10
}

variable "cpu" {
  description = "CPU limit"
  type        = string
  default     = "2"
}

variable "memory" {
  description = "Memory limit"
  type        = string
  default     = "2Gi"
}

variable "env_vars" {
  description = "Environment variables"
  type        = map(string)
  default     = {}
}
