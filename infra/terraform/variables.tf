variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region for Cloud Run and compute"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP zone for the DB compute instance"
  type        = string
  default     = "us-central1-a"
}

variable "image_registry" {
  description = "Container image registry prefix (e.g. docker.io/myuser or ghcr.io/myorg)"
  type        = string
  default     = "docker.io"
}

variable "image_tag" {
  description = "Container image tag to deploy"
  type        = string
  default     = "latest"
}

variable "db_instance_name" {
  description = "Name of the GCE instance running TimescaleDB + Redis"
  type        = string
  default     = "omni-sense-db"
}

variable "enable_db_instance" {
  description = "Create the GCE e2-small DB instance (set to false if using managed DB)"
  type        = bool
  default     = true
}
