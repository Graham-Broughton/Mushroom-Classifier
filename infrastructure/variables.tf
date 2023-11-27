variable "gcp_region" {
  description = "Region in GCP"
  default = "us-central1"
}

variable "gcp_project_id" {
  description = "Project ID in GCP"
}

variable "gcp_credentials" {
  default = "./terraform-account.json"
}
variable "prefix" {
  description = "Prefix for the services"
  default = "terraform-mlops-project"
}

variable "credentials" {
  default = "./terraform-account.json"
}
variable "service-account" {
}

variable "model_bucket_name" {
  default = "mush-model-repo-tf"
}

variable "data_bucket_name" {
  default = "mush-img-repo-tf"
}