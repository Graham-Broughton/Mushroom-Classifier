variable "bucket_name" {
  default = "mush-img-repo-tf"
}

variable "service-account" {
  type = string
  description = "Service Account associated with the whole function service"
}