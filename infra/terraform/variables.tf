variable "region" {
  type    = string
  default = "us-east-1"
}

variable "create_s3" {
  type    = bool
  default = true
}

variable "artifact_bucket_name" {
  type    = string
  default = "llm-peft-trainer-artifacts"
}

variable "oidc_provider_arn" {
  type = string
}

variable "oidc_provider_url" {
  type = string
}
