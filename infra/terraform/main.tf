terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

resource "aws_ecr_repository" "trainer" {
  name                 = "llm-peft-trainer"
  image_tag_mutability = "MUTABLE"
}

resource "aws_s3_bucket" "artifacts" {
  count  = var.create_s3 ? 1 : 0
  bucket = var.artifact_bucket_name
}

resource "aws_iam_role" "irsa" {
  name               = "llm-peft-trainer-irsa"
  assume_role_policy = data.aws_iam_policy_document.irsa_assume_role.json
}

data "aws_iam_policy_document" "irsa_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [var.oidc_provider_arn]
    }
    condition {
      test     = "StringLike"
      variable = "${var.oidc_provider_url}:sub"
      values   = ["system:serviceaccount:argo:*"]
    }
  }
}
