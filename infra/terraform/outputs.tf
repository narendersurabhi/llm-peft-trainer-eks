output "ecr_repo_url" {
  value = aws_ecr_repository.trainer.repository_url
}

output "artifact_bucket" {
  value = var.create_s3 ? aws_s3_bucket.artifacts[0].id : "disabled"
}

output "irsa_role_arn" {
  value = aws_iam_role.irsa.arn
}
