# AGENTS.md

## Repository guidance
- This repository implements two LLM fine-tuning backends:
  - MLX backend for Apple Silicon local development.
  - HF/Accelerate backend for EKS NVIDIA production training.
- Keep this file updated whenever structural or major functional changes are made.
- Prefer config-driven behavior through files in `configs/`.
- Ensure CI (`.github/workflows/ci.yaml`) remains green for Python lint/test/smoke validation.

## Current status
- Initial production scaffold created for training, eval, packaging, MLflow on K8s, Argo templates, and Terraform bootstrap.
- Includes sample JSONL dataset and CPU smoke pipeline.

## Change log
- Added Python package `src/llm_peft_trainer` with CLI, config validation, data loading, S3 IO, metrics, logging, MLflow helpers, MLX and HF training paths, eval logic, DeepSpeed config builder, and K8s env helpers.
- Added configuration profiles for MLX M5 16GB, HF single-node LoRA/QLoRA, HF multinode LoRA, and HF DeepSpeed CPU offload.
- Added Docker runtime for EKS CUDA training.
- Added K8s manifests for MLflow stack (namespace, secrets example, postgres, minio, mlflow, ingress).
- Added Argo workflow templates and a multinode rendezvous example.
- Added Terraform minimal infrastructure for ECR, optional S3, and IRSA role.
- Added helper scripts for local runs, image build/push, MLflow deployment, and Argo submissions.
- Added tests for config loading, DeepSpeed builder, and Argo YAML parsing.
- Added GitHub Actions CI pipeline for lint/type/test/smoke/yaml checks.
- Added Makefile and README with architecture, quickstarts, multinode instructions, and troubleshooting.
