#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG_PATH:=configs/hf_lora_single_node.yaml}"
argo submit --from workflowtemplate/llm-train -n argo \
  -p image="${IMAGE:-ghcr.io/org/llm-peft-trainer:latest}" \
  -p config_path="$CONFIG_PATH" \
  -p run_id="${RUN_ID:-manual-$(date +%s)}" \
  -p dataset_uri="${DATASET_URI:-s3://datasets/train.jsonl}" \
  -p output_uri="${OUTPUT_URI:-s3://llm-artifacts/runs}" \
  -p mlflow_uri="${MLFLOW_URI:-http://mlflow.mlops.svc.cluster.local:5000}" \
  -p backend="hf"
