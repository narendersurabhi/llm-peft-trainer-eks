#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG_PATH:=configs/hf_lora_single_node.yaml}"
argo submit --from workflowtemplate/llm-eval -n argo \
  -p image="${IMAGE:-ghcr.io/org/llm-peft-trainer:latest}" \
  -p config_path="$CONFIG_PATH" \
  -p run_id="${RUN_ID:-eval-$(date +%s)}" \
  -p dataset_uri="${DATASET_URI:-s3://datasets/eval.jsonl}" \
  -p output_uri="${OUTPUT_URI:-s3://llm-artifacts/eval}" \
  -p mlflow_uri="${MLFLOW_URI:-http://mlflow.mlops.svc.cluster.local:5000}" \
  -p backend="hf"
