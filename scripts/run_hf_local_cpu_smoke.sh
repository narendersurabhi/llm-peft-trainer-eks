#!/usr/bin/env bash
set -euo pipefail
cat > /tmp/hf_smoke.yaml <<'YAML'
train:
  backend: hf
  output_uri: outputs/hf-smoke
  base_model: distilgpt2
  max_steps: 2
  save_steps: 1
  eval_steps: 1
  max_seq_len: 64
  micro_batch_size: 1
  grad_accum_steps: 1
  gradient_checkpointing: false
  lora_r: 4
  lora_alpha: 8
  target_modules: [c_attn]
  use_qlora: false
  use_deepspeed: false
  bf16: false
  fp16: false
data:
  train_uri: data/sample.jsonl
  eval_uri: data/sample.jsonl
  text_field: text
s3: {}
mlflow:
  tracking_uri: null
  experiment: smoke
runtime:
  run_id: smoke
  docker_image: local
YAML
python -m llm_peft_trainer.cli train --config /tmp/hf_smoke.yaml
