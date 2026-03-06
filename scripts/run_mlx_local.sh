#!/usr/bin/env bash
set -euo pipefail
python -m llm_peft_trainer.cli train --config configs/mlx_lora_m5_16gb.yaml
