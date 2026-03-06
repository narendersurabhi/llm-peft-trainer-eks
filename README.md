# llm-peft-trainer-eks

Production-grade dual-backend LLM PEFT training repo:
- **Backend A (Apple Silicon / Metal):** MLX LoRA dev loop (no CUDA, no NCCL, no DeepSpeed).
- **Backend B (EKS / NVIDIA):** Transformers + Accelerate + PEFT LoRA/QLoRA + optional DeepSpeed ZeRO.

## Architecture

```text
                  ┌──────────────────────────────────────────────────────┐
                  │                  GitHub Actions CI                  │
                  │  ruff • mypy • pytest • CPU smoke • YAML checks     │
                  └───────────────────────┬──────────────────────────────┘
                                          │
┌─────────────────────────────────────────┴───────────────────────────────────────────┐
│                               llm-peft-trainer CLI                                 │
│    train | eval | package-adapter | print-config (Pydantic validated YAML)         │
└───────────────────────┬──────────────────────────────────────────┬──────────────────┘
                        │                                          │
            ┌───────────▼───────────┐                    ┌────────▼─────────────────┐
            │ Backend A: MLX (M5)   │                    │ Backend B: HF + CUDA     │
            │ train_mlx.py          │                    │ train_hf.py              │
            │ micro-batch=1         │                    │ LoRA / QLoRA / DeepSpeed │
            │ grad_accum=16         │                    │ accelerate / torchrun     │
            └───────────┬───────────┘                    └────────┬─────────────────┘
                        │                                           │
                        └──────────────┬────────────────────────────┘
                                       │
                           ┌───────────▼─────────────────────┐
                           │ Storage + Tracking              │
                           │ S3/MinIO checkpoints/artifacts  │
                           │ MLflow tracking on EKS          │
                           └─────────────────────────────────┘
```

## Mac Quickstart (MLX)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mlx]"
python -m llm_peft_trainer.cli print-config --config configs/mlx_lora_m5_16gb.yaml
bash scripts/run_mlx_local.sh
```

Memory-safe defaults for 16 GB unified memory:
- `micro_batch_size=1`
- `grad_accum_steps=16` (increase to 32 if OOM)
- `max_seq_len=512` (drop to 256 if OOM)
- `lora_r=8`

## EKS Quickstart (Argo + MLflow + MinIO)

```bash
pip install -e ".[dev]"
bash scripts/deploy_mlflow_stack.sh
kubectl -n argo apply -f k8s/argo/workflowtemplates.yaml
bash scripts/submit_argo_train.sh
bash scripts/submit_argo_eval.sh
```

### Swap MinIO for AWS S3
- In `k8s/mlflow/mlflow.yaml`, remove `MLFLOW_S3_ENDPOINT_URL` and set IAM/IRSA permissions.
- Update training configs: set `s3.endpoint_url: null`, `s3.bucket: <aws-bucket>`.

## Multi-node enablement

1. Apply rendezvous service and workflow example:
```bash
kubectl apply -f k8s/argo/multinode-example.yaml
```
2. Ensure nodegroup has GPU capacity for `WORLD_SIZE` workers.
3. Submit workflow:
```bash
argo submit -n argo k8s/argo/multinode-example.yaml
```

## Smoke test (CPU)

```bash
bash scripts/run_hf_local_cpu_smoke.sh
```

Uses `distilgpt2`, 2 steps, CPU-only.

## Checkpoint & Resume
- HF backend resumes if `output_uri/checkpoint-*` exists.
- MLX backend saves periodic checkpoint JSON stubs.
- Final artifacts include `manifest.json` for reproducibility metadata.

## Troubleshooting OOM playbook
1. Reduce `max_seq_len` (1024 -> 512 -> 256).
2. Increase `grad_accum_steps` while keeping micro-batch low.
3. Reduce `lora_r` (16 -> 8).
4. Enable `gradient_checkpointing`.
5. Lower eval workload (`max_eval_samples`, batch size).

## Repo layout

- `src/llm_peft_trainer/`: trainer code
- `configs/`: production and dev YAML configs
- `k8s/mlflow/`: MLflow + Postgres + MinIO manifests
- `k8s/argo/`: train/eval templates + multinode example
- `infra/terraform/`: ECR/S3/IRSA minimal infra
- `scripts/`: local run, deploy, submit helpers

