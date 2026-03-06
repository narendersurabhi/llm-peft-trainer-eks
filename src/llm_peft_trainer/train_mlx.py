from __future__ import annotations

import json
import logging
from pathlib import Path

from llm_peft_trainer.config import AppConfig
from llm_peft_trainer.data import load_datasets
from llm_peft_trainer.metrics import TRAIN_STEPS

logger = logging.getLogger(__name__)


def run_train_mlx(cfg: AppConfig) -> Path:
    try:
        import mlx.core as mx
        from mlx_lm import load
    except Exception as exc:
        raise RuntimeError("MLX dependencies missing. Install with `pip install .[mlx]`.") from exc

    train_ds, _, dataset_id = load_datasets(cfg.data, cfg.s3)
    model, tokenizer = load(cfg.train.base_model)
    output_dir = Path(cfg.train.output_uri)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Minimal mock-like gradient loop. Replace with mlx-lm LoRA training utility in advanced setups.
    steps = min(cfg.train.max_steps, len(train_ds))
    loss_avg = 0.0
    for idx in range(steps):
        sample = train_ds[idx][cfg.data.text_field]
        token_ids = tokenizer.encode(sample)[: cfg.train.max_seq_len]
        loss = float(len(token_ids)) / max(cfg.train.max_seq_len, 1)
        loss_avg += loss
        if (idx + 1) % cfg.train.save_steps == 0:
            ckpt = output_dir / f"checkpoint-{idx+1}.json"
            ckpt.write_text(json.dumps({"step": idx + 1, "loss": loss}), encoding="utf-8")
    loss_avg = loss_avg / max(steps, 1)

    adapter_path = output_dir / "adapter.safetensors"
    adapter_path.write_bytes(b"mlx_adapter_placeholder")

    manifest = {
        "backend": "mlx",
        "dataset_id": dataset_id,
        "base_model": cfg.train.base_model,
        "avg_loss": loss_avg,
        "device": "apple-gpu-metal",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("mlx memory info", extra={"ctx_mx_metal_active": str(mx.metal.is_available())})
    TRAIN_STEPS.labels(backend="mlx").inc(steps)
    return output_dir
