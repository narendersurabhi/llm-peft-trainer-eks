from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class S3Config(BaseModel):
    endpoint_url: str | None = None
    bucket: str | None = None
    prefix: str = ""
    region: str | None = None


class DataConfig(BaseModel):
    train_uri: str
    eval_uri: str | None = None
    text_field: str = "text"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


class MLflowConfig(BaseModel):
    tracking_uri: str | None = None
    experiment: str = "llm-peft-trainer"
    run_name: str | None = None


class TrainConfig(BaseModel):
    backend: Literal["mlx", "hf"]
    output_uri: str
    base_model: str
    seed: int = 42
    max_steps: int = 200
    save_steps: int = 50
    eval_steps: int = 50
    learning_rate: float = 2e-4
    max_seq_len: int = 512
    micro_batch_size: int = 1
    grad_accum_steps: int = 16
    gradient_checkpointing: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    use_qlora: bool = False
    use_deepspeed: bool = False
    deepspeed_stage: int = 2
    deepspeed_cpu_offload: bool = False
    bf16: bool = True
    fp16: bool = False
    save_total_limit: int = 3

    @model_validator(mode="after")
    def validate_backend_constraints(self) -> "TrainConfig":
        if self.backend == "mlx" and (self.use_qlora or self.use_deepspeed):
            raise ValueError("MLX backend does not support QLoRA or DeepSpeed.")
        if self.use_qlora and self.backend != "hf":
            raise ValueError("QLoRA is only available for HF backend.")
        return self


class RuntimeConfig(BaseModel):
    run_id: str = "local"
    docker_image: str = "local-dev"
    resume: bool = True


class AppConfig(BaseModel):
    train: TrainConfig
    data: DataConfig
    s3: S3Config = Field(default_factory=S3Config)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("train")
    @classmethod
    def validate_mac_defaults(cls, v: TrainConfig) -> TrainConfig:
        if v.backend == "mlx" and v.micro_batch_size > 2:
            raise ValueError("For Apple 16GB defaults, keep micro_batch_size <= 2.")
        return v


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid config '{p}': {exc}") from exc


def print_config(path: str | Path) -> str:
    cfg = load_config(path)
    return cfg.model_dump_json(indent=2)
