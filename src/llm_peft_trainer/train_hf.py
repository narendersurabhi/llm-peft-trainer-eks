from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from llm_peft_trainer.config import AppConfig
from llm_peft_trainer.data import load_datasets
from llm_peft_trainer.deepspeed_config_builder import build_deepspeed_config
from llm_peft_trainer.metrics import TRAIN_STEPS

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603,S607
        )
    except Exception:
        return "unknown"


def _tokenize(examples: dict, tokenizer, text_field: str, max_seq_len: int):
    return tokenizer(
        examples[text_field],
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
    )


def run_train_hf(cfg: AppConfig) -> Path:
    train_ds, eval_ds, dataset_id = load_datasets(cfg.data, cfg.s3)
    tokenizer = AutoTokenizer.from_pretrained(cfg.train.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.train.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.train.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float16,
    )
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.train.lora_r,
        lora_alpha=cfg.train.lora_alpha,
        lora_dropout=cfg.train.lora_dropout,
        target_modules=cfg.train.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    tokenized_train = train_ds.map(
        lambda x: _tokenize(x, tokenizer, cfg.data.text_field, cfg.train.max_seq_len),
        batched=True,
    )
    tokenized_eval = (
        eval_ds.map(lambda x: _tokenize(x, tokenizer, cfg.data.text_field, cfg.train.max_seq_len), batched=True)
        if eval_ds
        else None
    )

    output_dir = Path(cfg.train.output_uri)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_config_path = None
    if cfg.train.use_deepspeed:
        ds_config = build_deepspeed_config(
            stage=cfg.train.deepspeed_stage,
            gradient_accumulation_steps=cfg.train.grad_accum_steps,
            cpu_offload=cfg.train.deepspeed_cpu_offload,
            bf16=cfg.train.bf16,
        )
        ds_config_path = output_dir / "deepspeed_config.json"
        ds_config_path.write_text(json.dumps(ds_config, indent=2), encoding="utf-8")

    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.micro_batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum_steps,
        logging_steps=10,
        save_steps=cfg.train.save_steps,
        eval_steps=cfg.train.eval_steps,
        evaluation_strategy="steps" if tokenized_eval is not None else "no",
        learning_rate=cfg.train.learning_rate,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        seed=cfg.train.seed,
        save_total_limit=cfg.train.save_total_limit,
        deepspeed=str(ds_config_path) if ds_config_path else None,
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )

    resume = cfg.runtime.resume and any(output_dir.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir / "adapter"))

    manifest = {
        "git_sha": _git_sha(),
        "docker_image": cfg.runtime.docker_image,
        "dataset_id": dataset_id,
        "base_model": cfg.train.base_model,
        "backend": cfg.train.backend,
        "use_qlora": cfg.train.use_qlora,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    TRAIN_STEPS.labels(backend="hf").inc(cfg.train.max_steps)

    if torch.cuda.is_available():
        logger.info("gpu memory", extra={"ctx_allocated": int(torch.cuda.memory_allocated())})
        if os.system("which nvidia-smi >/dev/null 2>&1") == 0:
            os.system("nvidia-smi --query-gpu=memory.used,memory.total --format=csv")

    return output_dir
