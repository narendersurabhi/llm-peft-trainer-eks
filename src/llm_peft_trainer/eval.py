from __future__ import annotations

import json
import math
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from llm_peft_trainer.config import AppConfig
from llm_peft_trainer.data import load_datasets
from llm_peft_trainer.metrics import EVAL_RUNS


def run_eval(cfg: AppConfig) -> Path:
    _, eval_ds, _ = load_datasets(cfg.data, cfg.s3)
    if eval_ds is None:
        raise ValueError("Eval dataset is required for eval command.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.train.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.train.base_model)

    tokenized_eval = eval_ds.map(
        lambda x: tokenizer(x[cfg.data.text_field], truncation=True, max_length=cfg.train.max_seq_len),
        batched=True,
    )

    args = TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=1, report_to=[])
    trainer = Trainer(model=model, args=args, eval_dataset=tokenized_eval)
    out = trainer.evaluate()
    loss = float(out.get("eval_loss", 0.0))
    ppl = math.exp(loss) if loss < 20 else float("inf")

    report = {"eval_loss": loss, "perplexity": ppl}
    output_dir = Path(cfg.train.output_uri)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "eval_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    EVAL_RUNS.labels(backend=cfg.train.backend).inc()
    return path
