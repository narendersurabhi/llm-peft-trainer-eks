from llm_peft_trainer.deepspeed_config_builder import build_deepspeed_config


def test_ds_builder() -> None:
    ds = build_deepspeed_config(stage=3, gradient_accumulation_steps=8, cpu_offload=True, bf16=True)
    assert ds["zero_optimization"]["stage"] == 3
    assert ds["gradient_accumulation_steps"] == 8
