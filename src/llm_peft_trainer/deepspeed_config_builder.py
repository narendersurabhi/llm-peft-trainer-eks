from __future__ import annotations


def build_deepspeed_config(
    stage: int = 2,
    gradient_accumulation_steps: int = 1,
    cpu_offload: bool = False,
    bf16: bool = True,
) -> dict:
    offload = {"device": "cpu", "pin_memory": True} if cpu_offload else {"device": "none"}
    return {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": {"enabled": bf16},
        "fp16": {"enabled": not bf16},
        "zero_optimization": {
            "stage": stage,
            "offload_optimizer": offload,
            "offload_param": offload,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }
