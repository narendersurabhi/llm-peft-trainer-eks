from llm_peft_trainer.config import load_config


def test_load_mlx_config() -> None:
    cfg = load_config("configs/mlx_lora_m5_16gb.yaml")
    assert cfg.train.backend == "mlx"
    assert cfg.train.micro_batch_size == 1


def test_load_hf_config() -> None:
    cfg = load_config("configs/hf_lora_single_node.yaml")
    assert cfg.train.backend == "hf"
    assert cfg.train.lora_r == 16
