from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from llm_peft_trainer.config import load_config, print_config
from llm_peft_trainer.eval import run_eval
from llm_peft_trainer.logging import configure_logging
from llm_peft_trainer.metrics import FAILURES, start_metrics_server
from llm_peft_trainer.mlflow_utils import log_params, log_runtime_metadata, start_run
from llm_peft_trainer.s3_io import S3IO
from llm_peft_trainer.train_hf import run_train_hf
from llm_peft_trainer.train_mlx import run_train_mlx

logger = logging.getLogger(__name__)


def package_adapter(config_path: str, merge: bool = False) -> Path:
    cfg = load_config(config_path)
    out_dir = Path(cfg.train.output_uri)
    adapter_dir = out_dir / "adapter"
    if merge:
        merged = out_dir / "merged-model"
        merged.mkdir(exist_ok=True)
        shutil.copytree(adapter_dir, merged / "adapter", dirs_exist_ok=True)
        (merged / "README.txt").write_text("Merged model placeholder; extend with merge logic.")
        return merged
    return adapter_dir


def main() -> None:
    configure_logging()
    start_metrics_server(8000)
    parser = argparse.ArgumentParser(description="LLM PEFT trainer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ["train", "eval", "package-adapter", "print-config"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "package-adapter":
            p.add_argument("--merge", action="store_true")

    args = parser.parse_args()

    if args.cmd == "print-config":
        print(print_config(args.config))
        return

    cfg = load_config(args.config)
    try:
        with start_run(cfg.mlflow.tracking_uri, cfg.mlflow.experiment, cfg.mlflow.run_name):
            log_params(cfg.model_dump())
            log_runtime_metadata("auto", cfg.runtime.docker_image, cfg.data.train_uri)
            if args.cmd == "train":
                out = run_train_mlx(cfg) if cfg.train.backend == "mlx" else run_train_hf(cfg)
                logger.info("training complete", extra={"ctx_output": str(out)})
            elif args.cmd == "eval":
                report = run_eval(cfg)
                logger.info("eval complete", extra={"ctx_report": str(report)})
            elif args.cmd == "package-adapter":
                artifact = package_adapter(args.config, merge=args.merge)
                logger.info("packaging complete", extra={"ctx_artifact": str(artifact)})

            if cfg.s3.bucket:
                s3 = S3IO(cfg.s3.bucket, cfg.s3.endpoint_url, cfg.s3.region)
                manifest = Path(cfg.train.output_uri) / "manifest.json"
                if manifest.exists() and cfg.s3.prefix:
                    key = f"{cfg.s3.prefix.rstrip('/')}/{cfg.runtime.run_id}/manifest.json"
                    s3.upload_file(manifest, key)
    except Exception:
        FAILURES.labels(backend=cfg.train.backend).inc()
        logger.exception("command failed")
        raise


if __name__ == "__main__":
    main()
