from __future__ import annotations

import os
from typing import Any

import mlflow


def start_run(tracking_uri: str | None, experiment: str, run_name: str | None = None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any]) -> None:
    safe_params = {k: str(v) for k, v in params.items()}
    mlflow.log_params(safe_params)


def log_runtime_metadata(git_sha: str, image_tag: str, dataset_id: str) -> None:
    mlflow.set_tags(
        {
            "git_sha": git_sha,
            "docker_image": image_tag,
            "dataset_id": dataset_id,
            "host": os.getenv("HOSTNAME", "unknown"),
        }
    )
