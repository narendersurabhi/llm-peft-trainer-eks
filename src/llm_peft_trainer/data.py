from __future__ import annotations

import tempfile
from pathlib import Path

import jsonlines
from datasets import Dataset

from llm_peft_trainer.config import DataConfig, S3Config
from llm_peft_trainer.s3_io import S3IO, file_sha256


def _read_jsonl(path: Path, text_field: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with jsonlines.open(path) as reader:
        for row in reader:
            text = row.get(text_field)
            if text:
                rows.append({text_field: str(text)})
    return rows


def _download_if_s3(uri: str, s3: S3Config) -> Path:
    if not uri.startswith("s3://"):
        return Path(uri)
    if not s3.bucket:
        raise ValueError("S3 bucket must be set in config.s3.bucket for s3:// URIs")
    key = uri.removeprefix("s3://")
    tmp = Path(tempfile.mkdtemp()) / Path(key).name
    S3IO(bucket=s3.bucket, endpoint_url=s3.endpoint_url, region=s3.region).download_file(key, tmp)
    return tmp


def load_datasets(data: DataConfig, s3: S3Config) -> tuple[Dataset, Dataset | None, str]:
    train_path = _download_if_s3(data.train_uri, s3)
    train_rows = _read_jsonl(train_path, data.text_field)
    if data.max_train_samples:
        train_rows = train_rows[: data.max_train_samples]
    train = Dataset.from_list(train_rows)

    eval_ds = None
    if data.eval_uri:
        eval_path = _download_if_s3(data.eval_uri, s3)
        eval_rows = _read_jsonl(eval_path, data.text_field)
        if data.max_eval_samples:
            eval_rows = eval_rows[: data.max_eval_samples]
        eval_ds = Dataset.from_list(eval_rows)

    dataset_id = file_sha256(train_path) if train_path.exists() else data.train_uri
    return train, eval_ds, dataset_id
