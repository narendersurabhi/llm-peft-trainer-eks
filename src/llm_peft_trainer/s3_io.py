from __future__ import annotations

import hashlib
from pathlib import Path

import boto3
from botocore.client import BaseClient


class S3IO:
    def __init__(self, bucket: str, endpoint_url: str | None = None, region: str | None = None) -> None:
        self.bucket = bucket
        self.client: BaseClient = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)

    def upload_file(self, local_path: str | Path, key: str) -> None:
        self.client.upload_file(str(local_path), self.bucket, key)

    def download_file(self, key: str, local_path: str | Path) -> None:
        self.client.download_file(self.bucket, key, str(local_path))

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
