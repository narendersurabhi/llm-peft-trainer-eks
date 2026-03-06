from pathlib import Path

import yaml


def test_argo_templates_parse() -> None:
    docs = list(yaml.safe_load_all(Path("k8s/argo/workflowtemplates.yaml").read_text(encoding="utf-8")))
    assert len(docs) >= 2
    assert all("kind" in d for d in docs if d)
