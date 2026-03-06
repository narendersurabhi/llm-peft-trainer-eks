.PHONY: install lint type test smoke train-mlx validate-yaml

install:
	pip install -e .[dev]

lint:
	ruff check .

type:
	mypy src

test:
	pytest -q

smoke:
	bash scripts/run_hf_local_cpu_smoke.sh

train-mlx:
	bash scripts/run_mlx_local.sh

validate-yaml:
	python -c "import yaml, pathlib; [yaml.safe_load_all(pathlib.Path('k8s/argo/workflowtemplates.yaml').read_text())]"
