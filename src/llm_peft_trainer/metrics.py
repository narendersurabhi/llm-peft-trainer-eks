from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

from prometheus_client import Counter, Histogram, start_http_server

TRAIN_STEPS = Counter("trainer_train_steps_total", "Training steps processed", ["backend"])
EVAL_RUNS = Counter("trainer_eval_runs_total", "Evaluation runs", ["backend"])
FAILURES = Counter("trainer_failures_total", "Training failures", ["backend"])
STEP_SECONDS = Histogram("trainer_step_seconds", "Train step duration seconds", ["backend"])


def start_metrics_server(port: int = 8000) -> None:
    start_http_server(port)


@contextmanager
def timed_step(backend: str):
    start = perf_counter()
    try:
        yield
    finally:
        STEP_SECONDS.labels(backend=backend).observe(perf_counter() - start)
