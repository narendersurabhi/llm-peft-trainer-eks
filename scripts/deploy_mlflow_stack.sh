#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f k8s/mlflow/namespace.yaml
kubectl apply -f k8s/mlflow/secrets.example.yaml
kubectl apply -f k8s/mlflow/postgres.yaml
kubectl apply -f k8s/mlflow/minio.yaml
kubectl apply -f k8s/mlflow/mlflow.yaml
kubectl apply -f k8s/mlflow/ingress.yaml
