#!/usr/bin/env bash
set -euo pipefail
: "${IMAGE_REPO:?Set IMAGE_REPO, e.g. <acct>.dkr.ecr.<region>.amazonaws.com/llm-peft-trainer}"
: "${IMAGE_TAG:=latest}"
docker build -f docker/Dockerfile -t "${IMAGE_REPO}:${IMAGE_TAG}" .
docker push "${IMAGE_REPO}:${IMAGE_TAG}"
