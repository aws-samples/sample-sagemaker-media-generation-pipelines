# checkov:skip=CKV_DOCKER_2: Fallback Dockerfile — not used in production
# checkov:skip=CKV_DOCKER_3: Fallback Dockerfile — not used in production
# nosemgrep: dockerfile.security.missing-user-entrypoint.missing-user-entrypoint
# Fallback Dockerfile for DockerImageAsset when no ECR URI is provided.
# Not used in production — all steps use pre-built ECR images.
FROM python:3.13-slim
COPY . /opt/ml/processing/code/
WORKDIR /opt/ml/processing/code
