# checkov:skip=CKV_DOCKER_2: SageMaker Processing container — lifecycle managed by SageMaker, not Docker orchestrator
# checkov:skip=CKV_DOCKER_3: SageMaker Processing container — requires root for /opt/ml/ mounts
# nosemgrep: dockerfile.security.missing-user-entrypoint.missing-user-entrypoint
FROM python:3.13-slim

RUN pip install --no-cache-dir uv

# Install step-specific dependencies
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Install common dependencies
COPY common/requirements.txt /tmp/common-req.txt
RUN uv pip install --system --no-cache -r /tmp/common-req.txt \
    && rm /tmp/common-req.txt

# Copy application code
COPY . /opt/ml/processing/code/

WORKDIR /opt/ml/processing/code

ENTRYPOINT ["python3", "main.py"]
