# checkov:skip=CKV_DOCKER_2: SageMaker Processing container — lifecycle managed by SageMaker, not Docker orchestrator
# checkov:skip=CKV_DOCKER_3: SageMaker Processing container — requires root for /opt/ml/ mounts
# nosemgrep: dockerfile.security.missing-user-entrypoint.missing-user-entrypoint
FROM --platform=linux/amd64 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

ENV PYTHONUNBUFFERED=TRUE \
    LOCAL_OUTPUT_DIR=/opt/ml/processing/output

RUN apt-get update && rm -rf /var/lib/apt/lists/* && apt-get clean

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /opt/ml/processing/code/

# Install captioning-specific dependencies
COPY requirements.txt /tmp/captioning-req.txt
RUN uv pip install --system -r /tmp/captioning-req.txt \
    && rm /tmp/captioning-req.txt

# Install common dependencies
COPY common/requirements.txt /tmp/common-req.txt
RUN uv pip install --system -r /tmp/common-req.txt \
    && rm /tmp/common-req.txt

COPY . .

ENTRYPOINT ["python3", "main.py"]
