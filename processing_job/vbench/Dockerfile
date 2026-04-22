# checkov:skip=CKV_DOCKER_2: SageMaker Processing container — lifecycle managed by SageMaker, not Docker orchestrator
# checkov:skip=CKV_DOCKER_3: SageMaker Processing container — requires root for /opt/ml/ mounts
# nosemgrep: dockerfile.security.missing-user-entrypoint.missing-user-entrypoint
FROM --platform=linux/amd64 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin/:$PATH"

# Clone VBench and install its deps in a single layer
WORKDIR /opt/ml/processing/code/
RUN git clone --depth 1 https://github.com/Vchitect/VBench.git \
    && cd VBench \
    && git fetch --depth 1 origin dc62783c0fb4fd333249c0b669027fe102696682 \
    && git checkout dc62783c0fb4fd333249c0b669027fe102696682 \
    && sed -i '/transformers==4.33.2/d' requirements.txt \
    && uv pip install --system --only-binary=tokenizers -r requirements.txt

# Clone dino repo for subject_consistency (torch.hub expects it at CACHE_DIR/dino_model/facebookresearch_dino_main/)
RUN git clone --depth 1 https://github.com/facebookresearch/dino.git /root/.cache/vbench/dino_model/facebookresearch_dino_main

# Copy requirements first for layer caching
COPY common/requirements.txt /tmp/common-requirements.txt
RUN uv pip install --system -r /tmp/common-requirements.txt \
    && rm /tmp/common-requirements.txt

# Copy application code last
COPY . /opt/ml/processing/code
WORKDIR /opt/ml/processing/code

ENTRYPOINT ["python3"]
