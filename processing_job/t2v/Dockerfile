# checkov:skip=CKV_DOCKER_2: SageMaker Processing container — lifecycle managed by SageMaker, not Docker orchestrator
# checkov:skip=CKV_DOCKER_3: SageMaker Processing container — requires root for /opt/ml/ mounts
# nosemgrep: dockerfile.security.missing-user-entrypoint.missing-user-entrypoint
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HOME=/opt/ml/processing/code/ \
    COMFY_HOME=/opt/ml/processing/code/ComfyUI \
    LOCAL_OUTPUT_DIR=/opt/ml/processing/output \
    COMFY_TRACKING_DISABLED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 python3-pip python3-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/opt/ml/processing/code/.local/bin:$PATH"

WORKDIR $HOME

# Install ComfyUI + ComfyScript
RUN uv pip install --system --no-cache comfy-cli==1.5.4 \
    && comfy --skip-prompt --workspace=$COMFY_HOME install --nvidia \
    && cd $COMFY_HOME/custom_nodes \
    && git clone --depth 1 https://github.com/Chaoses-Ib/ComfyScript.git \
    && cd ComfyScript \
    && uv pip install --system --no-cache -e ".[default]"

# Force PyTorch to CUDA 12.8 after comfy install — ml.g5 driver supports CUDA 12.8
RUN uv pip install --system --no-cache \
    --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Models are downloaded from S3 as a SageMaker ProcessingInput
# and symlinked into ComfyUI models dir by the entrypoint script

# Copy requirements first for layer caching
COPY common/requirements.txt /tmp/common-req.txt
COPY common/videogen-requirements.txt /tmp/videogen-req.txt
RUN uv pip install --system --no-cache \
    -r /tmp/common-req.txt -r /tmp/videogen-req.txt \
    && rm /tmp/common-req.txt /tmp/videogen-req.txt

# Copy application code last
COPY . .
