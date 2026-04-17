FROM --platform=linux/amd64 nvidia/cuda:12.3.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev curl git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV HOME=/opt/ml/processing/code/
ENV COMFY_HOME=/opt/ml/processing/code/ComfyUI
ENV LOCAL_OUTPUT_DIR=/opt/ml/processing/output
ENV COMFY_TRACKING_DISABLED=1

WORKDIR $HOME
RUN pip install --upgrade comfy-cli
RUN comfy --skip-prompt --workspace=$COMFY_HOME install --nvidia

RUN ln -s $COMFY_HOME/output $LOCAL_OUTPUT_DIR
COPY . .
