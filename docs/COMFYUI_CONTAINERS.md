# Running ComfyUI Workloads in Containers

This guide covers how to extend the framework to run ComfyUI workflows as headless batch jobs inside SageMaker Processing Job containers. It explains the container setup, how to convert a ComfyUI workflow into a Python script using ComfyScript's transpiler, and how to wire everything together as a new pipeline step.

## How It Works

Generation steps (t2v, i2v, flf2v, t2i, t2a) run ComfyUI in headless mode inside Docker containers on SageMaker Processing Jobs. There is no GUI — workflows are expressed as Python scripts using [ComfyScript](https://github.com/Chaoses-Ib/ComfyScript), a Python frontend for ComfyUI that maps ComfyUI nodes to Python function calls. The container:

1. Starts ComfyUI as a background process (`comfy launch --background`)
2. Symlinks model weights from SageMaker input channels into ComfyUI's `models/` directory
3. Imports the ComfyScript workflow module, which connects to the local ComfyUI server
4. Queues one workflow per input prompt
5. Polls the ComfyUI queue until all workflows complete
6. Copies generated outputs to the SageMaker output directory

## Container Setup

### Dockerfile

Every ComfyUI-based container follows the same pattern. Here's the annotated structure:

```dockerfile
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HOME=/opt/ml/processing/code/ \
    COMFY_HOME=/opt/ml/processing/code/ComfyUI \
    LOCAL_OUTPUT_DIR=/opt/ml/processing/output \
    COMFY_TRACKING_DISABLED=1

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 python3-pip python3-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# uv for fast dependency management
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

# Force PyTorch to match the GPU CUDA version on ml.g5 instances
RUN uv pip install --system --no-cache \
    --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Step-specific dependencies
COPY common/requirements.txt /tmp/common-req.txt
RUN uv pip install --system --no-cache -r /tmp/common-req.txt \
    && rm /tmp/common-req.txt

# Application code (copied last for layer caching)
COPY . .
```

Key points:
- `comfy-cli` installs ComfyUI into `$COMFY_HOME`
- ComfyScript is cloned as a ComfyUI custom node and installed with `[default]` extras
- PyTorch is reinstalled with the correct CUDA index to match the GPU driver on SageMaker instances
- `COMFY_TRACKING_DISABLED=1` prevents ComfyUI from phoning home

### Custom Nodes

If your workflow uses custom ComfyUI nodes (beyond the built-in ones), add them to the Dockerfile:

```dockerfile
# Install custom nodes after ComfyUI
RUN cd $COMFY_HOME/custom_nodes \
    && git clone --depth 1 https://github.com/<org>/<custom-node-repo>.git \
    && cd <custom-node-repo> \
    && uv pip install --system --no-cache -r requirements.txt
```

## Converting a ComfyUI Workflow to Python

ComfyScript includes a transpiler that converts ComfyUI workflow JSON files into Python scripts. This is the recommended way to create new workflow files for this framework.

### Step 1: Export the Workflow JSON

In the ComfyUI web UI, save the workflow as "API Format". This saves the workflow as a JSON file in API format (node IDs + connections, not the visual layout format)

The API format JSON is required. The standard "Save" format includes UI layout data that the transpiler cannot process.

### Step 2: Transpile to Python

With ComfyUI running locally (needed so the transpiler can resolve node types):

```bash
# Using an installed ComfyScript
python -m comfy_script.transpile "workflow_api.json" --api http://127.0.0.1:8188/


This outputs a Python script that uses ComfyScript's runtime API. For example, a simple text-to-image workflow transpiles to:

```python
model, clip, vae = CheckpointLoaderSimple('v1-5-pruned-emaonly.ckpt')
conditioning = CLIPTextEncode('beautiful scenery nature glass bottle landscape', clip)
conditioning2 = CLIPTextEncode('text, watermark', clip)
latent = EmptyLatentImage(512, 512, 1)
latent = KSampler(model, 156680208700286, 20, 8, 'euler', 'normal',
                  conditioning, conditioning2, latent, 1)
image = VAEDecode(latent, vae)
SaveImage(image, 'ComfyUI')
```

Each line maps 1:1 to a ComfyUI node. The function names match the node class names, and the arguments match the node inputs.

### Step 3: Wrap in a Workflow Function

Take the transpiled output and wrap it in a reusable function with the standard ComfyScript boilerplate:

```python
"""My custom workflow for ComfyUI.

Provides run_workflow() for generating outputs.
Import and call directly — do not run as a script.
"""

from comfy_script.runtime import *

load()  # Connects to the running ComfyUI server
from comfy_script.runtime.nodes import *


def run_workflow(prompt: str, file_prefix: str = "output", seed: int = 0) -> None:
    """Run a single generation workflow."""
    with Workflow():
        # --- Paste transpiled code here, parameterizing prompt/seed/prefix ---
        model, clip, vae = CheckpointLoaderSimple('my_model.safetensors')
        conditioning = CLIPTextEncode(prompt, clip)
        conditioning2 = CLIPTextEncode('', clip)
        latent = EmptyLatentImage(1024, 1024, 1)
        latent = KSampler(model, seed, 20, 7, 'euler', 'normal',
                          conditioning, conditioning2, latent, 1)
        image = VAEDecode(latent, vae)
        SaveImage(image, f"output/{file_prefix}")
```

Key patterns:
- `load()` at module level connects to the local ComfyUI server (must be called before importing nodes)
- `from comfy_script.runtime.nodes import *` imports all available ComfyUI nodes as Python functions
- `with Workflow():` queues the workflow for execution (non-blocking — ComfyUI processes it asynchronously)
- Replace hardcoded prompts, seeds, and filenames with function parameters
- Use `file_prefix` in `SaveImage`/`SaveVideo` to control output filenames

See existing workflow files for real examples:
- [`processing_job/t2i/z_image_turbo.py`](../processing_job/t2i/z_image_turbo.py) — simplest example (text-to-image)
- [`processing_job/common/wan22.py`](../processing_job/common/wan22.py) — two-pass video generation with LoRA
- [`processing_job/t2a/ace_step.py`](../processing_job/t2a/ace_step.py) — audio generation
- [`processing_job/flf2v/wan_flf.py`](../processing_job/flf2v/wan_flf.py) — first-last-frame-to-video

## Wiring Up a New ComfyUI Step

### 1. Create the Step Directory

```
processing_job/my_step/
├── Dockerfile           # Use the template above
├── buildspec.yml        # Standard CodeBuild spec (copy from t2i/)
├── main.py              # Entry point
├── my_workflow.py       # Transpiled + wrapped ComfyScript workflow
└── requirements.txt     # Any extra Python deps (optional)
```

### 2. Write main.py

The entry point follows a standard pattern:

```python
"""my_step: Description of what this step does."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from loguru import logger

COMFY_HOME = os.environ.get("COMFY_HOME", "/opt/ml/processing/code/ComfyUI")
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output")
COMFY_STARTUP_WAIT = 120
QUEUE_POLL_INTERVAL = 15


def symlink_models_to_comfyui():
    """Symlink S3 model files into ComfyUI's models directory."""
    comfy_models = os.path.join(COMFY_HOME, "models")
    sm_input = "/opt/ml/processing/input"
    for d in sorted(os.listdir(sm_input)):
        if d.startswith("models") and os.path.isdir(os.path.join(sm_input, d)):
            model_dir = os.path.join(sm_input, d)
            for entry in os.scandir(model_dir):
                if entry.is_dir():
                    dest_dir = os.path.join(comfy_models, entry.name)
                    os.makedirs(dest_dir, exist_ok=True)
                    for sub in os.scandir(entry.path):
                        dest = os.path.join(dest_dir, sub.name)
                        if not os.path.exists(dest):
                            os.symlink(sub.path, dest)


def wait_for_queue_empty():
    """Poll ComfyUI queue until all workflows complete."""
    from common.is_queue_empty import get_queue_size

    while True:
        size = get_queue_size()
        if size == 0:
            subprocess.run(["comfy", "stop"], check=False)
            return
        time.sleep(QUEUE_POLL_INTERVAL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", required=True)
    parser.parse_args()

    # 1. Symlink models from SageMaker input channels
    symlink_models_to_comfyui()

    # 2. Launch ComfyUI headlessly
    subprocess.run(["comfy", "stop"], check=False, capture_output=True)
    subprocess.run(["comfy", "launch", "--background"], check=True)
    time.sleep(COMFY_STARTUP_WAIT)

    # 3. Import workflow AFTER ComfyUI is running
    from my_workflow import run_workflow

    # 4. Load inputs and queue workflows
    inputs = json.load(open("/opt/ml/processing/input/input/inputs.json"))
    for item in inputs:
        run_workflow(prompt=item["prompt"], file_prefix=item["id"])

    # 5. Wait for completion and copy outputs
    wait_for_queue_empty()
    src = os.path.join(COMFY_HOME, "output")
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(LOCAL_OUTPUT_DIR, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    # 6. Log to DynamoDB
    subprocess.run(
        [sys.executable, "-m", "common.log_outputs", "--extensions", ".png", ".mp4"],
        check=True,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
```

The critical ordering: ComfyUI must be running before you import the workflow module, because `comfy_script.runtime.load()` (called at module level in the workflow file) connects to the ComfyUI server on import.

### 3. Add to Pipeline Config

In your pipeline config YAML (e.g. `config/pipeline/config_my_pipeline.yaml`):

```yaml
steps:
  my_step:
    instance_type: ml.g5.xlarge
    instance_count: 1
    volume_size_gb: 100
    container_entrypoint: ["python3", "main.py", "--generate"]
    models_prefix:
      - unet
      - clip
      - vae
    environment:
      NUM_ASSETS_PER_PROMPT: "1"

pipeline_graph:
  my_step: []  # No dependencies (root step), or ["parent_step"]
```

### 4. Add Model Downloads

Add model weight URLs to `s3_downloads` in the same config:

```yaml
s3_downloads:
  - url: "https://huggingface.co/org/model/resolve/main/model.safetensors"
    s3_key: "unet/model.safetensors"
```

The model download step downloads these to S3, and the processing job mounts them as input channels based on `models_prefix`. In CI/CD deployments, the download runs as a CodeBuild project in the ModelDownloadAndUpload stage. For manual deploys, invoke the model download trigger Lambda.

### 5. Deploy

```bash
make deploy
```

CodeBuild automatically builds the container image and pushes it to ECR. CDK creates all the SageMaker resources. No CDK code changes needed.

## Model Downloads

Before a ComfyUI container can run, the model weights it references must exist in the S3 models bucket. The `model_download` processing job handles this — it reads a manifest of URLs and S3 destinations, downloads each file (skipping files already present), and uploads them to S3.

### How It Works

1. At CDK synth time, `app.py` writes each pipeline config's `s3_downloads` list as a JSON manifest (`{construct_id}_downloads.json`) into the `processing_job/model_download/` directory
2. The manifest gets baked into the model_download container image at build time
3. When the job runs, it reads the manifest, checks S3 for each file, and downloads only what's missing
4. Large files (>100MB) use parallel range requests for faster downloads; smaller files stream directly

In the CI/CD pipeline, the ModelDownloadAndUpload stage triggers this job automatically after deploy. For manual deploys, invoke the model download trigger Lambda from the console or CLI.

### s3_downloads Config

Each entry in `s3_downloads` maps a source URL to a destination path in the models bucket:

```yaml
s3_downloads:
  # Single model file
  - url: "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors"
    path: "z_image_turbo/diffusion_models/z_image_turbo_bf16.safetensors"

  # Zip archive — downloaded, extracted into the path directory, then deleted
  - url: "https://github.com/org/repo/releases/download/v1/weights.zip"
    path: "my_model/weights/"
    extract: true
```

| Field | Type | Required | Description |
|---|---|---|---|
| `url` | string | Yes | HTTP/HTTPS URL to download from (HuggingFace, GitHub releases, etc.) |
| `path` | string | Yes | Destination path within the S3 models bucket. This becomes the S3 key. |
| `extract` | bool | No | If `true`, treat the download as a zip archive — extract contents into `path` and delete the zip. Default: `false`. |

### Finding Model URLs

For HuggingFace models, use the "resolve" URL format:
```
https://huggingface.co/{org}/{repo}/resolve/main/{path_to_file}
```

You can find the exact paths by browsing the model's "Files and versions" tab on HuggingFace. For example, the Z-Image-Turbo model files are at:
- `https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors`
- `https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors`
- `https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors`

### Connecting Downloads to Steps

The `path` in `s3_downloads` and the `models_prefix` in your step config must align. The top-level directory in the S3 path becomes the input channel name:

```yaml
# Downloads go to s3://bucket/z_image_turbo/diffusion_models/...
s3_downloads:
  - url: "https://..."
    path: "z_image_turbo/diffusion_models/model.safetensors"
  - url: "https://..."
    path: "z_image_turbo/vae/ae.safetensors"

steps:
  t2i:
    # This mounts s3://bucket/z_image_turbo/ at /opt/ml/processing/input/models/
    models_prefix: ["z_image_turbo"]
```

Inside the container, the model appears at `/opt/ml/processing/input/models/diffusion_models/model.safetensors`. The `symlink_models_to_comfyui()` function then symlinks it into ComfyUI's `models/diffusion_models/` directory so ComfyUI can find it by filename.

## Model Symlinking

SageMaker mounts model weights as read-only input channels at `/opt/ml/processing/input/models*`. ComfyUI expects models in its own `models/` directory tree (e.g. `models/unet/`, `models/clip/`, `models/vae/`). The `symlink_models_to_comfyui()` function bridges this by creating symlinks from the SageMaker paths into ComfyUI's expected directory structure.

The `models_prefix` config controls which S3 prefixes are mounted as separate input channels. For example, `models_prefix: [unet, clip, vae]` creates three channels that appear as subdirectories under `/opt/ml/processing/input/models/`.

## Queue Monitoring

ComfyUI processes workflows asynchronously. After queuing all workflows, the container polls the ComfyUI queue via its REST API until the queue is empty. The `common/is_queue_empty.py` utility handles this.

For video outputs, ComfyScript's internal watch callback can fail (PIL can't parse video files), causing the queue to appear non-empty even after all workflows complete. The existing containers handle this with a stale-detection mechanism — if the queue size doesn't change for a configurable number of polls, the container assumes completion and proceeds.

## Troubleshooting

- **ComfyUI fails to start**: Check that PyTorch CUDA version matches the instance GPU driver. Use `--index-url https://download.pytorch.org/whl/cu128` for ml.g5 instances.
- **"No module named comfy_script"**: Ensure ComfyScript is installed as a custom node AND as a pip package (`-e ".[default]"`).
- **Models not found**: Verify `models_prefix` in the pipeline config matches the S3 key prefixes in `s3_downloads`. Check that `symlink_models_to_comfyui()` runs before importing the workflow.
- **Queue never empties**: Check ComfyUI logs at `$COMFY_HOME/comfyui.log`. Common causes: missing custom nodes, incompatible model formats, or OOM on the GPU.

← [Back to docs](../README.md) | [Extending the Framework](EXTENDING.md) | [Config Authoring Guide](CONFIG_GUIDE.md)
