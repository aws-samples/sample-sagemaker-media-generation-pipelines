# ComfyUI Processing Job Container

This directory contains a Docker container and scripts for running **any ComfyUI workflow** on AWS SageMaker Processing Jobs using a queue-based processing pattern.

## Overview

**This pattern works with ANY ComfyUI workflow** - we're using Z-Image Turbo as an example. Whether you're doing image generation, image processing, video generation, or custom model inference, the queue-based processing pattern remains the same.

## Files

- **Dockerfile** - Container definition with ComfyUI and dependencies
- **run_job.sh** - Main processing script that manages the entire workflow
- **run_workflow.py** - Individual workflow runner with CLI parameters
- **is_queue_empty.py** - Queue monitoring script for processing completion
- **image_z_image_turbo.json** - Example ComfyUI workflow (Z-Image Turbo)
- **prompts.txt** - Example prompt file for text generation

## Container Architecture

### Dockerfile Breakdown

The container is optimized for SageMaker Processing Jobs with GPU support:

```dockerfile
FROM --platform=linux/amd64 nvidia/cuda:12.3.1-runtime-ubuntu22.04
```
- NVIDIA CUDA 12.3.1 runtime for GPU acceleration
- AMD64 platform for SageMaker compatibility

```dockerfile
ENV HOME=/opt/ml/processing/code/
ENV COMFY_HOME=/opt/ml/processing/code/ComfyUI
ENV LOCAL_OUTPUT_DIR=/opt/ml/processing/output
ENV COMFY_TRACKING_DISABLED=1
```
- SageMaker-compatible directory structure
- Automatic output directory linking for S3 sync
- Privacy-focused (analytics disabled)

```dockerfile
RUN pip install --upgrade comfy-cli
RUN comfy --skip-prompt --workspace=$COMFY_HOME install --nvidia
```
- Headless ComfyUI installation with GPU support
- Uses [ComfyUI CLI](https://docs.comfy.org/comfy-cli/getting-started) for management

## Queue Processing Pattern

The system uses an efficient queue-based approach:

1. **Launch**: ComfyUI server starts in background
2. **Queue Requests**: Multiple workflow requests pushed to ComfyUI queue
3. **Process**: ComfyUI processes queued requests in parallel
4. **Monitor**: `is_queue_empty.py` continuously checks queue status
5. **Terminate**: Processing job ends automatically when queue reaches zero


## Usage

### Scripts

#### run_job.sh - Main Processing Script
Orchestrates the entire workflow:
1. Launches ComfyUI server in background
2. Downloads required models from HuggingFace (To reduce docker image size)
3. Pushes multiple workflow requests to queue
4. Monitors queue until completion

```bash
./run_job.sh 5    # Runs the workflow 5 times
./run_job.sh 20   # Runs the workflow 20 times
./run_job.sh      # Default (10 runs)
```

#### run_workflow.py - Individual Workflow Runner
Parameterized workflow execution:

```bash
# Use custom workflow
python run_workflow.py --workflow my_workflow.json --seed 42

# Custom workflow and prompts
python run_workflow.py --workflow flux_dev.json --prompt-file custom_prompts.txt --seed 456
```

#### is_queue_empty.py - Queue Monitor
Checks ComfyUI queue status:

```bash
python is_queue_empty.py
# Output: Queue size: 0
```

## Workflow Customization

### Using Different Workflows

To swap workflows (e.g., from Z-Image to ANY):

1. **Export Workflow**: Export your ComfyUI workflow as JSON (API format)
2. **Add File**: Place JSON file in the processing_job directory
3. **Update Script**: Modify `run_job.sh` to reference your workflow:
   ```bash
   python3 run_workflow.py --seed $i --workflow your_workflow.json
   ```
4. **Update Models**: Modify model downloads in `run_job.sh` for your workflow's requirements
5. **Custom Nodes**: Install if needed using [ComfyUI CLI](https://docs.comfy.org/comfy-cli/getting-started#manage-custom-nodes)

## Configuration

### Environment Variables
- `COMFY_HOME`: ComfyUI installation (`/opt/ml/processing/code/ComfyUI`)
- `HOME`: Base directory (`/opt/ml/processing/code/`)
- `LOCAL_OUTPUT_DIR`: Output directory (`/opt/ml/processing/output`)

### Current Example (Z-Image Turbo)

**Models Downloaded:**
- `z_image_turbo_bf16.safetensors` - Diffusion model
- `qwen_3_4b.safetensors` - Text encoder
- `ae.safetensors` - VAE model

**Workflow Settings:**
- 1024x1024 image generation
- 9 sampling steps
- CFG scale of 1
- `res_multistep` sampler
- Configurable prompts and seeds

### Output
Generated files are automatically synced to the output S3 bucket