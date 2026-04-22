"""t2i: Generates images using Z-Image-Turbo via ComfyUI.

Usage: python3 main.py --generate
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from loguru import logger

try:
    from common.models import ImageSidecarEntry, VisualEntry
except ImportError:
    from processing_job.common.models import ImageSidecarEntry, VisualEntry
from pydantic import ValidationError

COMFY_HOME = os.environ.get("COMFY_HOME", "/opt/ml/processing/code/ComfyUI")
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output")
SM_INPUT_DIR = "/opt/ml/processing/input/input"

COMFY_STARTUP_WAIT = 120
QUEUE_POLL_INTERVAL = 15

# Deferred imports — set by main() after ComfyUI starts.
load_inputs = None
run_workflow = None


def log_directory_tree(path: str, label: str, max_depth: int = 3) -> None:
    """Log directory contents recursively up to max_depth."""
    logger.info("=== {} ({}) ===", label, path)
    if not os.path.exists(path):
        logger.warning("  Path does not exist: {}", path)
        return
    count = 0
    for root, dirs, files in os.walk(path):
        depth = root.replace(path, "").count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        indent = "  " * (depth + 1)
        rel = os.path.relpath(root, path)
        if rel != ".":
            logger.info("{}📁 {}/", indent, os.path.basename(root))
        for f in files:
            full = os.path.join(root, f)
            try:
                size = os.path.getsize(full)
                size_str = f"{size / (1024 * 1024):.1f}MB" if size > 1024 * 1024 else f"{size}B"
            except OSError:
                size_str = "?"
            logger.info("{}  {} ({})", indent, f, size_str)
            count += 1
    logger.info("  Total entries: {}", count)


def symlink_models_to_comfyui() -> None:
    """Symlink S3 model files into ComfyUI's models directory."""
    comfy_models = os.path.join(COMFY_HOME, "models")
    sm_input = "/opt/ml/processing/input"
    logger.info("Symlinking models into ComfyUI models dir: {}", comfy_models)

    if not os.path.isdir(sm_input):
        logger.error("SageMaker input root does not exist: {}", sm_input)
        return

    model_dirs = [
        os.path.join(sm_input, d)
        for d in sorted(os.listdir(sm_input))
        if d.startswith("models") and os.path.isdir(os.path.join(sm_input, d))
    ]
    if not model_dirs:
        logger.warning("No models* input channels found under {}", sm_input)
        return

    for model_dir in model_dirs:
        logger.info("Processing model channel: {}", model_dir)
        for entry in os.scandir(model_dir):
            if entry.is_dir():
                dest_dir = os.path.join(comfy_models, entry.name)
                os.makedirs(dest_dir, exist_ok=True)
                for sub in os.scandir(entry.path):
                    dest = os.path.join(dest_dir, sub.name)
                    if not os.path.exists(dest):
                        os.symlink(sub.path, dest)
                        logger.info("  {}/{} -> {}", entry.name, sub.name, dest)
            else:
                dest = os.path.join(comfy_models, entry.name)
                if not os.path.exists(dest):
                    os.symlink(entry.path, dest)
                    logger.info("  {} -> {}", entry.name, dest)


def wait_for_queue_empty() -> None:
    """Poll ComfyUI queue until empty, then stop the server."""
    try:
        from common.is_queue_empty import get_queue_size
    except ImportError:
        from is_queue_empty import get_queue_size

    logger.info("Starting queue monitoring...")
    while True:
        size = get_queue_size()
        if size is not None:
            logger.info("Current queue size: {}", size)
        if size == 0:
            logger.info("Queue is empty. Stopping comfy...")
            subprocess.run(["comfy", "stop"], check=False)
            logger.info("Comfy stop command executed.")
            return
        time.sleep(QUEUE_POLL_INTERVAL)


def copy_outputs() -> None:
    """Copy generated images from ComfyUI output to SageMaker output dir."""
    src = os.path.join(COMFY_HOME, "output", "z-image-turbo")
    logger.info("Copying generated images to SageMaker output directory...")
    if os.path.isdir(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(LOCAL_OUTPUT_DIR, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    logger.info("Files in output dir: {}", os.listdir(LOCAL_OUTPUT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="t2i image generation")
    parser.add_argument("--generate", action="store_true", help="Run image generation")
    parser.parse_args()

    logger.info("Starting t2i image generation with Z-Image-Turbo")

    # --- Models from S3 ---
    logger.info("--- Model Diagnostics ---")
    log_directory_tree("/opt/ml/processing/input/", "SageMaker input dir")
    log_directory_tree(os.path.join(COMFY_HOME, "models"), "ComfyUI models (before symlink)", max_depth=2)

    symlink_models_to_comfyui()
    log_directory_tree(os.path.join(COMFY_HOME, "models"), "ComfyUI models (after symlink)", max_depth=2)

    # Kill any stale ComfyUI process
    subprocess.run(["comfy", "stop"], check=False, capture_output=True)
    time.sleep(2)

    # Launch ComfyUI in background
    subprocess.run(["comfy", "launch", "--background"], check=True)
    logger.info("Launching ComfyUI in background... waiting {}s", COMFY_STARTUP_WAIT)
    time.sleep(COMFY_STARTUP_WAIT)

    # Import after ComfyUI is running — comfy_script.runtime.load() connects on import
    global load_inputs, run_workflow
    if load_inputs is None:
        from common.ltx import load_inputs as _load_inputs
        from z_image_turbo import run_workflow as _run_workflow

        load_inputs = _load_inputs
        run_workflow = _run_workflow

    raw_inputs = load_inputs()
    logger.info("Loaded {} raw inputs from shards/input", len(raw_inputs))

    validated_inputs = []
    for raw in raw_inputs:
        try:
            validated_inputs.append(VisualEntry.model_validate(raw))
        except ValidationError as e:
            logger.error("Shard validation failed: {}", e)
            continue

    inputs = validated_inputs
    logger.info("Validated {} inputs", len(inputs))
    for idx, inp in enumerate(inputs):
        prompt_preview = inp.prompt
        if len(prompt_preview) > 60:
            prompt_preview = prompt_preview[:60] + "..."
        logger.info("  input[{}]: id={}, prompt={}", idx, inp.id, prompt_preview)

    if not inputs:
        logger.error("No inputs loaded — nothing to generate. Check shards dir or inputs.json.")

    num_assets = int(os.environ.get("NUM_ASSETS_PER_PROMPT", "1"))
    BASE_SEED = 0
    logger.info("Generating {} asset(s) per prompt (base seed={})", num_assets, BASE_SEED)

    image_metadata = {}

    logger.info("Submitting {} Z-Image-Turbo workflows ({} assets each)...", len(inputs), num_assets)
    for _input in inputs:
        input_id = _input.id
        prompt = _input.prompt
        for gen_idx in range(num_assets):
            seed = BASE_SEED + gen_idx * 10
            suffix = f"_g{gen_idx}" if num_assets > 1 else ""
            file_prefix = f"{input_id}_z_image_turbo{suffix}"
            image_metadata[file_prefix] = ImageSidecarEntry(
                input_id=input_id,
                model="z_image_turbo",
                mode="t2i",
                prompt=prompt,
                seed=seed,
                generation_index=gen_idx,
            ).model_dump()
            logger.info("Queuing workflow: id={}, prefix={}, seed={}", input_id, file_prefix, seed)
            try:
                run_workflow(prompt, file_prefix, seed=seed)
                logger.info("Workflow queued: {}", file_prefix)
            except Exception as e:
                logger.error("Workflow failed for {}: {}", file_prefix, e)

    logger.info("Total workflows queued: {} (from {} inputs)", len(image_metadata), len(inputs))
    wait_for_queue_empty()
    copy_outputs()

    # Write sidecar metadata for log_outputs
    sidecar_path = os.path.join(LOCAL_OUTPUT_DIR, "_image_metadata.json")
    with open(sidecar_path, "w") as f:
        json.dump(image_metadata, f)
    logger.info("Wrote image metadata sidecar with {} entries", len(image_metadata))

    # Log to DynamoDB
    logger.info("Logging generated images to DynamoDB...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "common.log_outputs",
            "--extensions",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            "--sidecar-file",
            sidecar_path,
        ],
        check=True,
    )
    logger.info("Script completed successfully.")


if __name__ == "__main__":
    main()
    sys.exit(0)
