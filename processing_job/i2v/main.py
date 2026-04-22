"""i2v: Image-to-video generation using LTX and Wan 2.2 via ComfyUI.

Usage: python3 main.py --model <ltx|wan|both>
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
    from common.models import VideoSidecarEntry, VisualEntry
except ImportError:
    from processing_job.common.models import VideoSidecarEntry, VisualEntry
try:
    from common.s3_utils import is_s3_uri
except ImportError:
    from processing_job.common.s3_utils import is_s3_uri
from pydantic import ValidationError

COMFY_HOME = os.environ.get("COMFY_HOME", "/opt/ml/processing/code/ComfyUI")
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output")
SM_INPUT_DIR = "/opt/ml/processing/input/input"

COMFY_STARTUP_WAIT = 120
QUEUE_POLL_INTERVAL = 15
MODE = "i2v"

# Deferred imports — set by main() after ComfyUI starts.
# Declared here so tests can monkeypatch them.
load_inputs = None
ltx_run_workflow = None
wan_run_i2v = None


def resolve_image(image_field: str, s3_client=None) -> str:
    """Resolve image field to a local path inside ComfyUI's input directory.

    S3 URIs are downloaded to $COMFY_HOME/input/ so ComfyUI can read them.
    Local filenames are returned as-is (already linked by link_inputs_to_comfyui).
    """
    if is_s3_uri(image_field):
        if s3_client is None:
            import boto3

            s3_client = boto3.client("s3")

        from urllib.parse import urlparse

        parsed = urlparse(image_field)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        filename = os.path.basename(key)

        comfy_input = os.path.join(COMFY_HOME, "input")
        os.makedirs(comfy_input, exist_ok=True)
        local_path = os.path.join(comfy_input, filename)

        logger.info("Downloading s3://{}/{} → {}", bucket, key, local_path)
        s3_client.download_file(bucket, key, local_path)
        return filename  # ComfyUI resolves relative to its input dir
    return image_field


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


def link_inputs_to_comfyui() -> None:
    """Symlink SageMaker input files into ComfyUI's input directory."""
    comfy_input = os.path.join(COMFY_HOME, "input")
    os.makedirs(comfy_input, exist_ok=True)
    if not os.path.isdir(SM_INPUT_DIR):
        logger.warning("SageMaker input dir not found: {}", SM_INPUT_DIR)
        return
    for entry in os.scandir(SM_INPUT_DIR):
        dest = os.path.join(comfy_input, entry.name)
        if not os.path.exists(dest):
            os.symlink(entry.path, dest)
            logger.info("Linked {} -> {}", entry.name, dest)


def symlink_models_to_comfyui() -> None:
    """Symlink S3 model files into ComfyUI's models directory.

    Discovers all SageMaker input channels whose directory name starts with
    ``models`` (e.g. ``models``, ``models_ltx``, ``models_wan_22_i2v``) and
    symlinks their contents into the matching ComfyUI models subdirectory.
    """
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
    """Poll ComfyUI queue until empty or stale, then stop the server.

    ComfyScript's watch callback may fail on video outputs (PIL can't parse
    video files), preventing the queue from fully draining. If the queue size
    doesn't change for STALE_THRESHOLD polls, we assume all workflows have
    completed and proceed.
    """
    try:
        from common.is_queue_empty import get_queue_size
    except ImportError:
        from is_queue_empty import get_queue_size

    logger.info("Starting queue monitoring...")
    STALE_THRESHOLD = 120  # 120 * 15s = 30 minutes of no progress
    stale_count = 0
    last_size = None

    while True:
        size = get_queue_size()
        if size is not None:
            logger.info("Current queue size: {}", size)
        if size == 0:
            logger.info("Queue is empty. Stopping comfy...")
            subprocess.run(["comfy", "stop"], check=False)
            logger.info("Comfy stop command executed.")
            return

        if size == last_size:
            stale_count += 1
        else:
            stale_count = 0
        last_size = size

        if stale_count >= STALE_THRESHOLD:
            logger.warning(
                "Queue stale for {} polls (size={}). "
                "ComfyScript watch likely failed on video output. "
                "Proceeding with output copy.",
                stale_count,
                size,
            )
            video_dir = os.path.join(COMFY_HOME, "output", "video")
            if os.path.isdir(video_dir):
                files = os.listdir(video_dir)
                logger.info("ComfyUI output/video contains {} files: {}", len(files), files[:20])
            subprocess.run(["comfy", "stop"], check=False)
            return

        time.sleep(QUEUE_POLL_INTERVAL)


def copy_outputs() -> None:
    src = os.path.join(COMFY_HOME, "output", "video")
    logger.info("Copying generated videos to SageMaker output directory...")
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
    parser = argparse.ArgumentParser(description="i2v video generation")
    parser.add_argument("--model", choices=["ltx", "wan", "both"], default="both")
    args = parser.parse_args()

    logger.info("Model: {}, Mode: {}", args.model, MODE)

    # --- Models from S3 ---
    logger.info("--- Model Diagnostics ---")
    log_directory_tree("/opt/ml/processing/input/", "SageMaker input dir")
    log_directory_tree("/opt/ml/processing/input/input/", "SageMaker input/input dir")

    # Symlink all models* input channels into ComfyUI
    symlink_models_to_comfyui()

    # Link SageMaker input files into ComfyUI's input directory
    link_inputs_to_comfyui()

    # Kill any stale ComfyUI process (can happen if container image cached one)
    subprocess.run(["comfy", "stop"], check=False, capture_output=True)
    time.sleep(2)

    # Launch ComfyUI in background
    subprocess.run(["comfy", "launch", "--background"], check=True)
    logger.info("Launching ComfyUI in background... waiting {}s", COMFY_STARTUP_WAIT)
    time.sleep(COMFY_STARTUP_WAIT)

    # Import after ComfyUI is running — comfy_script.runtime.load() connects on import
    # Assign to module globals so tests can monkeypatch them.
    global load_inputs, ltx_run_workflow, wan_run_i2v
    if load_inputs is None:
        from common import ltx as _ltx_mod
        from common import wan22 as _wan_mod

        load_inputs = _ltx_mod.load_inputs
        ltx_run_workflow = _ltx_mod.run_workflow
        wan_run_i2v = _wan_mod.run_i2v

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
        logger.info("  input[{}]: id={}, prompt={}, image={}", idx, inp.id, prompt_preview, inp.image)

    if not inputs:
        logger.error("No inputs loaded — nothing to generate. Check shards dir or inputs.json.")

    num_assets = int(os.environ.get("NUM_ASSETS_PER_PROMPT", "1"))
    BASE_SEED = 42
    logger.info("Generating {} asset(s) per prompt (base seed={})", num_assets, BASE_SEED)

    video_metadata = {}  # {uuid: {input_id, model, mode, prompt, image}}

    if args.model in ("ltx", "both"):
        logger.info("Submitting {} LTX i2v workflows ({} assets each)...", len(inputs), num_assets)
        for _input in inputs:
            input_id = _input.id
            prompt = _input.prompt
            image = _input.image
            try:
                resolved_path = resolve_image(image)
            except Exception as e:
                logger.warning("Skipping input {} — image not found: {}", input_id, e)
                continue
            for gen_idx in range(num_assets):
                seed = BASE_SEED + gen_idx * 10
                suffix = f"_g{gen_idx}" if num_assets > 1 else ""
                file_prefix = f"{input_id}_ltx23{suffix}"
                video_metadata[file_prefix] = VideoSidecarEntry(
                    input_id=input_id,
                    model="ltx23",
                    mode=MODE,
                    prompt=prompt,
                    source_filename=image,
                    seed=seed,
                    generation_index=gen_idx,
                ).model_dump()
                logger.info("Queuing LTX workflow: id={}, prefix={}, seed={}", input_id, file_prefix, seed)
                try:
                    ltx_run_workflow(
                        prompt, resolved_path, disable_i2v=False, input_id=input_id, file_prefix=file_prefix, seed=seed
                    )
                    logger.info("LTX workflow queued: {}", file_prefix)
                except Exception as e:
                    logger.error("LTX workflow failed for {}: {}", file_prefix, e)

    if args.model in ("wan", "both"):
        logger.info("Submitting {} Wan i2v workflows ({} assets each)...", len(inputs), num_assets)
        for _input in inputs:
            input_id = _input.id
            prompt = _input.prompt
            image = _input.image
            try:
                resolved_path = resolve_image(image)
            except Exception as e:
                logger.warning("Skipping input {} — image not found: {}", input_id, e)
                continue
            for gen_idx in range(num_assets):
                seed = BASE_SEED + gen_idx * 10
                suffix = f"_g{gen_idx}" if num_assets > 1 else ""
                file_prefix = f"{input_id}_wan22{suffix}"
                video_metadata[file_prefix] = VideoSidecarEntry(
                    input_id=input_id,
                    model="wan22",
                    mode=MODE,
                    prompt=prompt,
                    source_filename=image,
                    seed=seed,
                    generation_index=gen_idx,
                ).model_dump()
                logger.info("Queuing Wan workflow: id={}, prefix={}, seed={}", input_id, file_prefix, seed)
                try:
                    wan_run_i2v(prompt, resolved_path, input_id, file_prefix, seed=seed)
                    logger.info("Wan workflow queued: {}", file_prefix)
                except Exception as e:
                    logger.error("Wan workflow failed for {}: {}", file_prefix, e)

    logger.info("Total workflows queued: {} (from {} inputs)", len(video_metadata), len(inputs))
    wait_for_queue_empty()
    copy_outputs()

    # Write sidecar metadata for log_outputs
    sidecar_path = os.path.join(LOCAL_OUTPUT_DIR, "_video_metadata.json")
    with open(sidecar_path, "w") as f:
        json.dump(video_metadata, f)
    logger.info("Wrote video metadata sidecar with {} entries", len(video_metadata))

    # Log to DynamoDB
    logger.info("Logging generated videos to DynamoDB...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "common.log_outputs",
            "--extensions",
            ".mp4",
            ".webm",
            ".mkv",
            "--metadata",
            "--sidecar-file",
            sidecar_path,
        ],
        check=True,
    )
    logger.info("Script completed successfully.")


if __name__ == "__main__":
    main()
    sys.exit(0)
