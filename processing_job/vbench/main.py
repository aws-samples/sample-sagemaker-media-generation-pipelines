"""vbench: Video quality evaluation using VBench metrics.

Usage: python3 main.py --script <script> --videos-path <path> --output-path <path> --models-path <path>
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

try:
    from common.dynamodb import DynamoDBOperations
except ImportError:
    from processing_job.common.dynamodb import DynamoDBOperations

try:
    from common.models import VBenchMetrics
except ImportError:
    from processing_job.common.models import VBenchMetrics

from pydantic import ValidationError

from schema.columns import COL

DYNAMODB_TABLE_NAME: str = os.environ["DYNAMODB_TABLE_NAME"]

db_ops = DynamoDBOperations(table_name=DYNAMODB_TABLE_NAME)

CACHE_PATH = os.path.expanduser("~/.cache/vbench")


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


def format_json_for_dynamodb(file_name: str) -> dict[str, dict]:
    """Parse vbench eval results into {video_filename_without_ext: {metric: value}}."""
    logger.info("Formatting {} for DynamoDB...", file_name)
    with open(file_name) as file:
        vbench_data = json.load(file)

    new_data = {}
    metrics = list(vbench_data.keys())

    for metric in metrics:
        for video_info in vbench_data[metric][1]:
            video_path = video_info["video_path"]
            filename = os.path.basename(video_path)
            filename_without_ext = os.path.splitext(filename)[0]

            if filename_without_ext not in new_data:
                new_data[filename_without_ext] = {}

            if metric == "dynamic_degree":
                new_data[filename_without_ext][metric] = 1 if video_info["video_results"] else 0
            else:
                if isinstance(video_info["video_results"], str):
                    video_info["video_results"] = 0
                new_data[filename_without_ext][metric] = video_info["video_results"]

    logger.info("Formatted {} videos with {} metrics", len(new_data), len(metrics))
    return new_data


def _resolve_videos_path(videos_path: str) -> str:
    """If videos_path contains a single subdirectory and no video files, descend into it.

    The upstream step writes videos into an execution-ID subfolder, e.g.
    /opt/ml/processing/input/videos/52w4zz02fj27/*.mp4.  SageMaker mounts
    the S3 prefix at videos_path, so we need to walk one level down.
    """
    VBENCH_EXTENSIONS = {".mp4", ".gif"}
    entries = os.listdir(videos_path)
    top_level_videos = [f for f in entries if os.path.splitext(f)[1].lower() in VBENCH_EXTENSIONS]
    if top_level_videos:
        return videos_path

    subdirs = [e for e in entries if os.path.isdir(os.path.join(videos_path, e))]
    if len(subdirs) == 1:
        candidate = os.path.join(videos_path, subdirs[0])
        logger.info("No videos at top level, descending into subdirectory: {}", candidate)
        return candidate

    return videos_path


import re

from schema.models import KNOWN_MODELS

# Pattern: {input_id}_{model}_{comfy_counter}_ or {input_id}_{model}_g{idx}_{comfy_counter}_
# Model alternatives built from the shared registry (longer variants first).
_MODEL_ALT = "|".join(re.escape(m) for m in KNOWN_MODELS)
_FILENAME_RE = re.compile(rf"^(.+?)_({_MODEL_ALT})(?:_g(\d+))?_\d+_$")


def _parse_video_filename(filename_no_ext: str, upstream_step: str) -> dict | None:
    """Parse a video filename into DynamoDB key components.

    Filename pattern: {input_id}_{model}_{comfy_counter}_ (no ext)
    Returns dict with 'id' and 'step', or None if pattern doesn't match.
    """
    m = _FILENAME_RE.match(filename_no_ext)
    if not m:
        return None
    input_id = m.group(1)
    model = m.group(2)
    gen_idx = int(m.group(3)) if m.group(3) else 0
    step = f"{upstream_step}#{model}#g{gen_idx}"
    return {"id": input_id, "step": step}


def build_prompt_file(videos_path: str) -> str | None:
    """Build a VBench prompt_file JSON mapping {video_filename: prompt} from DynamoDB.

    Parses each video filename to extract the DynamoDB key (id + step) and
    does a direct get_item lookup. Falls back to scan for non-standard filenames.
    Returns path to the temp prompt file, or None if no prompts found.
    """
    # Use the same extensions VBench checks in build_full_info_json
    VBENCH_EXTENSIONS = {".mp4", ".gif"}

    all_entries = os.listdir(videos_path)
    video_files = [f for f in all_entries if os.path.splitext(f)[1].lower() in VBENCH_EXTENSIONS]

    logger.info(
        "videos_path={} total_entries={} video_files={}",
        videos_path,
        len(all_entries),
        len(video_files),
    )
    for f in sorted(all_entries):
        logger.info("  entry: {} (ext={})", f, os.path.splitext(f)[1].lower())

    if not video_files:
        logger.warning("No video files found in {}", videos_path)
        return None

    prompt_map = {}
    upstream_step = os.environ.get("UPSTREAM_STEP", "")
    for basename in video_files:
        filename_no_ext = os.path.splitext(basename)[0]
        # Parse filename: {input_id}_{model}_{comfy_counter}_ → id, step
        parsed = _parse_video_filename(filename_no_ext, upstream_step)
        if parsed:
            item = db_ops.get_item(id=parsed["id"], step=parsed["step"])
            if item and item.get("prompt"):
                prompt_map[basename] = item["prompt"]
            else:
                logger.warning(
                    "No prompt found in DynamoDB for video '{}' (id={}, step={})",
                    basename,
                    parsed["id"],
                    parsed["step"],
                )
        else:
            # Fallback to scan for non-standard filenames
            matches = db_ops.query_by_filename_prefix(filename_no_ext)
            if matches and matches[0].get("prompt"):
                prompt_map[basename] = matches[0]["prompt"]
            else:
                logger.warning("No prompt found in DynamoDB for video '{}'", basename)

    if not prompt_map:
        raise RuntimeError(
            f"No prompts found in DynamoDB for any of the {len(video_files)} videos in {videos_path}. "
            "Ensure the upstream step (t2v/i2v) wrote prompt metadata to DynamoDB before vbench runs."
        )

    if len(prompt_map) < len(video_files):
        missing = [f for f in video_files if f not in prompt_map]
        raise RuntimeError(
            f"Only {len(prompt_map)}/{len(video_files)} videos have prompts in DynamoDB. "
            f"Missing prompts for: {missing}. "
            "VBench requires prompts for ALL videos when using prompt-dependent dimensions."
        )

    logger.info("Built prompt file with {} video-prompt mappings from DynamoDB", len(prompt_map))
    for k, v in prompt_map.items():
        logger.info("  prompt_map[{}] = {}", k, v[:80] if len(v) > 80 else v)

    prompt_file_path = "/tmp/vbench_prompts.json"
    with open(prompt_file_path, "w") as f:
        json.dump(prompt_map, f)

    return prompt_file_path


def main(script: str, videos_path: str, models_path: str, output_path: str):
    logger.info("--- Directory Diagnostics ---")
    logger.info("videos_path={}, models_path={}, output_path={}", videos_path, models_path, output_path)
    log_directory_tree("/opt/ml/processing", "Full processing tree", max_depth=3)
    log_directory_tree(videos_path, "Videos input", max_depth=2)
    log_directory_tree(models_path, "Models input", max_depth=2)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(CACHE_PATH, exist_ok=True)

    # Copy models to VBench cache directory
    logger.info("Copying model files to cache directory...")
    try:
        for item in os.listdir(models_path):
            s = os.path.join(models_path, item)
            d = os.path.join(CACHE_PATH, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        logger.info("Files successfully copied to {}", CACHE_PATH)
    except Exception as e:
        logger.error("Failed to copy files to cache directory: {}", e)
        sys.exit(1)

    log_directory_tree(CACHE_PATH, "VBench cache after copy", max_depth=2)

    # Resolve videos path — upstream steps write into an execution-ID subfolder
    videos_path = _resolve_videos_path(videos_path)
    logger.info("Resolved videos_path: {}", videos_path)
    log_directory_tree(videos_path, "Resolved videos directory", max_depth=2)

    # Build prompt file (raises RuntimeError if prompts are missing)
    prompt_file = build_prompt_file(videos_path)
    if prompt_file is None:
        logger.info("No video files on this instance — nothing to evaluate (normal for sharded jobs)")
        sys.exit(0)

    dimensions = [
        # Reference-free dimensions
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        # Prompt-dependent dimensions
        "temporal_flickering",
        "temporal_style",
        "overall_consistency",
        "human_action",
    ]

    logger.info("Running {} dimensions with prompt file", len(dimensions))

    cmd = [
        "python3",
        script,
        "--dimension",
        *dimensions,
        "--videos_path",
        videos_path,
        "--mode",
        "custom_input",
        "--output_path",
        output_path,
        "--load_ckpt_from_local",
        "True",
        "--prompt_file",
        prompt_file,
    ]

    logger.info("Running VBench: {}", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        logger.info("Evaluation completed successfully")
    except subprocess.CalledProcessError:
        logger.error("Evaluation failed")
        log_directory_tree(output_path, "Output dir after failure", max_depth=2)
        sys.exit(1)

    log_directory_tree(output_path, "Output dir after evaluation", max_depth=2)

    files = glob.glob(f"{output_path}/*_eval_results.json")
    logger.info("Eval result files: {}", files)
    dynamodb_dict = format_json_for_dynamodb(files[0])

    step_name = os.environ.get("STEP_NAME", "vbench")
    upstream_step = os.environ.get("UPSTREAM_STEP", "")
    timestamp = datetime.now(timezone.utc).isoformat()

    for video_filename, metrics in dynamodb_dict.items():
        # Validate metrics via VBenchMetrics model
        # Convert int values to float (e.g. dynamic_degree returns 0/1 as int)
        metrics_float = {k: float(v) for k, v in metrics.items()}
        try:
            validated = VBenchMetrics.model_validate(metrics_float)
        except ValidationError as e:
            logger.error("Metrics validation failed for '{}': {}", video_filename, e)
            continue

        metrics_dict = validated.model_dump(exclude_none=True)

        # Look up existing DynamoDB row by direct key lookup (parsed from filename)
        filename_no_ext = os.path.splitext(video_filename)[0]
        parsed = _parse_video_filename(filename_no_ext, upstream_step)
        if parsed:
            row = db_ops.get_item(id=parsed["id"], step=parsed["step"])
            if not row:
                logger.warning(
                    "No DynamoDB row found for video '{}' (id={}, step={}), skipping",
                    video_filename,
                    parsed["id"],
                    parsed["step"],
                )
                continue
            matches = [row]
        else:
            matches = db_ops.query_by_filename_prefix(video_filename)
            if not matches:
                logger.warning("No DynamoDB row found for video '{}', skipping", video_filename)
                continue

        for row in matches:
            row_id = row["id"]
            row_step = row["step"]

            # Write a dedicated vbench row for this video
            parts = row_step.split("#")
            model = parts[1] if len(parts) >= 2 else "unknown"
            gen_suffix = "#".join(parts[2:]) if len(parts) >= 3 else ""
            vbench_sort_key = f"{step_name}#{model}" + (f"#{gen_suffix}" if gen_suffix else "")
            vbench_data = {
                **metrics_dict,
                COL.SOURCE_FILENAME: video_filename,
                COL.UPSTREAM_STEP: row_step,
                COL.PIPELINE_EXECUTION_ID: os.environ.get("EXECUTION_ID", ""),
                COL.TIMESTAMP: timestamp,
            }
            db_ops.put_item(id=row_id, step=vbench_sort_key, data=vbench_data)
            logger.info("Wrote vbench row id={} step={} for '{}'", row_id, vbench_sort_key, video_filename)

            # Only accumulate for the average from the upstream step's rows
            if upstream_step and not row_step.startswith(upstream_step + "#"):
                continue

    logger.info("VBench evaluation and DynamoDB writes complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, required=True)
    parser.add_argument("--videos-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--models-path", type=str, required=True)
    args = parser.parse_args()
    main(
        script=(Path("VBench") / args.script).as_posix(),
        videos_path=args.videos_path,
        models_path=args.models_path,
        output_path=args.output_path,
    )
    logger.info("Done!")
