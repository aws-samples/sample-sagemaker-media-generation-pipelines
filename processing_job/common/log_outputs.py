"""Log generated output files to DynamoDB.

Reads STEP_NAME and DYNAMODB_TABLE_NAME from environment variables.
Scans LOCAL_OUTPUT_DIR for files matching the given extensions and
writes one DynamoDB item per file.

Usage:
    python -m common.log_outputs --extensions .mp4 .webm --sidecar-file /path/to/_video_metadata.json
    python -m common.log_outputs --extensions .mp4 .webm --extra model=ltx mode=t2v
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone

try:
    from common.dynamodb import DynamoDBOperations
    from common.utils import VIDEO_EXTENSIONS, get_video_metadata, scan_output_files
except ImportError:
    from processing_job.common.dynamodb import DynamoDBOperations
    from processing_job.common.utils import VIDEO_EXTENSIONS, get_video_metadata, scan_output_files
from loguru import logger

from schema.columns import COL

DYNAMODB_TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]
STEP_NAME = os.environ["STEP_NAME"]
OUTPUT_S3_URI = os.environ.get("OUTPUT_S3_URI", "").rstrip("/")
EXECUTION_ID = os.environ.get("EXECUTION_ID", "")

DEFAULT_EXTENSIONS = (".mp4", ".webm", ".mkv", ".png", ".jpg", ".jpeg")


def _load_sidecar(sidecar_file: str | None) -> dict[str, dict] | None:
    """Load sidecar metadata JSON: {file_prefix: {input_id, model, mode, prompt, image}}."""
    if not sidecar_file or not os.path.exists(sidecar_file):
        return None
    try:
        with open(sidecar_file, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load sidecar file {}: {}", sidecar_file, exc)
        return None


def _match_sidecar(filename: str, sidecar: dict[str, dict]) -> dict | None:
    """Match a filename to its sidecar metadata by file_prefix."""
    for prefix, meta in sidecar.items():
        if filename.startswith(prefix):
            return meta
    return None


def main(
    extensions: tuple[str, ...] | None = None,
    extra: dict[str, str] | None = None,
    metadata: bool = False,
    sidecar_file: str | None = None,
):
    db_ops = DynamoDBOperations(table_name=DYNAMODB_TABLE_NAME)

    output_dir = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output")
    exts = extensions or DEFAULT_EXTENSIONS
    files = scan_output_files(output_dir, extensions=exts)

    if not files:
        logger.warning("No output files found in {}", output_dir)
        return

    timestamp = datetime.now(timezone.utc).isoformat()

    # Load sidecar metadata if provided
    sidecar = _load_sidecar(sidecar_file)
    if sidecar:
        logger.info("Loaded sidecar metadata with {} entries", len(sidecar))

    for f in files:
        # Skip internal sidecar/metadata files (prefixed with _)
        if f["filename"].startswith("_"):
            logger.debug("Skipping internal file: {}", f["filename"])
            continue
        s3_uri = f"{OUTPUT_S3_URI}/{f['filename']}" if OUTPUT_S3_URI else ""
        data = {
            COL.FILENAME: f["filename"],
            COL.S3_URI: s3_uri,
            COL.PIPELINE_EXECUTION_ID: EXECUTION_ID,
            COL.SIZE_BYTES: f["size_bytes"],
            COL.EXTENSION: f["extension"],
            COL.TIMESTAMP: timestamp,
            **(extra or {}),
        }

        # Match metadata from sidecar (file_prefix -> {input_id, model, mode, prompt, image})
        partition_key = str(uuid.uuid4())  # fallback if no sidecar match
        sort_key = STEP_NAME
        if sidecar:
            matched = _match_sidecar(f["filename"], sidecar)
            if matched:
                matched = dict(matched)  # copy so pop doesn't mutate sidecar
                raw_id = matched.pop("input_id", "")
                partition_key = raw_id if raw_id else partition_key
                model = matched.get("model", "")
                gen_idx = matched.get("generation_index")
                # Always include generation index in sort key so every
                # asset gets a unique record (e.g. t2i#z_image_turbo#g0)
                if model and gen_idx is not None:
                    sort_key = f"{STEP_NAME}#{model}#g{gen_idx}"
                elif model:
                    sort_key = f"{STEP_NAME}#{model}"
                data.update(matched)
            else:
                logger.warning("No sidecar match for filename: {}", f["filename"])

        if metadata and f["extension"] in VIDEO_EXTENSIONS:
            data.update(get_video_metadata(f["path"]))

        success = db_ops.put_item(id=partition_key, step=sort_key, data=data)
        if success:
            logger.info("Logged {} to DynamoDB (id={}, step={})", f["filename"], partition_key, sort_key)
        else:
            logger.error("Failed to log {} to DynamoDB", f["filename"])


def _parse_extra(pairs: list[str] | None) -> dict[str, str]:
    """Parse key=value pairs from CLI args."""
    if not pairs:
        return {}
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not value:
            raise argparse.ArgumentTypeError(f"Expected key=value, got: {pair}")
        out[key] = value
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log output files to DynamoDB")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="File extensions to scan (e.g. .mp4 .webm). Defaults to all media types.",
    )
    parser.add_argument(
        "--extra",
        nargs="+",
        default=None,
        help="Extra key=value pairs to include in each DynamoDB item (e.g. model=ltx mode=t2v)",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=False,
        help="Extract video metadata (width, height, fps, duration, frame_count) for video files",
    )
    parser.add_argument(
        "--sidecar-file",
        default=None,
        help="Path to sidecar JSON mapping UUID prefixes to per-video metadata",
    )
    args = parser.parse_args()

    exts = tuple(args.extensions) if args.extensions else None
    extra = _parse_extra(args.extra)
    main(extensions=exts, extra=extra, metadata=args.metadata, sidecar_file=args.sidecar_file)
