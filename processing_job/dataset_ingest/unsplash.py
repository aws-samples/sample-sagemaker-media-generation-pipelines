"""Unsplash Lite dataset loader.

Downloads images from the Unsplash Lite HuggingFace parquet dataset,
uploads them to S3 under ``images/{photo_id}.jpg``, and returns metadata
for each successfully uploaded image.

Standard interface
------------------
``load_and_upload(s3_client, bucket, limit) -> list[dict]``

Each returned dict has keys: ``id``, ``description``, ``s3_uri``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from botocore.exceptions import ClientError
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_URL = os.environ.get("DATASET_URL", "1aurent/unsplash-lite")
IMAGE_DOWNLOAD_TIMEOUT = 30  # seconds
S3_RETRY_ATTEMPTS = 3
S3_RETRY_BASE_DELAY = 1  # seconds (exponential: 1, 2, 4)
PROGRESS_LOG_INTERVAL = 500
MAX_WORKERS = 16
MAX_IMAGE_WIDTH = 1664  # Resize images so the longest side fits within this limit


# ---------------------------------------------------------------------------
# Description fallback chain
# ---------------------------------------------------------------------------


def resolve_description(row: dict) -> str | None:
    """Resolve the best available description for a parquet row.

    Fallback chain:
        1. ``ai["description"]`` — if present, not None, not ``"nan"``, not empty
        2. ``description`` — if not None, not ``"nan"``, not empty
        3. Comma-joined ``keywords`` — if the keywords list is non-empty
        4. ``None`` — row should be skipped

    Args:
        row: A dict representing a single parquet row with optional keys
             ``ai``, ``description``, and ``keywords``.

    Returns:
        The resolved description string, or ``None`` if no usable description.
    """
    # 1. Try ai["description"]
    ai = row.get("ai")
    if isinstance(ai, dict):
        ai_desc = ai.get("description")
        if ai_desc is not None and str(ai_desc).strip() and str(ai_desc).strip().lower() != "nan":
            return str(ai_desc).strip()

    # 2. Try description
    desc = row.get("description")
    if desc is not None and str(desc).strip() and str(desc).strip().lower() != "nan":
        return str(desc).strip()

    # 3. Try comma-joined keywords
    keywords = row.get("keywords")
    if keywords and isinstance(keywords, list):
        keyword_strings = []
        for kw in keywords:
            if isinstance(kw, dict):
                val = kw.get("keyword")
                if val and str(val).strip():
                    keyword_strings.append(str(val).strip())
            elif isinstance(kw, str) and kw.strip():
                keyword_strings.append(kw.strip())
        if keyword_strings:
            return ", ".join(keyword_strings)

    # 4. No usable description
    return None


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------


def download_image(url: str, timeout: int = IMAGE_DOWNLOAD_TIMEOUT) -> bytes:
    """Download an image from the Unsplash CDN.

    Args:
        url: The image URL.
        timeout: HTTP request timeout in seconds.

    Returns:
        The raw image bytes.

    Raises:
        requests.RequestException: If the download fails.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def resize_image(image_data: bytes, max_width: int = MAX_IMAGE_WIDTH) -> bytes:
    """Resize image so the longest side is at most max_width, preserving aspect ratio.

    Returns the original bytes unchanged if already within the limit.
    """
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(image_data))
    w, h = img.size
    if max(w, h) <= max_width:
        return image_data

    if w >= h:
        new_w = max_width
        new_h = int(h * max_width / w)
    else:
        new_h = max_width
        new_w = int(w * max_width / h)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# S3 upload with retry
# ---------------------------------------------------------------------------


def upload_to_s3(s3_client, bucket: str, key: str, data: bytes) -> bool:
    """Upload data to S3 with retry (3 attempts, exponential backoff: 1s, 2s, 4s).

    Args:
        s3_client: A boto3 S3 client.
        bucket: The S3 bucket name.
        key: The S3 object key.
        data: The raw bytes to upload.

    Returns:
        True if the upload succeeded, False if all retries were exhausted.
    """
    for attempt in range(S3_RETRY_ATTEMPTS):
        try:
            s3_client.put_object(Bucket=bucket, Key=key, Body=data)
            return True
        except ClientError as exc:
            delay = S3_RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "S3 upload failed for s3://{}/{} (attempt {}/{}): {} — retrying in {}s",
                bucket,
                key,
                attempt + 1,
                S3_RETRY_ATTEMPTS,
                exc,
                delay,
            )
            if attempt < S3_RETRY_ATTEMPTS - 1:
                time.sleep(delay)
    logger.error("S3 upload failed after {} attempts for s3://{}/{}", S3_RETRY_ATTEMPTS, bucket, key)
    return False


# ---------------------------------------------------------------------------
# Check if S3 key already exists (for new-vs-overwrite logging)
# ---------------------------------------------------------------------------


def _key_exists(s3_client, bucket: str, key: str) -> bool:
    """Check whether an S3 key already exists."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


# ---------------------------------------------------------------------------
# Single-row processing (download + upload)
# ---------------------------------------------------------------------------


def _flatten_row(row: dict) -> dict:
    """Flatten a dataset row so id/image_url/description live at the top level.

    The HuggingFace ``1aurent/unsplash-lite`` dataset nests ``id``,
    ``image_url``, and ``description`` inside a ``photo`` dict.  Other
    columns (``ai``, ``keywords``, …) are already top-level.  This helper
    merges the ``photo`` fields up so the rest of the code can access them
    uniformly via ``row["id"]``, ``row["image_url"]``, etc.
    """
    flat = dict(row)
    photo = flat.pop("photo", None)
    if isinstance(photo, dict):
        flat.update(photo)
    return flat


def _process_row(s3_client, bucket: str, row: dict) -> dict | None:
    """Process a single parquet row: resolve description, download image, upload to S3.

    Returns:
        A dict ``{id, description, s3_uri}`` on success, or ``None`` on failure/skip.
    """
    row = _flatten_row(row)
    photo_id = row.get("id")
    image_url = row.get("image_url")

    if not photo_id or not image_url:
        logger.warning("Skipping row with missing id or image_url: {}", row.get("id", "<unknown>"))
        return None

    description = resolve_description(row)
    if description is None:
        logger.warning("Skipping photo {} — no usable description", photo_id)
        return None

    # Download image
    try:
        image_data = download_image(str(image_url))
        image_data = resize_image(image_data)
    except Exception as exc:
        logger.warning("Failed to download image for photo {}: {}", photo_id, exc)
        return None

    # Check if key already exists (for logging)
    s3_key = f"images/{photo_id}.jpg"
    exists = _key_exists(s3_client, bucket, s3_key)

    # Upload to S3
    if not upload_to_s3(s3_client, bucket, s3_key, image_data):
        return None

    action = "overwrite" if exists else "new upload"
    logger.debug("Uploaded s3://{}/{} ({})", bucket, s3_key, action)

    return {
        "id": str(photo_id),
        "description": description,
        "s3_uri": f"s3://{bucket}/{s3_key}",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_and_upload(s3_client, bucket: str, limit: int) -> list[dict]:
    """Load the Unsplash Lite parquet dataset and upload images to S3.

    Standard interface called by ``main.py``.

    Args:
        s3_client: A boto3 S3 client.
        bucket: The S3 bucket name for image uploads.
        limit: Maximum number of rows to process from the dataset.

    Returns:
        A list of dicts ``{id, description, s3_uri}`` for each successfully
        uploaded image.
    """
    from datasets import load_dataset

    logger.info("Loading Unsplash Lite dataset from '{}' (limit={})", DATASET_URL, limit)
    ds = load_dataset(DATASET_URL, split="train")
    total_rows = min(len(ds), limit)
    logger.info("Dataset loaded: {} total rows, processing {} (equally spaced)", len(ds), total_rows)

    # Sample equally spaced indices across the full dataset
    if total_rows >= len(ds):
        indices = list(range(len(ds)))
    else:
        indices = [int(i * len(ds) / total_rows) for i in range(total_rows)]

    results: list[dict] = []
    failures = 0
    skipped = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for idx in indices:
            row = ds[idx]
            future = executor.submit(_process_row, s3_client, bucket, row)
            futures[future] = idx

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Unexpected error processing row {}: {}", futures[future], exc)
                failures += 1
                continue

            if result is None:
                skipped += 1
            else:
                results.append(result)

            processed = len(results) + failures + skipped
            if processed % PROGRESS_LOG_INTERVAL == 0 and processed > 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total_rows - processed
                logger.info(
                    "Progress: {}/{} processed, {} remaining ({} uploaded, {} skipped, {} failed) — {:.1f}s elapsed, {:.1f} img/s",
                    processed,
                    total_rows,
                    remaining,
                    len(results),
                    skipped,
                    failures,
                    elapsed,
                    rate,
                )

    elapsed = time.time() - start_time
    logger.info(
        "Upload complete: {} uploaded, {} skipped, {} failed out of {} rows in {:.1f}s",
        len(results),
        skipped,
        failures,
        total_rows,
        elapsed,
    )
    return results
