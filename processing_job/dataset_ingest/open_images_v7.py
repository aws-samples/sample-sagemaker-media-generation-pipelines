"""Open Images V7 dataset loader.

Downloads image metadata from the Open Images V7 CSV hosted on Google Storage,
filters images by resolution using a streaming header-only approach, downloads
qualifying images, uploads them to S3 under ``images/{ImageID}.jpg``, and returns
metadata for each successfully uploaded image.

Standard interface
------------------
``load_and_upload(s3_client, bucket, limit) -> list[dict]``

Each returned dict has keys: ``id``, ``description``, ``s3_uri``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests
from loguru import logger
from PIL import Image
from unsplash import _key_exists, download_image, resize_image, upload_to_s3

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_URL = os.environ.get("DATASET_URL", "")
MIN_WIDTH = int(os.environ.get("MIN_WIDTH", "200"))
MAX_WIDTH = int(os.environ.get("MAX_WIDTH", "1280"))
MIN_HEIGHT = int(os.environ.get("MIN_HEIGHT", "200"))
MAX_HEIGHT = int(os.environ.get("MAX_HEIGHT", "720"))
CSV_READ_MULTIPLIER = int(os.environ.get("CSV_READ_MULTIPLIER", "3"))
MAX_WORKERS = 8
RESIZE_IMAGES = os.environ.get("RESIZE_IMAGES", "false").lower() == "true"
DOWNLOAD_MAX_RETRIES = 5
DOWNLOAD_BACKOFF_BASE = 2.0
PROGRESS_LOG_INTERVAL = 500


# ---------------------------------------------------------------------------
# CSV download
# ---------------------------------------------------------------------------


def _download_csv(url: str, nrows: int, skiprows: int = 0) -> pd.DataFrame:
    """Download CSV metadata from the given URL, reading only ``nrows`` rows.

    Uses ``pandas.read_csv`` with the ``nrows`` and ``skiprows`` parameters
    to paginate through large CSVs without loading all rows into memory.

    Args:
        url: The URL to the CSV file.
        nrows: Maximum number of rows to read.
        skiprows: Number of data rows to skip (header is always read).

    Returns:
        A pandas DataFrame with the CSV data.

    Raises:
        Exception: If the download or parsing fails, with a descriptive
            error message including the URL.
    """
    try:
        # Read header first, then skip + read nrows
        if skiprows > 0:
            # Read header from first row, skip `skiprows` data rows
            header = pd.read_csv(url, nrows=0)
            df = pd.read_csv(url, skiprows=range(1, skiprows + 1), nrows=nrows, header=0, names=header.columns)
        else:
            df = pd.read_csv(url, nrows=nrows)
        logger.info("Downloaded CSV from '{}': {} rows read (nrows={}, skiprows={})", url, len(df), nrows, skiprows)
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to download CSV from '{url}': {exc}") from exc


# ---------------------------------------------------------------------------
# Streaming header-based dimension check
# ---------------------------------------------------------------------------


def _check_image_dimensions(url: str) -> tuple[int, int] | None:
    """Stream just enough bytes to read the image header and extract dimensions.

    Retries with exponential backoff on HTTP 429 (Too Many Requests).

    Args:
        url: The image URL.

    Returns:
        A ``(width, height)`` tuple, or ``None`` if the header could not be
        parsed (network error, corrupt header, insufficient data, etc.).
    """
    for attempt in range(DOWNLOAD_MAX_RETRIES):
        try:
            resp = requests.get(url, stream=True, timeout=10)
            resp.raise_for_status()
            chunk = b""
            for part in resp.iter_content(chunk_size=16384):
                chunk += part
                if len(chunk) >= 16384:
                    break
            resp.close()

            img = Image.open(BytesIO(chunk))
            return img.size  # (width, height)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                delay = DOWNLOAD_BACKOFF_BASE**attempt
                time.sleep(delay)
                continue
            return None
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Resolution filter
# ---------------------------------------------------------------------------


def _passes_resolution_filter(width: int, height: int) -> bool:
    """Check whether the given dimensions fall within the configured bounds.

    Returns ``True`` if ``MIN_WIDTH <= width <= MAX_WIDTH`` and
    ``MIN_HEIGHT <= height <= MAX_HEIGHT``.
    """
    return MIN_WIDTH <= width <= MAX_WIDTH and MIN_HEIGHT <= height <= MAX_HEIGHT


# ---------------------------------------------------------------------------
# Single-image processing (download + resize + upload)
# ---------------------------------------------------------------------------


def _process_image(s3_client, bucket: str, image_id: str, url: str) -> dict | None:
    """Download an image, optionally resize, and upload to S3.

    Retries with exponential backoff on HTTP 429 (Too Many Requests).

    Args:
        s3_client: A boto3 S3 client.
        bucket: The S3 bucket name.
        image_id: The Open Images ImageID.
        url: The image URL (OriginalURL from CSV).

    Returns:
        A dict ``{id, description, s3_uri}`` on success, or ``None`` on failure.
    """
    # Download with retry on 429
    image_data = None
    for attempt in range(DOWNLOAD_MAX_RETRIES):
        try:
            image_data = download_image(url)
            break
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                delay = DOWNLOAD_BACKOFF_BASE**attempt
                logger.debug(
                    "429 for image {}, retrying in {:.1f}s (attempt {}/{})",
                    image_id,
                    delay,
                    attempt + 1,
                    DOWNLOAD_MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            logger.warning("Failed to download image {}: {}", image_id, exc)
            return None
        except Exception as exc:
            logger.warning("Failed to download image {}: {}", image_id, exc)
            return None

    if image_data is None:
        logger.warning("Image {} exhausted retries (429)", image_id)
        return None

    # Validate image and check dimensions
    try:
        img = Image.open(BytesIO(image_data))
        width, height = img.size
    except Exception as exc:
        logger.warning("Failed to read dimensions for image {}: {}", image_id, exc)
        return None

    if not _passes_resolution_filter(width, height):
        logger.debug("Image {} skipped: {}x{} outside resolution bounds", image_id, width, height)
        return None

    # Resize (only if enabled)
    if RESIZE_IMAGES:
        try:
            image_data = resize_image(image_data)
        except Exception as exc:
            logger.warning("Failed to resize image {}: {}", image_id, exc)
            return None

    # Check if key already exists (for new/overwrite logging)
    s3_key = f"images/{image_id}.jpg"
    exists = _key_exists(s3_client, bucket, s3_key)

    # Upload to S3
    if not upload_to_s3(s3_client, bucket, s3_key, image_data):
        return None

    action = "overwrite" if exists else "new upload"
    logger.debug("Uploaded s3://{}/{} ({})", bucket, s3_key, action)

    return {
        "id": image_id,
        "description": image_id,
        "s3_uri": f"s3://{bucket}/{s3_key}",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def load_and_upload(s3_client, bucket: str, limit: int) -> list[dict]:
    """Load Open Images V7 metadata and upload qualifying images to S3.

    Standard interface called by ``main.py``.

    Reads the CSV in batches of ``CSV_READ_MULTIPLIER * limit`` rows,
    paginating through the dataset until ``limit`` images have been
    successfully uploaded or the CSV is exhausted.

    Args:
        s3_client: A boto3 S3 client.
        bucket: The S3 bucket name for image uploads.
        limit: Maximum number of images to upload.

    Returns:
        A list of dicts ``{id, description, s3_uri}`` for each successfully
        uploaded image.
    """
    if not DATASET_URL:
        raise ValueError("DATASET_URL environment variable is required but not set")

    batch_size = CSV_READ_MULTIPLIER * limit
    results: list[dict] = []
    failures = 0
    skipped = 0
    csv_offset = 0
    start_time = time.time()

    logger.info(
        "Loading Open Images V7 CSV from '{}' (target={}, batch_size={}, CSV_READ_MULTIPLIER={})",
        DATASET_URL,
        limit,
        batch_size,
        CSV_READ_MULTIPLIER,
    )

    while len(results) < limit:
        df = _download_csv(DATASET_URL, batch_size, skiprows=csv_offset)
        if df.empty:
            logger.warning("CSV exhausted at offset {} with only {} images uploaded", csv_offset, len(results))
            break

        csv_offset += len(df)
        logger.info(
            "Processing batch: {} rows (offset={}, {} uploaded so far, target={})",
            len(df),
            csv_offset,
            len(results),
            limit,
        )

        def _handle_row(row) -> dict | None:
            """Process a single CSV row: download and upload directly."""
            image_id = str(row.get("ImageID", ""))
            # Prefer Thumbnail300KURL (~640x480, faster) over OriginalURL
            url = str(row.get("Thumbnail300KURL", ""))
            if not url or url == "nan":
                url = str(row.get("OriginalURL", ""))

            if not image_id or not url or url == "nan":
                logger.warning("Skipping row with missing ImageID or URL")
                return None

            return _process_image(s3_client, bucket, image_id, url)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for _, row in df.iterrows():
                future = executor.submit(_handle_row, row)
                futures[future] = row.get("ImageID", "<unknown>")

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Unexpected error processing image {}: {}", futures[future], exc)
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
                    logger.info(
                        "Progress: {} uploaded, {} skipped, {} failed — {:.1f}s elapsed, {:.1f} img/s",
                        len(results),
                        skipped,
                        failures,
                        elapsed,
                        rate,
                    )

                if len(results) >= limit:
                    # Cancel remaining futures to avoid waiting for the full batch
                    for f in futures:
                        f.cancel()
                    break

        if len(results) >= limit:
            break

    elapsed = time.time() - start_time

    if len(results) < limit:
        logger.warning(
            "Shortfall: only {} of {} requested images passed filtering and uploaded successfully",
            len(results),
            limit,
        )

    logger.info(
        "Upload complete: {} uploaded, {} skipped, {} failed out of {} CSV rows in {:.1f}s",
        len(results),
        skipped,
        failures,
        csv_offset,
        elapsed,
    )
    return results
