"""S3 URI helpers for downloading and cleaning up temporary image files.

Reusable across any container that needs to resolve S3 URIs to local paths
(e.g. i2v downloading images referenced by retrieval output shards).
"""

from __future__ import annotations

import os
import tempfile
from urllib.parse import urlparse

from loguru import logger


def is_s3_uri(path: str) -> bool:
    """Return True if *path* starts with ``s3://``."""
    return path.startswith("s3://")


def download_s3_to_temp(s3_uri: str, s3_client=None) -> str:
    """Download an S3 object to a local temp file and return the local path.

    Parameters
    ----------
    s3_uri:
        Full S3 URI, e.g. ``s3://bucket/key/image.jpg``.
    s3_client:
        Optional pre-configured ``boto3`` S3 client.  A new client is created
        when *None* is passed.

    Returns
    -------
    str
        Absolute path to the downloaded temporary file.
    """
    if s3_client is None:
        import boto3

        s3_client = boto3.client("s3")

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    # Preserve the original file extension so downstream code can detect format.
    _, ext = os.path.splitext(key)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)

    logger.info("Downloading s3://{}/{} → {}", bucket, key, tmp_path)
    s3_client.download_file(bucket, key, tmp_path)
    return tmp_path


def cleanup_temp_file(path: str) -> None:
    """Delete a temporary file if it exists.  Missing files are silently ignored."""
    try:
        os.remove(path)
        logger.debug("Cleaned up temp file: {}", path)
    except FileNotFoundError:
        logger.debug("Temp file already removed: {}", path)
