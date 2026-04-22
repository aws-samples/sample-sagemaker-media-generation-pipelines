#!/usr/bin/env python3
"""
Model download script for uploading model files to S3.

Downloads files from URLs (including HuggingFace) and uploads to S3.
Uses parallel downloads and streaming uploads to handle large files
without memory issues. Runs in CodeBuild (standalone or CI/CD pipeline).
"""

import json
import os
import re
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import boto3
from loguru import logger

# S3 client
s3_client = boto3.client("s3")

_ALLOWED_URL_SCHEMES = {"https", "http"}


def _validate_url_scheme(url: str) -> None:
    """Reject non-HTTP(S) URLs to prevent file:// and custom-scheme access."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"Unsupported URL scheme {scheme!r} — only https/http allowed")


# Constants
PART_SIZE = 50 * 1024 * 1024  # 50 MB per part
MAX_WORKERS = 8
LOCAL_DOWNLOAD_DIR = "/tmp/model_download"
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 30  # seconds — doubles each retry (30s, 60s, 120s)


def _fmt_size(nbytes: int | None) -> str:
    """Format byte count as human-readable string."""
    if nbytes is None:
        return "unknown"
    if nbytes >= 1024**3:
        return f"{nbytes / (1024**3):.2f} GB"
    if nbytes >= 1024**2:
        return f"{nbytes / (1024**2):.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} B"


def _model_name(s3_key: str) -> str:
    """Extract a short model group name from the S3 key (first path segment)."""
    return s3_key.strip("/").split("/")[0] if s3_key else s3_key


# HuggingFace URL pattern
HF_URL_PATTERN = re.compile(
    r"^https://huggingface\.co/(?P<repo_id>[^/]+/[^/]+)/resolve/(?P<revision>[^/]+)/(?P<filepath>.+)$"
)


def parse_hf_url(url: str) -> dict | None:
    """Parse a HuggingFace URL into repo_id, revision, and filepath."""
    m = HF_URL_PATTERN.match(url.strip())
    if not m:
        return None
    return {
        "repo_id": m.group("repo_id"),
        "revision": m.group("revision"),
        "filepath": m.group("filepath"),
    }


def s3_key_exists(bucket: str, key: str) -> bool:
    """Check whether an S3 key exists using head_object. Returns False on any error (fail-safe)."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        # 404 means not found; any other error is also treated as "not found" (fail-safe)
        return False


def s3_prefix_has_objects(bucket: str, prefix: str) -> bool:
    """Check whether at least one object exists under an S3 prefix. Returns False on any error (fail-safe)."""
    try:
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return resp.get("KeyCount", 0) > 0
    except Exception:
        return False


def get_file_size(url: str) -> int | None:
    """Get file size via HEAD request."""
    try:
        _validate_url_scheme(url)
        req = Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urlopen(req, timeout=30) as resp:
            cl = resp.headers.get("Content-Length")
            if cl:
                return int(cl)
    except Exception as e:
        logger.warning("HEAD request failed", url=url[:80], error=str(e))
    return None


def download_range(url: str, start: int, end: int) -> bytes:
    """Download a byte range from a URL."""
    _validate_url_scheme(url)
    req = Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    req.add_header("User-Agent", "Mozilla/5.0")
    with urlopen(req, timeout=300) as resp:
        return resp.read()


def parallel_download_to_s3(url: str, bucket: str, s3_key: str, file_size: int) -> None:
    """Download using parallel HTTP Range requests and multipart S3 upload."""
    logger.info("  Method: parallel range download ({})", _fmt_size(file_size))
    t0 = time.time()

    mpu = s3_client.create_multipart_upload(Bucket=bucket, Key=s3_key)
    upload_id = mpu["UploadId"]
    parts = []

    try:
        ranges = []
        part_num = 1
        for offset in range(0, file_size, PART_SIZE):
            end = min(offset + PART_SIZE - 1, file_size - 1)
            ranges.append((part_num, offset, end))
            part_num += 1

        logger.info("Download plan: {} parts × {} each", len(ranges), _fmt_size(PART_SIZE))

        def download_and_upload_part(part_info):
            pn, start, end = part_info
            data = download_range(url, start, end)
            resp = s3_client.upload_part(
                Bucket=bucket,
                Key=s3_key,
                UploadId=upload_id,
                PartNumber=pn,
                Body=data,
            )
            return {"PartNumber": pn, "ETag": resp["ETag"]}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(download_and_upload_part, r): r[0] for r in ranges}
            for i, future in enumerate(as_completed(futures), 1):
                part_result = future.result()
                parts.append(part_result)
                pct = round(100 * i / len(ranges), 1)
                if i % 10 == 0 or i == len(ranges):
                    elapsed = time.time() - t0
                    logger.info("  ↳ {:.1f}% ({}/{} parts, {:.0f}s elapsed)", pct, i, len(ranges), elapsed)

        parts.sort(key=lambda p: p["PartNumber"])
        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

        elapsed = time.time() - t0
        speed_mbps = (file_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        logger.info("  ✅ Done ({} at {:.1f} MB/s in {:.0f}s)", _fmt_size(file_size), speed_mbps, elapsed)

    except Exception as e:
        logger.error("Parallel download failed, aborting", error=str(e))
        s3_client.abort_multipart_upload(Bucket=bucket, Key=s3_key, UploadId=upload_id)
        raise


def stream_upload_to_s3(url: str, bucket: str, s3_key: str, file_size: int | None = None) -> None:
    """Stream download directly to S3 using multipart upload."""
    _validate_url_scheme(url)
    size_str = _fmt_size(file_size) if file_size else "unknown size"
    logger.info("Starting stream upload ({})", size_str, url=url[:80])
    t0 = time.time()

    mpu = s3_client.create_multipart_upload(Bucket=bucket, Key=s3_key)
    upload_id = mpu["UploadId"]
    parts = []
    total_bytes = 0

    try:
        req = Request(url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urlopen(req, timeout=600) as resp:
            part_num = 1
            while True:
                chunk = resp.read(PART_SIZE)
                if not chunk:
                    break
                resp_part = s3_client.upload_part(
                    Bucket=bucket,
                    Key=s3_key,
                    UploadId=upload_id,
                    PartNumber=part_num,
                    Body=chunk,
                )
                parts.append({"PartNumber": part_num, "ETag": resp_part["ETag"]})
                total_bytes += len(chunk)
                if part_num % 5 == 0:
                    elapsed = time.time() - t0
                    if file_size:
                        pct = round(100 * total_bytes / file_size, 1)
                        logger.info(
                            "  ↳ {:.1f}% ({} / {}, {:.0f}s elapsed)",
                            pct,
                            _fmt_size(total_bytes),
                            _fmt_size(file_size),
                            elapsed,
                        )
                    else:
                        logger.info("  ↳ {} uploaded ({:.0f}s elapsed)", _fmt_size(total_bytes), elapsed)
                part_num += 1

        s3_client.complete_multipart_upload(
            Bucket=bucket,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

        elapsed = time.time() - t0
        logger.info("  ✅ Done ({} in {:.1f}s)", _fmt_size(total_bytes), elapsed)

    except Exception as e:
        logger.error("Stream upload failed, aborting", error=str(e))
        s3_client.abort_multipart_upload(Bucket=bucket, Key=s3_key, UploadId=upload_id)
        raise


def simple_upload(url: str, bucket: str, s3_key: str) -> None:
    """Download small file to memory and upload to S3."""
    _validate_url_scheme(url)
    logger.info("  Method: simple (small file)")
    t0 = time.time()

    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urlopen(req, timeout=300) as resp:
        data = resp.read()

    s3_client.put_object(Bucket=bucket, Key=s3_key, Body=data)

    elapsed = time.time() - t0
    logger.info("  ✅ Done ({} in {:.1f}s)", _fmt_size(len(data)), elapsed)


def download_to_local(url: str, local_path: str) -> None:
    """Download file to local disk."""
    _validate_url_scheme(url)
    logger.info("  Downloading to local disk...")
    t0 = time.time()

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    req = Request(url)
    req.add_header("User-Agent", "Mozilla/5.0")
    with urlopen(req, timeout=600) as resp, open(local_path, "wb") as f:
        shutil.copyfileobj(resp, f)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info("  Local download complete ({:.1f} MB in {:.1f}s)", size_mb, elapsed)


def extract_and_upload_zip(local_zip: str, bucket: str, s3_prefix: str) -> None:
    """Extract zip and upload contents to S3, preserving internal directory structure."""
    logger.info("  Extracting zip ({} files)...", "?")

    with zipfile.ZipFile(local_zip, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        logger.info("  Zip contains {} files", len(names))

        for j, name in enumerate(names, 1):
            data = zf.read(name)
            dest_key = f"{s3_prefix}/{name}"
            logger.info("  Uploading extracted {}/{}: {} ({})", j, len(names), name, _fmt_size(len(data)))
            s3_client.put_object(Bucket=bucket, Key=dest_key, Body=data)

    logger.info("  ✅ Zip extraction complete")


def download_and_upload(item: dict, bucket: str, index: int = 0, total: int = 0) -> dict:
    """Download a single item and upload to S3."""
    url = item["url"]
    s3_key = item["path"]
    extract = item.get("extract", False)
    filename = urlparse(url).path.split("/")[-1]
    model = _model_name(s3_key)

    logger.info("[{}/{}] 📦 {} — {}{}", index, total, model, filename, " [zip→extract]" if extract else "")
    logger.info("  src: {}", url)
    logger.info("  dst: s3://{}/{}", bucket, s3_key)

    # S3 existence check — skip if already present
    if extract:
        if s3_prefix_has_objects(bucket, s3_key):
            logger.info("  ⏭️  Skipped (objects already exist under prefix)")
            return {"status": "skipped", "path": s3_key, "reason": "already exists"}
    else:
        if s3_key_exists(bucket, s3_key):
            logger.info("  ⏭️  Skipped (S3 key already exists)")
            return {"status": "skipped", "path": s3_key, "reason": "already exists"}

    if extract:
        # Download to local, extract, upload contents
        local_path = os.path.join(LOCAL_DOWNLOAD_DIR, filename)
        download_to_local(url, local_path)
        try:
            extract_and_upload_zip(local_path, bucket, s3_key)
        finally:
            try:
                os.remove(local_path)
            except OSError:
                pass
        return {"status": "success", "path": s3_key, "extracted": True}

    # Regular file upload
    file_size = get_file_size(url)

    if file_size and file_size > PART_SIZE:
        # Try Range request
        try:
            test_req = Request(url)
            test_req.add_header("Range", "bytes=0-0")
            test_req.add_header("User-Agent", "Mozilla/5.0")
            with urlopen(test_req, timeout=30) as resp:
                if resp.status == 206:
                    parallel_download_to_s3(url, bucket, s3_key, file_size)
                    return {"status": "success", "path": s3_key}
        except Exception as e:
            logger.warning("Range test failed", error=str(e))

        stream_upload_to_s3(url, bucket, s3_key, file_size=file_size)
    elif file_size and file_size <= PART_SIZE:
        simple_upload(url, bucket, s3_key)
    else:
        # Unknown size, stream it
        stream_upload_to_s3(url, bucket, s3_key)

    return {"status": "success", "path": s3_key}


def _download_with_retry(item: dict, bucket: str, index: int, total: int) -> dict:
    """Attempt download_and_upload with exponential backoff on failure."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return download_and_upload(item, bucket, index=index, total=total)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "  ⚠️  Attempt {}/{} failed, retrying in {}s: {}",
                    attempt,
                    MAX_RETRIES,
                    wait,
                    str(e),
                )
                time.sleep(wait)
            else:
                logger.error("  ❌ All {} attempts failed: {}", MAX_RETRIES, str(e))
    raise last_error  # type: ignore[misc]


def _get_manifest_path() -> str:
    """Return the path to the per-config downloads manifest.

    Reads CONFIG_PREFIX and SHARED_PREFIX env vars to derive the construct_id.
    CONFIG_PREFIX = shared_prefix + construct_id (e.g. "abcvr", "devma").
    Strips SHARED_PREFIX to get construct_id (e.g. "vr", "ma"), then looks
    for ``{construct_id}_downloads.json`` next to this module.  Falls
    back to ``downloads.json`` for backward compatibility.
    """
    base_dir = os.path.dirname(__file__)
    config_prefix = os.environ.get("CONFIG_PREFIX", "")
    shared_prefix = os.environ.get("SHARED_PREFIX", "")
    if config_prefix:
        construct_id = config_prefix.removeprefix(shared_prefix) if shared_prefix else config_prefix
        per_config = os.path.join(base_dir, f"{construct_id}_downloads.json")
        if os.path.exists(per_config):
            return per_config
    # Fallback to legacy single-file manifest
    return os.path.join(base_dir, "downloads.json")


def main():
    """Main entry point."""
    # Get config from environment
    bucket = os.environ.get("MODELS_BUCKET")

    if not bucket:
        logger.error("MODELS_BUCKET environment variable not set")
        sys.exit(1)

    # Read downloads manifest baked into the container image at build time
    downloads_file = _get_manifest_path()
    if not os.path.exists(downloads_file):
        logger.error("Downloads manifest not found: {}", downloads_file)
        sys.exit(1)

    try:
        with open(downloads_file) as f:
            downloads = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to read downloads manifest: {}", e)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("MODEL DOWNLOAD JOB")
    logger.info("=" * 60)
    logger.info("Bucket: {}", bucket)
    logger.info("Total items: {}", len(downloads))

    # Compute total size estimate upfront
    total_size = 0
    for item in downloads:
        sz = get_file_size(item["url"])
        if sz:
            total_size += sz
    logger.info("Estimated total download size: {}", _fmt_size(total_size) if total_size else "unknown")
    logger.info("=" * 60)

    # Create local download directory
    os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

    # Process each download
    failed = []
    skipped = 0
    job_t0 = time.time()
    try:
        for i, item in enumerate(downloads, 1):
            try:
                result = _download_with_retry(item, bucket, index=i, total=len(downloads))
                if result.get("status") == "skipped":
                    skipped += 1
                logger.info("  ✅ Item {}/{} complete", i, len(downloads))
            except Exception as e:
                logger.error("  ❌ Item {}/{} FAILED: {}", i, len(downloads), str(e))
                failed.append(item)
    finally:
        # Clean up local download directory to conserve disk space
        if os.path.exists(LOCAL_DOWNLOAD_DIR):
            shutil.rmtree(LOCAL_DOWNLOAD_DIR, ignore_errors=True)
            logger.info("Cleaned up local download directory", path=LOCAL_DOWNLOAD_DIR)

    if failed:
        logger.error("{} of {} downloads FAILED", len(failed), len(downloads))
        for f in failed:
            logger.error("  ❌ {}", f.get("path", f.get("url", "")[:80]))
        sys.exit(1)

    job_elapsed = time.time() - job_t0
    downloaded = len(downloads) - len(failed) - skipped
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info("Bucket: {}", bucket)
    logger.info("Files: {}", len(downloads))
    logger.info("Skipped: {}", skipped)
    logger.info("Downloaded: {}", downloaded)
    logger.info("Failed: {}", len(failed))
    logger.info("Total time: {:.0f}s ({:.1f} min)", job_elapsed, job_elapsed / 60)
    for i, item in enumerate(downloads, 1):
        flag = " [extracted]" if item.get("extract") else ""
        logger.info("  {}/{}: s3://{}/{}{}", i, len(downloads), bucket, item["path"], flag)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
