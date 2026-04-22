"""
Retrieval ingestion load test Lambda handler.

Invoked ad-hoc via CLI to load-test the image retrieval ingestion pipeline
(S3 → SQS → ingest Lambda → Bedrock Titan embedding → OpenSearch Serverless).

Generates N random PNG images in /tmp, uploads them to the retrieval S3 bucket's
images/ prefix, monitors pipeline progress (SQS queue depth, CloudWatch metrics,
AOSS document count), and returns a structured JSON report with pass/fail status,
wall-clock time, throughput, peak concurrency, and time-series snapshots.

Environment variables (set by CDK):
    RETRIEVAL_BUCKET_NAME: S3 bucket for ingestion
    INGEST_QUEUE_URL: SQS queue URL for ingest events
    INGEST_FUNCTION_NAME: Ingest Lambda function name (for CloudWatch metrics)
    AOSS_ENDPOINT: OpenSearch Serverless collection endpoint
    AOSS_INDEX_NAME: OpenSearch index name for image vectors
"""

import os
import random
import shutil
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import boto3
from aws_lambda_powertools import Logger
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

# Environment variables
RETRIEVAL_BUCKET: str = os.environ.get("RETRIEVAL_BUCKET_NAME", "")
INGEST_QUEUE_URL: str = os.environ.get("INGEST_QUEUE_URL", "")
INGEST_FUNCTION_NAME: str = os.environ.get("INGEST_FUNCTION_NAME", "")
AOSS_ENDPOINT: str = os.environ.get("AOSS_ENDPOINT", "")
AOSS_INDEX_NAME: str = os.environ.get("AOSS_INDEX_NAME", "image-vectors")
REGION: str = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

logger = Logger()


def parse_event(event: dict) -> dict:
    """Extract and validate load test parameters from the Lambda event payload.

    Applies defaults for missing fields and validates constraints on
    ``num_images`` and ``timeout``.

    Args:
        event: Lambda invocation event payload.

    Returns:
        Dict with keys: num_images, poll_interval, timeout, image_size, cleanup.

    Raises:
        ValueError: If num_images is not a positive integer or timeout > 870.
    """
    num_images = event.get("num_images", 1000)
    poll_interval = event.get("poll_interval", 5)
    timeout = event.get("timeout", 600)
    image_size = event.get("image_size", 64)
    cleanup = event.get("cleanup", True)

    if not isinstance(num_images, int) or num_images <= 0:
        raise ValueError(f"num_images must be a positive integer, got {num_images!r}")

    if timeout > 870:
        raise ValueError(f"timeout must be <= 870 seconds (Lambda max minus 30s margin), got {timeout}")

    return {
        "num_images": num_images,
        "poll_interval": poll_interval,
        "timeout": timeout,
        "image_size": image_size,
        "cleanup": cleanup,
    }


def _make_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Create a minimal solid-color RGB PNG entirely in pure Python.

    Uses struct + zlib only — no Pillow or other native C extensions required,
    so the Lambda zip works on any architecture without platform-specific wheels.
    """

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    # IHDR: width, height, bit_depth=8, color_type=2 (RGB)
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))

    # IDAT: raw pixel rows, each prefixed with filter byte 0 (None)
    row = b"\x00" + bytes([r, g, b]) * width
    raw = row * height
    idat = _chunk(b"IDAT", zlib.compress(raw))

    iend = _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def generate_images(
    num_images: int,
    image_size: int,
    tmp_dir: str = "/tmp/loadtest-images",
) -> list[str]:
    """Generate N solid-color PNG files in tmp_dir.

    Each file is named ``loadtest-{i:04d}.png`` with a random RGB fill color,
    producing files of ~1-3 KB at 64×64. Uses a pure-Python PNG encoder to
    avoid Pillow's native C extension (which fails on cross-platform Lambda zips).

    Args:
        num_images: Number of images to generate.
        image_size: Width and height in pixels for each image.
        tmp_dir: Directory to write images into.

    Returns:
        List of absolute file paths for the generated PNGs.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    file_paths: list[str] = []

    for i in range(num_images):
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        png_bytes = _make_png(image_size, image_size, r, g, b)
        path = f"{tmp_dir}/loadtest-{i:04d}.png"
        with open(path, "wb") as f:
            f.write(png_bytes)
        file_paths.append(path)

    logger.info(
        "Generated images",
        num_images=len(file_paths),
        tmp_dir=tmp_dir,
    )
    return file_paths


def upload_images(
    file_paths: list[str],
    bucket: str,
    s3_client=None,
) -> tuple[int, int, float]:
    """Upload files to s3://{bucket}/images/ using ThreadPoolExecutor.

    Each file ``loadtest-XXXX.png`` is uploaded to key ``images/loadtest-XXXX.png``.
    After all uploads complete, the local /tmp files are cleaned up regardless of
    the ``cleanup`` event flag.

    Args:
        file_paths: List of local file paths to upload.
        bucket: S3 bucket name.
        s3_client: Optional boto3 S3 client (for dependency injection in tests).

    Returns:
        Tuple of (num_uploaded, num_failed, upload_start) where upload_start is
        the wall-clock timestamp recorded before the first upload.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    upload_start = time.time()
    num_uploaded = 0
    num_failed = 0

    def _upload_one(file_path: str) -> bool:
        try:
            key = f"images/{os.path.basename(file_path)}"
            s3_client.upload_file(file_path, bucket, key)
            return True
        except Exception:
            logger.exception("Failed to upload %s", file_path)
            return False

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(_upload_one, fp): fp for fp in file_paths}
        for future in as_completed(futures):
            if future.result():
                num_uploaded += 1
            else:
                num_failed += 1

    # Clean up /tmp files after upload regardless of cleanup flag
    if file_paths:
        shutil.rmtree(os.path.dirname(file_paths[0]), ignore_errors=True)

    upload_duration = time.time() - upload_start
    logger.info(
        "Upload complete",
        num_uploaded=num_uploaded,
        num_failed=num_failed,
        duration_seconds=round(upload_duration, 2),
    )

    return (num_uploaded, num_failed, upload_start)


def get_oss_client() -> OpenSearch:
    """Create and return an authenticated OpenSearch Serverless client.

    Uses IAM credentials with AWS SigV4 signing for the ``aoss`` service.
    Same pattern as ``lambdas/retrieval_ingest/index.py``.

    Returns:
        Authenticated OpenSearch client configured for HTTPS on port 443.
    """
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, "aoss")
    host = AOSS_ENDPOINT.replace("https://", "").replace("http://", "")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def monitor_pipeline(
    baseline_doc_count: int,
    num_images: int,
    poll_interval: float,
    timeout: float,
    queue_url: str,
    function_name: str,
    aoss_client: OpenSearch,
    index_name: str,
    upload_start_time: float,
    sqs_client=None,
    cw_client=None,
) -> dict:
    """Poll SQS, CloudWatch, and AOSS until completion or timeout.

    Each API call is wrapped in try/except so that transient failures do not
    abort the monitoring loop.

    Args:
        baseline_doc_count: AOSS document count captured before uploads began.
        num_images: Number of images uploaded (target for completion).
        poll_interval: Seconds between poll iterations.
        timeout: Maximum seconds to wait for completion.
        queue_url: SQS queue URL for the ingest queue.
        function_name: Ingest Lambda function name (for CloudWatch metrics).
        aoss_client: Authenticated OpenSearch client.
        index_name: AOSS index name.
        upload_start_time: Wall-clock timestamp when uploads started.
        sqs_client: Optional boto3 SQS client (dependency injection for tests).
        cw_client: Optional boto3 CloudWatch client (dependency injection for tests).

    Returns:
        Dict with: completed, wall_clock_seconds, peak_concurrent_executions,
        documents_indexed, document_count_final, time_series.
    """
    if sqs_client is None:
        sqs_client = boto3.client("sqs")
    if cw_client is None:
        cw_client = boto3.client("cloudwatch")

    time_series: list[dict] = []
    peak_concurrent = 0
    last_doc_count = baseline_doc_count
    completed = False
    elapsed = 0.0

    while elapsed < timeout and not completed:
        time.sleep(poll_interval)
        elapsed = time.time() - upload_start_time

        snapshot = {
            "elapsed_seconds": round(elapsed, 2),
            "sqs_visible": 0,
            "sqs_in_flight": 0,
            "concurrent_executions": 0,
            "cumulative_invocations": 0,
            "documents_indexed": 0,
        }

        # --- SQS poll ---
        sqs_visible = 0
        sqs_in_flight = 0
        try:
            attrs = sqs_client.get_queue_attributes(
                QueueUrl=queue_url,
                AttributeNames=[
                    "ApproximateNumberOfMessages",
                    "ApproximateNumberOfMessagesNotVisible",
                ],
            )["Attributes"]
            sqs_visible = int(attrs.get("ApproximateNumberOfMessages", 0))
            sqs_in_flight = int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
        except Exception:
            logger.exception("SQS get_queue_attributes failed")

        # --- CloudWatch poll ---
        concurrent_executions = 0
        cumulative_invocations = 0
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=5)
            cw_response = cw_client.get_metric_data(
                MetricDataQueries=[
                    {
                        "Id": "concurrent",
                        "MetricStat": {
                            "Metric": {
                                "Namespace": "AWS/Lambda",
                                "MetricName": "ConcurrentExecutions",
                                "Dimensions": [
                                    {
                                        "Name": "FunctionName",
                                        "Value": function_name,
                                    }
                                ],
                            },
                            "Period": 60,
                            "Stat": "Maximum",
                        },
                    },
                    {
                        "Id": "invocations",
                        "MetricStat": {
                            "Metric": {
                                "Namespace": "AWS/Lambda",
                                "MetricName": "Invocations",
                                "Dimensions": [
                                    {
                                        "Name": "FunctionName",
                                        "Value": function_name,
                                    }
                                ],
                            },
                            "Period": 60,
                            "Stat": "Sum",
                        },
                    },
                ],
                StartTime=start_time,
                EndTime=end_time,
            )
            for result in cw_response.get("MetricDataResults", []):
                values = result.get("Values", [])
                if result["Id"] == "concurrent" and values:
                    concurrent_executions = int(max(values))
                elif result["Id"] == "invocations" and values:
                    cumulative_invocations = int(sum(values))
        except Exception:
            logger.exception("CloudWatch get_metric_data failed")

        # --- AOSS poll ---
        documents_indexed = 0
        current_doc_count = last_doc_count
        try:
            result = aoss_client.count(index=index_name)
            current_doc_count = result["count"]
            last_doc_count = current_doc_count
            documents_indexed = current_doc_count - baseline_doc_count
        except Exception:
            logger.exception("AOSS count query failed")
            documents_indexed = last_doc_count - baseline_doc_count

        # --- Update snapshot ---
        snapshot["sqs_visible"] = sqs_visible
        snapshot["sqs_in_flight"] = sqs_in_flight
        snapshot["concurrent_executions"] = concurrent_executions
        snapshot["cumulative_invocations"] = cumulative_invocations
        snapshot["documents_indexed"] = documents_indexed

        peak_concurrent = max(peak_concurrent, concurrent_executions)
        time_series.append(snapshot)

        logger.info("Monitor snapshot", **snapshot)

        # --- Completion check ---
        if documents_indexed >= num_images or sqs_visible == 0 and sqs_in_flight == 0 and documents_indexed > 0:
            completed = True

    return {
        "completed": completed,
        "wall_clock_seconds": round(time.time() - upload_start_time, 2),
        "peak_concurrent_executions": peak_concurrent,
        "documents_indexed": last_doc_count - baseline_doc_count,
        "document_count_final": last_doc_count,
        "time_series": time_series,
    }


def cleanup_artifacts(
    bucket: str,
    aoss_client,
    index_name: str,
    s3_client=None,
) -> dict:
    """Delete test artifacts from S3 and AOSS.

    S3: delete objects with ``images/loadtest-`` and ``base64/images/loadtest-``
    prefixes in batches of 1000 (the ``delete_objects`` API limit).
    AOSS: ``delete_by_query`` matching documents where ``description`` starts
    with ``images/loadtest-``.

    Args:
        bucket: S3 bucket name containing test artifacts.
        aoss_client: Authenticated OpenSearch client.
        index_name: AOSS index name.
        s3_client: Optional boto3 S3 client (for dependency injection in tests).

    Returns:
        Dict with keys: s3_objects_deleted, aoss_documents_deleted, errors.
    """
    if s3_client is None:
        s3_client = boto3.client("s3")

    s3_deleted = 0
    aoss_deleted = 0
    errors: list[str] = []

    # --- S3 cleanup ---
    for prefix in ["images/loadtest-", "base64/images/loadtest-"]:
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            batch: list[dict] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    batch.append({"Key": obj["Key"]})
                    if len(batch) == 1000:
                        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                        s3_deleted += len(batch)
                        batch = []
            if batch:
                s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                s3_deleted += len(batch)
        except Exception as exc:
            error_msg = f"S3 cleanup failed for prefix {prefix}: {exc}"
            logger.exception(error_msg)
            errors.append(error_msg)

    # --- AOSS cleanup ---
    try:
        # AOSS does not support delete_by_query — scroll and delete individually
        while True:
            results = aoss_client.search(
                index=index_name,
                body={
                    "size": 100,
                    "_source": ["description"],
                    "query": {"prefix": {"description": "images/loadtest-"}},
                },
            )
            hits = results.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                try:
                    aoss_client.delete(index=index_name, id=hit["_id"])
                    aoss_deleted += 1
                except Exception as exc:
                    errors.append(f"Failed to delete doc {hit['_id']}: {exc}")
    except Exception as exc:
        error_msg = f"AOSS cleanup failed: {exc}"
        logger.exception(error_msg)
        errors.append(error_msg)

    logger.info(
        "Cleanup complete",
        s3_objects_deleted=s3_deleted,
        aoss_documents_deleted=aoss_deleted,
        errors=errors,
    )

    return {
        "s3_objects_deleted": s3_deleted,
        "aoss_documents_deleted": aoss_deleted,
        "errors": errors,
    }


def assemble_result(
    num_uploaded: int,
    num_failed: int,
    num_images: int,
    monitor_result: dict,
    cleanup_results: dict | None = None,
) -> dict:
    """Build the JSON result object for the load test.

    Args:
        num_uploaded: Number of images successfully uploaded to S3.
        num_failed: Number of images that failed to upload.
        num_images: Total number of images requested.
        monitor_result: Dict returned by ``monitor_pipeline``.
        cleanup_results: Optional dict returned by ``cleanup_artifacts``.

    Returns:
        Structured result dict with status, metrics, time_series, and cleanup info.
    """
    documents_indexed = monitor_result["documents_indexed"]
    wall_clock_seconds = monitor_result["wall_clock_seconds"]

    status = "PASS" if documents_indexed >= num_images else "FAIL"
    throughput = documents_indexed / wall_clock_seconds if wall_clock_seconds > 0 else 0.0

    result = {
        "status": status,
        "num_images_uploaded": num_uploaded,
        "num_upload_failures": num_failed,
        "num_documents_indexed": documents_indexed,
        "wall_clock_seconds": wall_clock_seconds,
        "throughput_images_per_second": round(throughput, 2),
        "peak_concurrent_executions": monitor_result["peak_concurrent_executions"],
        "document_count_final": monitor_result["document_count_final"],
        "time_series": monitor_result["time_series"],
        "cleanup_results": cleanup_results or {},
    }

    if status == "FAIL":
        result["shortfall"] = num_images - documents_indexed

    return result


def purge_index(aoss_client: OpenSearch, index_name: str) -> dict:
    """Delete the AOSS index entirely and recreate it empty.

    AOSS does not support ``delete_by_query`` and individual document
    deletes have ID encoding issues. The simplest approach is to drop
    the index — the ingest Lambda's ``ensure_index`` will recreate it
    on the next ingestion.

    Args:
        aoss_client: Authenticated OpenSearch client.
        index_name: AOSS index name to purge.

    Returns:
        Dict with keys: deleted, errors.
    """
    errors: list[str] = []
    deleted = 0
    try:
        if aoss_client.indices.exists(index=index_name):
            count = aoss_client.count(index=index_name).get("count", 0)
            aoss_client.indices.delete(index=index_name)
            deleted = count
            logger.info("Deleted index", index=index_name, documents=deleted)
        else:
            logger.info("Index does not exist, nothing to purge", index=index_name)
    except Exception as exc:
        error_msg = f"Purge failed: {exc}"
        logger.exception(error_msg)
        errors.append(error_msg)

    return {"deleted": deleted, "errors": errors}


def lambda_handler(event: dict, context: object) -> dict:
    """Load test Lambda entry point.

    Supports two modes via the ``action`` field:

    - ``action: "purge"`` — Deletes ALL documents from the AOSS index.
    - Default (no action) — Runs the full load test pipeline.

    Args:
        event: Lambda invocation event payload.
        context: Lambda context object (unused).

    Returns:
        Structured JSON result dict.
    """
    # --- Purge mode: delete all AOSS documents ---
    if event.get("action") == "purge":
        aoss_client = get_oss_client()
        return purge_index(aoss_client, AOSS_INDEX_NAME)

    # --- Delete by prefix: selectively remove docs matching a description prefix ---
    if event.get("action") == "delete_by_prefix":
        prefix = event.get("prefix", "")
        if not prefix:
            return {"error": "prefix is required for delete_by_prefix action"}
        aoss_client = get_oss_client()
        return delete_by_prefix(aoss_client, AOSS_INDEX_NAME, prefix)

    params = parse_event(event)
    num_images = params["num_images"]
    poll_interval = params["poll_interval"]
    timeout = params["timeout"]
    image_size = params["image_size"]
    cleanup = params["cleanup"]

    aoss_client = get_oss_client()

    # Capture baseline document count
    try:
        baseline_doc_count = aoss_client.count(index=AOSS_INDEX_NAME)["count"]
    except Exception:
        logger.exception("Failed to get baseline doc count, defaulting to 0")
        baseline_doc_count = 0

    file_paths = generate_images(num_images, image_size)
    num_uploaded, num_failed, upload_start = upload_images(file_paths, RETRIEVAL_BUCKET)

    monitor_result = monitor_pipeline(
        baseline_doc_count,
        num_images,
        poll_interval,
        timeout,
        INGEST_QUEUE_URL,
        INGEST_FUNCTION_NAME,
        aoss_client,
        AOSS_INDEX_NAME,
        upload_start,
    )

    result = assemble_result(num_uploaded, num_failed, num_images, monitor_result)

    if cleanup:
        cleanup_results = cleanup_artifacts(RETRIEVAL_BUCKET, aoss_client, AOSS_INDEX_NAME)
        result["cleanup_results"] = cleanup_results

    logger.info(
        "Load test complete",
        **{k: v for k, v in result.items() if k != "time_series"},
    )

    return result


def delete_by_prefix(aoss_client: OpenSearch, index_name: str, prefix: str) -> dict:
    """Delete AOSS documents whose description field starts with a given prefix.

    The ingest Lambda stores the S3 key (e.g. 'images/loadtest-0000.png') as
    the ``description`` field. This allows selective deletion by S3 key prefix.

    Uses scroll + bulk delete since AOSS doesn't support delete_by_query.

    Args:
        aoss_client: Authenticated OpenSearch client.
        index_name: AOSS index name.
        prefix: Description prefix to match (e.g. 'images/loadtest').

    Returns:
        Dict with keys: deleted, errors.
    """
    errors: list[str] = []
    deleted = 0

    try:
        if not aoss_client.indices.exists(index=index_name):
            return {"deleted": 0, "errors": [], "message": "Index does not exist"}

        # Search for matching docs
        body = {
            "query": {"prefix": {"description": prefix}},
            "_source": False,
            "size": 10000,
        }
        resp = aoss_client.search(index=index_name, body=body)
        hits = resp.get("hits", {}).get("hits", [])

        if not hits:
            logger.info("No documents matching prefix", prefix=prefix)
            return {"deleted": 0, "errors": []}

        # Delete each matching doc individually
        for hit in hits:
            try:
                aoss_client.delete(index=index_name, id=hit["_id"])
                deleted += 1
            except Exception as exc:
                errors.append(f"Failed to delete {hit['_id']}: {exc}")

        logger.info("Deleted documents by prefix", prefix=prefix, deleted=deleted, errors=len(errors))

    except Exception as exc:
        error_msg = f"Delete by prefix failed: {exc}"
        logger.exception(error_msg)
        errors.append(error_msg)

    return {"deleted": deleted, "errors": errors}
