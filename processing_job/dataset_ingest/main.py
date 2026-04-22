"""dataset_ingest: Seeds the retrieval ingestion pipeline with images and generates video prompts.

Downloads a dataset (e.g. Unsplash Lite) via a dynamically-imported dataset script,
uploads images to the retrieval ingestion S3 bucket (triggering S3 → SQS → Lambda → AOSS),
generates video prompts via a Strands Agent (Bedrock Amazon Nova Lite), and writes
VisualEntry JSON to the SageMaker output channel.

Usage:
    python3 main.py --run       # Normal mode
    python3 main.py --cleanup   # Cleanup mode
"""

import argparse
import importlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from loguru import logger
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SM_OUTPUT_DIR = "/opt/ml/processing/output/output/"
OUTPUT_FILENAME = "inputs_t2v.json"
DEFAULT_MODEL_ID = "us.amazon.nova-2-lite-v1:0"

AOSS_POLL_INTERVAL_S = 30
AOSS_POLL_TIMEOUT_S = 300

SYSTEM_PROMPT = """\
You are a video prompt generation agent. Given an image,
visually analyze its content and generate a cinematic video generation
prompt that describes motion, camera movement, lighting, and mood
suitable for an image-to-video model.

Respond with only the video prompt text in paragraph form, no JSON wrapping.\
"""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

RETRIEVAL_BUCKET_NAME = os.environ.get("RETRIEVAL_BUCKET_NAME", "")
AOSS_INDEX_NAME = os.environ.get("AOSS_INDEX_NAME", "")
NUM_PROMPTS = int(os.environ.get("NUM_PROMPTS", "100"))
TEST_IMAGE_COUNT = int(os.environ.get("TEST_IMAGE_COUNT", "25000"))
DATASET_URL = os.environ.get("DATASET_URL", "")
DATASET_SCRIPT = os.environ.get("DATASET_SCRIPT", "")
PROMPT_WORKERS = int(os.environ.get("PROMPT_WORKERS", "10"))


def _resolve_aoss_endpoint() -> str:
    """Resolve AOSS endpoint from env var or SSM parameter."""
    endpoint = os.environ.get("AOSS_ENDPOINT", "")
    if endpoint:
        return endpoint
    ssm_name = os.environ.get("AOSS_ENDPOINT_SSM", "")
    if ssm_name:
        logger.info("Resolving AOSS endpoint from SSM: {}", ssm_name)
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        ssm = boto3.client("ssm", region_name=region)
        resp = ssm.get_parameter(Name=ssm_name)
        endpoint = resp["Parameter"]["Value"]
        logger.info("Resolved AOSS endpoint: {}", endpoint)
        return endpoint
    return ""


AOSS_ENDPOINT = _resolve_aoss_endpoint()


# ---------------------------------------------------------------------------
# OpenSearch client helpers
# ---------------------------------------------------------------------------


def get_sigv4_auth(region: str) -> AWS4Auth:
    """Build SigV4 auth for OpenSearch Serverless (service=aoss)."""
    creds = boto3.Session().get_credentials().get_frozen_credentials()
    return AWS4Auth(
        creds.access_key,
        creds.secret_key,
        region,
        "aoss",
        session_token=creds.token,
    )


def build_opensearch_client(endpoint: str, region: str) -> OpenSearch:
    """Create an OpenSearch client with SigV4 auth for Serverless."""
    host = endpoint.replace("https://", "").replace("http://", "")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=get_sigv4_auth(region),
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


# ---------------------------------------------------------------------------
# Dynamic dataset script import
# ---------------------------------------------------------------------------


_ALLOWED_DATASET_MODULES = {"unsplash", "open_images_v7"}


def import_dataset_script(script_name: str):
    """Dynamically import the dataset loader script (e.g. 'unsplash.py').

    The script must be in the same directory as main.py and implement:
        load_and_upload(s3_client, bucket, limit) -> list[dict]
    Each dict has keys: id, description, s3_uri.
    """
    module_name = script_name.removesuffix(".py")
    if module_name not in _ALLOWED_DATASET_MODULES:
        raise ValueError(f"Unknown dataset module {module_name!r} — allowed: {sorted(_ALLOWED_DATASET_MODULES)}")
    logger.info("Importing dataset script: {} (module: {})", script_name, module_name)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        logger.error("Dataset script '{}' not found", script_name)
        raise
    if not hasattr(module, "load_and_upload"):
        raise AttributeError(f"Dataset script '{script_name}' missing required 'load_and_upload' function")
    return module


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def create_agent(model_id: str = DEFAULT_MODEL_ID):
    """Create a Strands Agent for video prompt generation."""
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    model = BedrockModel(
        model_id=model_id,
        additional_request_fields={"inferenceConfig": {"reasoningConfig": {"type": "DISABLED"}}},
    )
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)
    logger.info("Created Strands Agent with model: {}", model_id)
    return agent


def build_multimodal_content(image_bytes: bytes) -> list[dict]:
    """Construct multimodal content blocks for the Strands Agent.

    Returns a list of two content blocks:
    1. A text block with the analysis instruction
    2. An image block with the raw JPEG bytes
    """
    return [
        {
            "text": "Analyze this image and generate a cinematic video prompt describing motion, camera movement, lighting, and mood."
        },
        {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
    ]


def generate_prompt(agent, s3_uri: str, s3_client) -> str:
    """Download image from S3 and invoke Strands Agent with multimodal content.

    Parses bucket and key from the S3 URI, downloads image bytes via
    s3_client.get_object, and constructs two content blocks:
    1. {"text": "instruction text"}
    2. {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}}

    Returns the generated prompt text.
    Raises ValueError if the agent returns an empty response.
    """
    # Parse bucket and key from s3://bucket/key
    path = s3_uri.replace("s3://", "")
    bucket = path.split("/", 1)[0]
    key = path.split("/", 1)[1]

    image_bytes = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    content = build_multimodal_content(image_bytes)
    response = agent(content)
    prompt_text = str(response).strip()
    if not prompt_text:
        raise ValueError("Agent returned empty response")
    return prompt_text


# ---------------------------------------------------------------------------
# AOSS verification
# ---------------------------------------------------------------------------


def verify_aoss(
    oss_client: OpenSearch,
    index: str,
    expected_count: int,
    poll_interval: int = AOSS_POLL_INTERVAL_S,
    timeout: int = AOSS_POLL_TIMEOUT_S,
) -> int:
    """Poll AOSS document count until it meets expected_count or timeout.

    Polls every ``poll_interval`` seconds for up to ``timeout`` seconds.
    Logs the result and any discrepancy. Never raises — returns the final count.
    """
    logger.info(
        "Starting AOSS verification: expecting {} documents in index '{}' (poll={}s, timeout={}s)",
        expected_count,
        index,
        poll_interval,
        timeout,
    )
    start = time.time()
    doc_count = 0

    while True:
        elapsed = time.time() - start
        try:
            resp = oss_client.count(index=index)
            doc_count = resp.get("count", 0)
            logger.info(
                "AOSS verification: {}/{} documents indexed ({:.0f}s elapsed)",
                doc_count,
                expected_count,
                elapsed,
            )
            if doc_count >= expected_count:
                logger.info("AOSS verification passed: all {} documents indexed", expected_count)
                return doc_count
        except Exception as e:
            logger.warning("AOSS count query failed ({:.0f}s elapsed): {}", elapsed, e)

        if elapsed >= timeout:
            break

        time.sleep(poll_interval)

    discrepancy = expected_count - doc_count
    if discrepancy > 0:
        logger.warning(
            "AOSS verification: {}/{} documents indexed after {:.0f}s — {} missing",
            doc_count,
            expected_count,
            time.time() - start,
            discrepancy,
        )
    else:
        logger.info("AOSS verification passed: {} documents indexed", doc_count)
    return doc_count


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def _delete_s3_prefix(s3_client, bucket: str, prefix: str) -> int:
    """Delete all objects under a prefix in S3. Returns count of deleted objects."""
    deleted = 0
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = page.get("Contents", [])
        if not objects:
            continue
        delete_keys = [{"Key": obj["Key"]} for obj in objects]
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": delete_keys})
        deleted += len(delete_keys)
        logger.info("Deleted {} objects from s3://{}/{} (batch)", len(delete_keys), bucket, prefix)
    return deleted


def _delete_aoss_documents(oss_client: OpenSearch, index: str) -> int:
    """Delete all documents from an AOSS index. Returns count of deleted documents."""
    try:
        resp = oss_client.count(index=index)
        initial_count = resp.get("count", 0)
    except Exception as e:
        logger.warning("Could not get AOSS document count before cleanup: {}", e)
        initial_count = 0

    if initial_count == 0:
        logger.info("AOSS index '{}' is already empty", index)
        return 0

    try:
        resp = oss_client.delete_by_query(
            index=index,
            body={"query": {"match_all": {}}},
            refresh=True,
        )
        deleted = resp.get("deleted", 0)
        logger.info("Deleted {} documents from AOSS index '{}'", deleted, index)
        return deleted
    except Exception as e:
        logger.error("Failed to delete AOSS documents: {}", e)
        raise


def cleanup(s3_client, bucket: str, oss_client: OpenSearch, index: str) -> None:
    """Delete all images from S3 and all documents from AOSS.

    Deletes objects under ``images/`` and ``base64/`` prefixes, then
    removes all documents from the AOSS index.
    """
    logger.info("Starting cleanup: bucket={}, index={}", bucket, index)

    images_deleted = _delete_s3_prefix(s3_client, bucket, "images/")
    base64_deleted = _delete_s3_prefix(s3_client, bucket, "base64/")
    logger.info(
        "S3 cleanup complete: {} images/ objects + {} base64/ objects deleted",
        images_deleted,
        base64_deleted,
    )

    aoss_deleted = _delete_aoss_documents(oss_client, index)
    logger.info(
        "Cleanup complete: {} S3 objects deleted, {} AOSS documents deleted",
        images_deleted + base64_deleted,
        aoss_deleted,
    )


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_visual_entries(entries: list[dict], output_dir: str) -> str:
    """Write VisualEntry dicts as a JSON array to inputs_t2v.json.

    Each dict must have keys: id, prompt, image.
    Returns the path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info("Wrote {} VisualEntry objects to {}", len(entries), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_normal_mode() -> None:
    """Normal mode: load dataset → generate prompts → verify AOSS → write output."""
    # Enforce num_prompts <= test_image_count at runtime
    num_prompts = min(NUM_PROMPTS, TEST_IMAGE_COUNT)
    logger.info(
        "Normal mode | bucket={} | index={} | num_prompts={} | test_image_count={} | dataset_url={} | dataset_script={}",
        RETRIEVAL_BUCKET_NAME,
        AOSS_INDEX_NAME,
        num_prompts,
        TEST_IMAGE_COUNT,
        DATASET_URL,
        DATASET_SCRIPT,
    )

    if not RETRIEVAL_BUCKET_NAME:
        logger.error("RETRIEVAL_BUCKET_NAME not set")
        sys.exit(1)
    if not DATASET_SCRIPT:
        logger.error("DATASET_SCRIPT not set")
        sys.exit(1)

    # 1. Import dataset script and load/upload images
    dataset_module = import_dataset_script(DATASET_SCRIPT)
    s3_client = boto3.client("s3")
    uploaded_images = dataset_module.load_and_upload(s3_client, RETRIEVAL_BUCKET_NAME, TEST_IMAGE_COUNT)
    logger.info("Dataset upload complete: {} images uploaded", len(uploaded_images))

    if not uploaded_images:
        logger.error("No images were uploaded — exiting")
        sys.exit(1)

    # 2. Generate video prompts via Strands Agent (parallel)
    #    Select images evenly spaced across the uploaded set so prompts
    #    cover a diverse range of the dataset rather than clustering at the start.
    visual_entries: list[dict] = []
    skipped = 0

    total = len(uploaded_images)
    step = max(1, total // num_prompts)
    candidates = uploaded_images[::step][:num_prompts]
    logger.info(
        "Generating {} prompts with {} workers from {} uploaded images",
        len(candidates),
        PROMPT_WORKERS,
        total,
    )

    # Thread-local agents (Strands Agent is not thread-safe)
    _thread_local = threading.local()

    def _get_agent():
        if not hasattr(_thread_local, "agent"):
            _thread_local.agent = create_agent()
        return _thread_local.agent

    def _generate_for_image(img: dict) -> dict | None:
        try:
            prompt_text = generate_prompt(_get_agent(), img["s3_uri"], s3_client)
            return {"id": img["id"], "prompt": prompt_text, "image": img["s3_uri"]}
        except Exception as e:
            logger.error("Failed to generate prompt for image {}: {}", img["id"], e)
            return None

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=PROMPT_WORKERS) as executor:
        futures = {executor.submit(_generate_for_image, img): img for img in candidates}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                if result is not None:
                    visual_entries.append(result)
                    if len(visual_entries) % 50 == 0:
                        logger.info("Generated {}/{} prompts", len(visual_entries), len(candidates))
                else:
                    skipped += 1

    logger.info(
        "Prompt generation complete: {}/{} prompts generated, {} skipped",
        len(visual_entries),
        num_prompts,
        skipped,
    )

    # 3. Verify AOSS indexing
    if AOSS_ENDPOINT and AOSS_INDEX_NAME:
        region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION", "us-east-1")
        try:
            oss_client = build_opensearch_client(AOSS_ENDPOINT, region)
            verify_aoss(oss_client, AOSS_INDEX_NAME, len(uploaded_images))
        except Exception as e:
            logger.warning("AOSS verification failed (non-fatal): {}", e)
    else:
        logger.warning("AOSS_ENDPOINT or AOSS_INDEX_NAME not set — skipping verification")

    # 4. Write VisualEntry output
    write_visual_entries(visual_entries, SM_OUTPUT_DIR)

    logger.info("Normal mode complete")


def run_cleanup_mode() -> None:
    """Cleanup mode: delete all S3 objects and AOSS documents, then exit."""
    logger.info("Cleanup mode | bucket={} | index={}", RETRIEVAL_BUCKET_NAME, AOSS_INDEX_NAME)

    if not RETRIEVAL_BUCKET_NAME:
        logger.error("RETRIEVAL_BUCKET_NAME not set")
        sys.exit(1)

    s3_client = boto3.client("s3")
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION", "us-east-1")

    oss_client = None
    if AOSS_ENDPOINT and AOSS_INDEX_NAME:
        try:
            oss_client = build_opensearch_client(AOSS_ENDPOINT, region)
        except Exception as e:
            logger.error("Failed to create AOSS client: {}", e)
            sys.exit(1)
    else:
        logger.error("AOSS_ENDPOINT or AOSS_INDEX_NAME not set — cannot clean AOSS")
        sys.exit(1)

    cleanup(s3_client, RETRIEVAL_BUCKET_NAME, oss_client, AOSS_INDEX_NAME)
    logger.info("Cleanup mode complete")


def main() -> None:
    """CLI entry point: parse --run or --cleanup and dispatch."""
    parser = argparse.ArgumentParser(description="Setup container: dataset download, upload, and prompt generation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Normal mode: download, upload, generate prompts, verify")
    group.add_argument("--cleanup", action="store_true", help="Cleanup mode: delete S3 objects and AOSS documents")
    args = parser.parse_args()

    if args.run:
        run_normal_mode()
    elif args.cleanup:
        run_cleanup_mode()


if __name__ == "__main__":
    main()
    sys.exit(0)
