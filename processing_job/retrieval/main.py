"""retrieval: Queries OpenSearch Serverless for images matching text prompts.

Reads text prompts from the SageMaker input channel, performs kNN search
against an OpenSearch Serverless collection using Bedrock embeddings
(supports both Titan multimodal and Nova multimodal models),
downloads matched images from S3, and writes them to the output channel
with the document ID as the filename.

Supports two input modes:
1. Legacy: read_prompts() reads .txt/.json files and returns prompt strings.
2. Shard-based: read_input_shards() reads {id}.json VragOutputEntry shards,
   uses retrieval_query for AOSS search, and writes VisualEntry shards with
   video_prompt as prompt and S3 URI as image (no image download).

Usage: python3 main.py --retrieve
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import boto3
from loguru import logger
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

try:
    from common.models import VisualEntry
except ImportError:
    from processing_job.common.models import VisualEntry

from schema.columns import COL

SM_INPUT_DIR = "/opt/ml/processing/input/input"
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "/opt/ml/processing/output/output")
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "")
STEP_NAME = os.environ.get("STEP_NAME", "retrieval")
EXECUTION_ID = os.environ.get("EXECUTION_ID", "")

AOSS_INDEX_NAME = os.environ.get("AOSS_INDEX_NAME", "image-vectors")
QUERY_K = int(os.environ.get("QUERY_K", "5"))
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-image-v1")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))
VECTOR_FIELD = "image_vector"


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


def _is_nova_model(model_id: str) -> bool:
    """Check if the model ID is a Nova embedding model."""
    return "nova" in model_id.lower()


def embed_query(text: str, bedrock_client, model_id: str) -> list[float]:
    """Embed query text via Bedrock embedding model.

    Supports both Titan multimodal and Nova multimodal embedding models.
    For retrieval queries, Nova uses IMAGE_RETRIEVAL purpose to match
    the GENERIC_INDEX purpose used during ingestion.
    """
    if _is_nova_model(model_id):
        request_body = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "IMAGE_RETRIEVAL",
                "embeddingDimension": EMBEDDING_DIMENSION,
                "text": {
                    "truncationMode": "END",
                    "value": text,
                },
            },
        }
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        return list(result["embeddings"][0]["embedding"])
    else:
        # Titan format
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps({"inputText": text}),
            contentType="application/json",
        )
        body = json.loads(response["body"].read())
        return body["embedding"]


def search_images(
    client: OpenSearch,
    query_vector: list[float],
    index_name: str,
    k: int,
) -> list[dict]:
    """Perform kNN vector search and return matched documents.

    Args:
        client: OpenSearch client instance.
        query_vector: Embedding vector for the text prompt.
        index_name: OpenSearch index to query.
        k: Number of nearest neighbors to retrieve.

    Returns:
        List of dicts with keys: doc_id, image_s3_uri, score.
    """
    body = {
        "size": k,
        "_source": {"exclude": [VECTOR_FIELD]},
        "query": {
            "knn": {VECTOR_FIELD: {"vector": query_vector, "k": k}},
        },
    }

    response = client.search(index=index_name, body=body)
    hits = response.get("hits", {}).get("hits", [])

    results = []
    for hit in hits:
        source = hit.get("_source", {})
        results.append(
            {
                "doc_id": hit.get("_id", ""),
                "image_s3_uri": source.get("image_s3_uri", ""),
                "score": hit.get("_score", 0.0),
            }
        )
    return results


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse 's3://bucket/key' into (bucket, key) tuple.

    Args:
        uri: S3 URI string.

    Returns:
        Tuple of (bucket, key).

    Raises:
        ValueError: If the URI is not a valid S3 URI.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with s3://): {uri}")
    stripped = uri[5:]
    bucket, _, key = stripped.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI (missing bucket or key): {uri}")
    return bucket, key


def download_image(s3_client, bucket: str, key: str) -> bytes:
    """Download an image from S3 and return its bytes.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Image bytes.
    """
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def read_prompts(input_dir: str) -> list[str]:
    """Read text prompts from the input channel directory.

    Reads all .txt and .json files from the input directory.
    For .json files, extracts the 'retrieval_query' or 'prompt' field.
    For .txt files, reads the entire content as a single prompt.

    Args:
        input_dir: Path to the SageMaker input channel directory.

    Returns:
        List of prompt strings.
    """
    prompts: list[str] = []

    if not os.path.isdir(input_dir):
        return prompts

    for filename in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        if filename.endswith(".json"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    prompt = data.get("retrieval_query") or data.get("prompt", "")
                    if prompt:
                        prompts.append(prompt.strip())
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            prompt = item.get("retrieval_query") or item.get("prompt", "")
                            if prompt:
                                prompts.append(prompt.strip())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to parse JSON file {}: {}", filename, exc)

        elif filename.endswith(".txt"):
            try:
                with open(filepath, encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    prompts.append(text)
            except OSError as exc:
                logger.warning("Failed to read text file {}: {}", filename, exc)

    return prompts


def read_input_shards(input_dir: str) -> list[dict]:
    """Read JSON shard files ({id}.json) from input directory.

    Each file is a single JSON object (not an array). Supports both
    VragOutputEntry shards and legacy formats. Returns list of dicts
    preserving all fields (id, retrieval_query, video_prompt, prompt, image).

    Args:
        input_dir: Path to the input directory containing JSON shards.

    Returns:
        List of dicts, one per shard file.
    """
    shards: list[dict] = []

    if not os.path.isdir(input_dir):
        logger.warning("Input directory does not exist: {}", input_dir)
        return shards

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                shards.append(data)
            else:
                logger.warning("Skipping non-dict JSON in {}", filename)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse shard file {}: {}", filename, exc)

    logger.info("Read {} input shards from {}", len(shards), input_dir)
    return shards


def write_visual_entry_shard(entry_id: str, prompt: str, image_s3_uri: str, output_dir: str) -> None:
    """Write a VisualEntry JSON shard as {id}.json.

    Uses the VisualEntry model for validation before writing.

    Args:
        entry_id: Entry identifier, used as the filename stem.
        prompt: The prompt field (typically video_prompt from upstream).
        image_s3_uri: S3 URI of the matched image.
        output_dir: Directory to write the shard file to.
    """
    entry = VisualEntry(id=entry_id, prompt=prompt, image=image_s3_uri)
    output_path = os.path.join(output_dir, f"{entry_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entry.model_dump(), f, indent=2)
    logger.info("Wrote VisualEntry shard: {}", output_path)


def get_file_extension(s3_key: str) -> str:
    """Extract file extension from an S3 key, defaulting to .jpg.

    Args:
        s3_key: S3 object key.

    Returns:
        File extension including the dot (e.g. '.jpg', '.png').
    """
    _, ext = os.path.splitext(s3_key)
    return ext if ext else ".jpg"


def main() -> None:
    """Entry point for the retrieval processing job."""
    parser = argparse.ArgumentParser(description="Retrieval step")
    parser.add_argument("--retrieve", action="store_true", required=True)
    parser.parse_args()

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    # Log configuration
    logger.info("=== Retrieval Step Configuration ===")
    logger.info("AOSS_ENDPOINT: {}", AOSS_ENDPOINT)
    logger.info("AOSS_INDEX_NAME: {}", AOSS_INDEX_NAME)
    logger.info("QUERY_K: {}", QUERY_K)
    logger.info("EMBEDDING_MODEL_ID: {}", EMBEDDING_MODEL_ID)
    logger.info("SM_INPUT_DIR: {}", SM_INPUT_DIR)
    logger.info("LOCAL_OUTPUT_DIR: {}", LOCAL_OUTPUT_DIR)

    # Validate required configuration
    if not AOSS_ENDPOINT:
        logger.error("AOSS_ENDPOINT environment variable is not set")
        sys.exit(1)

    # Determine AWS region from endpoint or environment
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION", "us-east-1")
    logger.info("AWS region: {}", region)

    # Check for shard-based input (JSON shard files from vrag_llm)
    has_json_shards = False
    if os.path.isdir(SM_INPUT_DIR):
        has_json_shards = any(f.endswith(".json") for f in os.listdir(SM_INPUT_DIR))

    if has_json_shards:
        _run_shard_flow(region)
    else:
        _run_legacy_flow(region)

    logger.info("Retrieval step completed successfully.")


def _run_shard_flow(region: str) -> None:
    """Shard-based flow: read VragOutputEntry shards, query AOSS, write VisualEntry shards."""
    logger.info("=== Shard-based retrieval flow ===")

    shards = read_input_shards(SM_INPUT_DIR)
    if not shards:
        logger.error("No input shards found in: {}", SM_INPUT_DIR)
        sys.exit(1)

    logger.info("Found {} input shards to process", len(shards))

    # Initialize clients
    logger.info("Initializing OpenSearch client...")
    os_client = build_opensearch_client(AOSS_ENDPOINT, region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    logger.info("Clients initialized")

    # Initialize DynamoDB if configured
    db_ops = None
    if DYNAMODB_TABLE_NAME:
        try:
            try:
                from common.dynamodb import DynamoDBOperations
            except ImportError:
                from processing_job.common.dynamodb import DynamoDBOperations
            db_ops = DynamoDBOperations(table_name=DYNAMODB_TABLE_NAME)
        except Exception as exc:
            logger.error("Failed to initialize DynamoDB: {}", exc)

    total_written = 0
    total_errors = 0
    total_skipped = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    for shard_idx, shard in enumerate(shards):
        entry_id = shard.get("id", "")
        retrieval_query = shard.get("retrieval_query", "")
        video_prompt = shard.get("video_prompt") or shard.get("prompt", "")

        if not entry_id:
            logger.warning("Shard {}/{} missing 'id', skipping", shard_idx + 1, len(shards))
            total_skipped += 1
            continue

        if not retrieval_query:
            logger.warning(
                "Shard {}/{} (id={}) missing 'retrieval_query', skipping", shard_idx + 1, len(shards), entry_id
            )
            total_skipped += 1
            continue

        logger.info("--- Processing shard {}/{}: id={} ---", shard_idx + 1, len(shards), entry_id)
        logger.info("Retrieval query: {}", retrieval_query[:200])

        # Embed the retrieval_query via Bedrock Titan
        try:
            query_vector = embed_query(retrieval_query, bedrock_client, EMBEDDING_MODEL_ID)
            logger.info("Embedding successful: {} dimensions", len(query_vector))
        except Exception as exc:
            logger.error("Failed to embed retrieval_query for id={}: {}", entry_id, exc)
            total_errors += 1
            continue

        # Perform kNN search
        try:
            results = search_images(os_client, query_vector, AOSS_INDEX_NAME, QUERY_K)
        except Exception as exc:
            logger.error("OpenSearch query failed for id={}: {}", entry_id, exc)
            total_errors += 1
            continue

        if not results:
            logger.warning("No AOSS results for id={}, skipping shard", entry_id)
            total_skipped += 1
            continue

        # Try each result until we find one whose S3 object exists
        s3_client = boto3.client("s3")
        image_s3_uri = ""
        score = 0.0
        image_filename = ""
        for rank, candidate in enumerate(results):
            uri = candidate.get("image_s3_uri", "")
            if not uri:
                continue
            try:
                parts = uri.replace("s3://", "").split("/", 1)
                s3_client.head_object(Bucket=parts[0], Key=parts[1])
                image_s3_uri = uri
                score = candidate["score"]
                image_filename = os.path.basename(uri)
                if rank > 0:
                    logger.info("Fell back to rank {} for id={} (top {} had stale S3 refs)", rank + 1, entry_id, rank)
                break
            except Exception:
                logger.warning("S3 object not found for id={} rank {}: {} — trying next", entry_id, rank + 1, uri)

        if not image_s3_uri:
            logger.warning("All {} AOSS results for id={} have stale S3 refs, skipping", len(results), entry_id)
            total_skipped += 1
            continue

        logger.info("Top match for id={}: score={:.4f}, uri={}", entry_id, score, image_s3_uri)

        # Write VisualEntry shard (video_prompt as prompt, S3 URI as image)
        try:
            write_visual_entry_shard(entry_id, video_prompt, image_s3_uri, LOCAL_OUTPUT_DIR)
            total_written += 1
        except Exception as exc:
            logger.error("Failed to write shard for id={}: {}", entry_id, exc)
            total_errors += 1
            continue

        # Log per-entry metadata to DynamoDB
        if db_ops:
            try:
                db_ops.put_item(
                    id=entry_id,
                    step=STEP_NAME,
                    data={
                        COL.SOURCE_FILENAME: image_filename,
                        COL.SOURCE_S3_URI: image_s3_uri,
                        COL.OPENSEARCH_SCORE: str(score),
                        COL.RETRIEVAL_QUERY: retrieval_query,
                        COL.PIPELINE_EXECUTION_ID: EXECUTION_ID,
                        COL.TIMESTAMP: timestamp,
                    },
                )
            except Exception as exc:
                logger.error("Failed to log DynamoDB metadata for id={}: {}", entry_id, exc)

    logger.info("=== Shard Retrieval Summary ===")
    logger.info("Total shards: {}", len(shards))
    logger.info("Shards written: {}", total_written)
    logger.info("Skipped: {}", total_skipped)
    logger.info("Errors: {}", total_errors)

    if total_written == 0:
        logger.error("No VisualEntry shards were written")
        sys.exit(1)

    # List output directory contents
    if os.path.isdir(LOCAL_OUTPUT_DIR):
        output_files = os.listdir(LOCAL_OUTPUT_DIR)
        logger.info("Output directory contains {} files: {}", len(output_files), output_files[:20])


def _run_legacy_flow(region: str) -> None:
    """Legacy flow: read prompts, query AOSS, download images to output directory."""
    logger.info("=== Legacy retrieval flow ===")

    # Read prompts from input channel
    logger.info("Reading prompts from input directory: {}", SM_INPUT_DIR)
    if os.path.isdir(SM_INPUT_DIR):
        logger.info("Input directory contents: {}", os.listdir(SM_INPUT_DIR))
    else:
        logger.warning("Input directory does not exist: {}", SM_INPUT_DIR)

    prompts = read_prompts(SM_INPUT_DIR)
    if not prompts:
        logger.error("No prompts found in input directory: {}", SM_INPUT_DIR)
        sys.exit(1)

    logger.info("Found {} prompts to process", len(prompts))
    for i, p in enumerate(prompts):
        logger.info("  Prompt {}: {}", i + 1, p[:200])

    # Initialize clients
    logger.info("Initializing OpenSearch client...")
    os_client = build_opensearch_client(AOSS_ENDPOINT, region)
    s3_client = boto3.client("s3")
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    logger.info("Clients initialized")

    total_downloaded = 0
    total_errors = 0

    for prompt_idx, prompt in enumerate(prompts):
        logger.info("--- Processing prompt {}/{} ---", prompt_idx + 1, len(prompts))
        logger.info("Prompt text: {}", prompt[:200])

        # Embed the text prompt via Bedrock Titan
        logger.info("Embedding prompt via Bedrock model: {}", EMBEDDING_MODEL_ID)
        try:
            query_vector = embed_query(prompt, bedrock_client, EMBEDDING_MODEL_ID)
            logger.info("Embedding successful: {} dimensions", len(query_vector))
        except Exception as exc:
            logger.error("Failed to embed prompt {}: {}", prompt_idx + 1, exc)
            total_errors += 1
            continue

        # Perform kNN search
        logger.info("Searching OpenSearch index '{}' with k={}", AOSS_INDEX_NAME, QUERY_K)
        try:
            results = search_images(os_client, query_vector, AOSS_INDEX_NAME, QUERY_K)
        except Exception as exc:
            logger.error("OpenSearch query failed for prompt {}: {}", prompt_idx + 1, exc)
            total_errors += 1
            continue

        if not results:
            logger.warning("No results for prompt {}/{}", prompt_idx + 1, len(prompts))
            continue

        logger.info("Found {} matches for prompt {}", len(results), prompt_idx + 1)
        for r in results:
            logger.info("  doc_id={}, score={:.4f}, uri={}", r["doc_id"], r["score"], r["image_s3_uri"])

        # Download each matched image and write to output
        for result in results:
            doc_id = result["doc_id"]
            image_s3_uri = result["image_s3_uri"]
            score = result["score"]

            if not image_s3_uri:
                logger.warning("Empty image_s3_uri for doc_id={}, skipping", doc_id)
                continue

            try:
                bucket, key = parse_s3_uri(image_s3_uri)
                logger.info("Downloading s3://{}/{} ...", bucket, key)
                image_bytes = download_image(s3_client, bucket, key)
                ext = get_file_extension(key)
                output_filename = f"{doc_id}{ext}"
                output_path = os.path.join(LOCAL_OUTPUT_DIR, output_filename)

                with open(output_path, "wb") as f:
                    f.write(image_bytes)

                total_downloaded += 1
                logger.info(
                    "Saved {} ({:.2f} KB, score={:.4f}) -> {}",
                    doc_id,
                    len(image_bytes) / 1024,
                    score,
                    output_filename,
                )
            except Exception as exc:
                logger.error("Failed to download image for doc_id={}: {}", doc_id, exc)
                total_errors += 1

    logger.info("=== Retrieval Summary ===")
    logger.info("Total prompts: {}", len(prompts))
    logger.info("Images downloaded: {}", total_downloaded)
    logger.info("Errors: {}", total_errors)

    if total_downloaded == 0 and total_errors > 0:
        logger.error("No images were downloaded and errors occurred")
        sys.exit(1)

    # List output directory contents
    if os.path.isdir(LOCAL_OUTPUT_DIR):
        output_files = os.listdir(LOCAL_OUTPUT_DIR)
        logger.info("Output directory contains {} files: {}", len(output_files), output_files[:20])

    # Log outputs to DynamoDB
    logger.info("Logging retrieval results to DynamoDB...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "common.log_outputs",
                "--extensions",
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp",
                ".tiff",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to log outputs to DynamoDB: {}", exc)


if __name__ == "__main__":
    main()
    sys.exit(0)
