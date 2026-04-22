"""
Retrieval ingestion Lambda handler.

Triggered by SQS when images are uploaded to S3 under the images/ prefix.
For each image:
- Downloads the image from S3
- Generates a base64-encoded copy and stores it under the base64/ prefix
- Invokes Bedrock Titan multimodal embedding model for a 1024-dim vector
- Indexes the vector and metadata into OpenSearch Serverless

Reports individual SQS batch item failures so that only failed messages
are retried (partial batch failure protocol).

Environment variables:
    RETRIEVAL_BUCKET_NAME: S3 bucket for ingestion (images/ and base64/ prefixes)
    AOSS_ENDPOINT: OpenSearch Serverless collection endpoint
    AOSS_INDEX_NAME: OpenSearch index name for image vectors
    EMBEDDING_MODEL_ID: Bedrock model ID (default: amazon.titan-embed-image-v1)
"""

import base64
import json
import os

import boto3
from aws_lambda_powertools import Logger
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

logger = Logger()

# Environment variables
RETRIEVAL_BUCKET: str = os.environ["RETRIEVAL_BUCKET_NAME"]
AOSS_ENDPOINT: str = os.environ["AOSS_ENDPOINT"]
AOSS_INDEX_NAME: str = os.environ["AOSS_INDEX_NAME"]
REGION: str = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
EMBEDDING_MODEL_ID: str = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-image-v1")
EMBEDDING_DIMENSION: int = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))

# AWS clients
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)


def get_oss_client() -> OpenSearch:
    """Create and return an authenticated OpenSearch Serverless client.

    Uses IAM credentials with AWS SigV4 signing for the ``aoss`` service.

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


def _is_nova_model(model_id: str) -> bool:
    """Check if the model ID is a Nova embedding model."""
    return "nova" in model_id.lower()


def ensure_index(oss_client: OpenSearch) -> None:
    """Create the kNN index if it does not already exist.

    The index is configured with:
    - ``index.knn`` enabled
    - ``image_vector``: knn_vector field with configurable dimension
    - ``description``: text field for the image key / description
    - ``image_s3_uri``: text field for the original image S3 URI
    - ``image_base64_s3_uri``: text field for the base64 copy S3 URI

    Args:
        oss_client: Authenticated OpenSearch client.
    """
    if not oss_client.indices.exists(index=AOSS_INDEX_NAME):
        oss_client.indices.create(
            index=AOSS_INDEX_NAME,
            body={
                "settings": {"index.knn": "true"},
                "mappings": {
                    "properties": {
                        "image_vector": {
                            "type": "knn_vector",
                            "dimension": EMBEDDING_DIMENSION,
                        },
                        "description": {"type": "text"},
                        "image_s3_uri": {"type": "text"},
                        "image_base64_s3_uri": {"type": "text"},
                    }
                },
            },
        )
        logger.info("Created OpenSearch index", index=AOSS_INDEX_NAME, dimension=EMBEDDING_DIMENSION)


def _detect_image_format(image_base64: str) -> str:
    """Detect image format from base64-encoded bytes. Returns 'jpeg', 'png', 'gif', or 'webp'."""
    header = base64.b64decode(image_base64[:32])
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if header[:3] == b"GIF":
        return "gif"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    return "jpeg"


def get_image_embedding(image_base64: str) -> list[float]:
    """Invoke Bedrock embedding model for an image.

    Supports both Titan multimodal and Nova multimodal embedding models.
    Titan uses a flat request format; Nova uses the structured schema.

    Args:
        image_base64: Base64-encoded image string.

    Returns:
        List of float values representing the image embedding.
    """
    if _is_nova_model(EMBEDDING_MODEL_ID):
        img_format = _detect_image_format(image_base64)
        request_body = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": EMBEDDING_DIMENSION,
                "image": {
                    "format": img_format,
                    "source": {"bytes": image_base64},
                },
            },
        }
        response = bedrock_client.invoke_model(
            body=json.dumps(request_body),
            modelId=EMBEDDING_MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        return list(result["embeddings"][0]["embedding"])
    else:
        # Titan format
        response = bedrock_client.invoke_model(
            body=json.dumps(
                {
                    "inputImage": image_base64,
                    "embeddingConfig": {"outputEmbeddingLength": EMBEDDING_DIMENSION},
                }
            ),
            modelId=EMBEDDING_MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        return list(result["embedding"])


def lambda_handler(event: dict, context: object) -> dict:
    """Process SQS-wrapped S3 ObjectCreated events for image ingestion.

    For each SQS record the handler:
    1. Parses the S3 event from the SQS message body.
    2. Downloads the image from S3.
    3. Base64-encodes the image and stores it under the ``base64/`` prefix.
    4. Calls Bedrock Titan to generate a 1024-dim embedding.
    5. Indexes the document into OpenSearch with vector, S3 URIs, and description.

    Failed records are reported individually via the SQS partial batch failure
    protocol so that only those messages become visible again for retry.

    Args:
        event: SQS event containing one or more records.
        context: Lambda context object (unused).

    Returns:
        Dict with ``batchItemFailures`` list. Each entry contains the
        ``itemIdentifier`` (SQS messageId) of a failed record.
    """
    oss_client = get_oss_client()
    ensure_index(oss_client)

    batch_item_failures: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id: str = record.get("messageId", "")
        try:
            body = json.loads(record["body"])
            s3_records: list[dict] = body.get("Records", [body])

            for s3_record in s3_records:
                bucket: str = s3_record["s3"]["bucket"]["name"]
                key: str = s3_record["s3"]["object"]["key"]

                logger.info("Processing image", bucket=bucket, key=key)

                # Download image from S3
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                image_bytes: bytes = obj["Body"].read()

                # Base64 encode
                image_base64: str = base64.b64encode(image_bytes).decode("utf-8")

                # Store base64 version under base64/ prefix
                base64_key: str = f"base64/{key}.txt"
                s3_client.put_object(
                    Bucket=RETRIEVAL_BUCKET,
                    Key=base64_key,
                    Body=image_base64.encode("utf-8"),
                )

                # Generate embedding
                embedding: list[float] = get_image_embedding(image_base64)

                # Index document into OpenSearch
                oss_client.index(
                    index=AOSS_INDEX_NAME,
                    body={
                        "image_vector": embedding,
                        "description": key,
                        "image_s3_uri": f"s3://{bucket}/{key}",
                        "image_base64_s3_uri": f"s3://{RETRIEVAL_BUCKET}/{base64_key}",
                    },
                )

                logger.info(
                    "Indexed image successfully",
                    key=key,
                    index=AOSS_INDEX_NAME,
                )

        except Exception:
            logger.exception("Error processing record", message_id=message_id)
            batch_item_failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": batch_item_failures}
