"""
RAG ingestion Lambda handler.

Triggered by SQS when images are uploaded to S3.
- Downloads image from S3
- Generates Bedrock Titan multimodal embedding
- Stores vector + metadata in OpenSearch Serverless
"""

from __future__ import annotations

import base64
import json
import os

import boto3
from aws_lambda_powertools import Logger
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

logger = Logger()

# Environment variables
RAG_BUCKET = os.environ["RAG_BUCKET_NAME"]
AOSS_ENDPOINT = os.environ["AOSS_ENDPOINT"]
AOSS_INDEX_NAME = os.environ["AOSS_INDEX_NAME"]
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-image-v1")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", "1024"))

# Clients
s3_client = boto3.client("s3")
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)


def get_oss_client() -> OpenSearch:
    """Create and return an authenticated OpenSearch Serverless client."""
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, "aoss")
    return OpenSearch(
        hosts=[{"host": AOSS_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def _is_nova_model(model_id: str) -> bool:
    """Check if the model ID is a Nova embedding model."""
    return "nova" in model_id.lower()


def ensure_index(oss_client: OpenSearch) -> None:
    """Create the kNN index if it doesn't exist."""
    if not oss_client.indices.exists(AOSS_INDEX_NAME):
        oss_client.indices.create(
            index=AOSS_INDEX_NAME,
            body={
                "settings": {"index.knn": "true"},
                "mappings": {
                    "properties": {
                        "image_vector": {"type": "knn_vector", "dimension": EMBEDDING_DIMENSION},
                        "description": {"type": "text"},
                        "image_s3_uri": {"type": "text"},
                        "image_base64_s3_uri": {"type": "text"},
                    }
                },
            },
        )
        logger.info("Created OpenSearch index", index=AOSS_INDEX_NAME, dimension=EMBEDDING_DIMENSION)


def get_image_embedding(image_base64: str) -> list[float]:
    """Invoke Bedrock embedding model for an image.

    Supports both Titan multimodal and Nova multimodal embedding models.

    Args:
        image_base64: Base64-encoded image string.

    Returns:
        List of embedding float values.
    """
    if _is_nova_model(EMBEDDING_MODEL_ID):
        request_body = {
            "schemaVersion": "nova-multimodal-embed-v1",
            "taskType": "SINGLE_EMBEDDING",
            "singleEmbeddingParams": {
                "embeddingPurpose": "GENERIC_INDEX",
                "embeddingDimension": EMBEDDING_DIMENSION,
                "image": {
                    "format": "jpeg",
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
    """
    Main handler for ingestion Lambda.

    Processes SQS-wrapped S3 ObjectCreated events.
    """
    oss_client = get_oss_client()
    ensure_index(oss_client)

    processed = 0
    failed = 0

    for record in event.get("Records", []):
        try:
            # Parse SQS body (contains S3 event records)
            body = json.loads(record["body"])
            s3_records = body.get("Records", [body])

            for s3_record in s3_records:
                bucket = s3_record["s3"]["bucket"]["name"]
                key = s3_record["s3"]["object"]["key"]

                logger.info("Processing image", bucket=bucket, key=key)

                # Download image from S3
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                image_bytes = obj["Body"].read()

                # Base64 encode
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                # Store base64 version in S3 under base64/ prefix
                base64_key = f"base64/{key}.txt"
                s3_client.put_object(
                    Bucket=RAG_BUCKET,
                    Key=base64_key,
                    Body=image_base64.encode("utf-8"),
                )

                # Get embedding
                embedding = get_image_embedding(image_base64)

                # Index into OpenSearch
                oss_client.index(
                    index=AOSS_INDEX_NAME,
                    body={
                        "image_vector": embedding,
                        "description": key,
                        "image_s3_uri": f"s3://{bucket}/{key}",
                        "image_base64_s3_uri": f"s3://{RAG_BUCKET}/{base64_key}",
                    },
                )

                logger.info(
                    "Indexed image successfully",
                    key=key,
                    index=AOSS_INDEX_NAME,
                )
                processed += 1

        except Exception as e:
            logger.exception("Error processing record", error=str(e))
            failed += 1
            # Return False to keep the message visible for retry
            if "messageId" in record:
                record["itemId"] = record["messageId"]

    logger.info("Ingestion complete", processed=processed, failed=failed)
    return {
        "statusCode": 200 if failed == 0 else 206,
        "processed": processed,
        "failed": failed,
    }
