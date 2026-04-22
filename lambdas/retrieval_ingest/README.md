# retrieval_ingest

Processes SQS messages triggered by S3 image uploads. Generates vector embeddings via Bedrock and indexes them into OpenSearch Serverless for kNN similarity search.

## How It Works

1. Receives SQS-wrapped S3 `ObjectCreated` events
2. Downloads the image from S3
3. Base64-encodes the image and stores it under the `base64/` prefix in S3
4. Invokes the configured Bedrock embedding model to generate a vector
5. Indexes the document into OpenSearch Serverless with the vector, S3 URIs, and description

Supports both Titan multimodal (`amazon.titan-embed-image-v1`) and Nova multimodal (`amazon.nova-2-multimodal-embeddings-v1:0`) embedding models, auto-detected from the configured `EMBEDDING_MODEL_ID`. For Nova models, automatically detects image format (JPEG, PNG, GIF, WebP) from file headers.

Uses the SQS partial batch failure protocol — only failed messages are retried.

## Packaging

Container image built by CodeBuild and pushed to ECR. Uses `aws-lambda-powertools` for structured logging and `opensearch-py` for AOSS access.

## Environment Variables

| Variable | Description |
|---|---|
| `RETRIEVAL_BUCKET_NAME` | S3 bucket for ingestion (`images/` and `base64/` prefixes) |
| `AOSS_ENDPOINT` | OpenSearch Serverless collection endpoint |
| `AOSS_INDEX_NAME` | OpenSearch index name for image vectors |
| `EMBEDDING_MODEL_ID` | Bedrock model ID (default: `amazon.titan-embed-image-v1`) |
| `EMBEDDING_DIMENSION` | Embedding vector dimension (default: `1024`) |
