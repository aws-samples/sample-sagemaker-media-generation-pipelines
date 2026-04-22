# retrieval_query

Performs kNN similarity search against the OpenSearch Serverless vector index. Embeds a query image via Bedrock and returns the top-k nearest neighbours.

## How It Works

1. Receives a query image (base64-encoded) or text
2. Generates an embedding vector via the configured Bedrock model
3. Runs a kNN search against the AOSS index
4. Returns the top-k matching documents with S3 URIs and similarity scores

Supports both Titan multimodal and Nova multimodal embedding models, auto-detected from the configured `EMBEDDING_MODEL_ID`.

## Packaging

Container image built by CodeBuild and pushed to ECR. Uses `aws-lambda-powertools` for structured logging and `opensearch-py` for AOSS access.

## Environment Variables

| Variable | Description |
|---|---|
| `RAG_BUCKET_NAME` | S3 bucket containing images |
| `AOSS_ENDPOINT` | OpenSearch Serverless collection endpoint |
| `AOSS_INDEX_NAME` | OpenSearch index name for kNN queries |
| `EMBEDDING_MODEL_ID` | Bedrock model ID (default: `amazon.titan-embed-image-v1`) |
| `EMBEDDING_DIMENSION` | Embedding vector dimension (default: `1024`) |
