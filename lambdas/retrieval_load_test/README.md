# retrieval_load_test

Load testing and verification utility for the retrieval ingestion pipeline. Supports document count checks, image upload with poll-for-indexing, and index purge. Supports both Titan multimodal and Nova multimodal embedding models.

## How It Works

Accepts a JSON payload with an optional `action` field. Three modes:

1. **Load test** (default) — Generates N random images, uploads them to S3 (triggering the ingestion pipeline), polls AOSS until all documents are indexed or timeout, then optionally cleans up test artifacts. Returns throughput metrics and pass/fail status.
2. **Purge** (`{"action": "purge"}`) — Deletes all documents from the AOSS index. Useful for clearing stale data before re-ingestion.
3. **Count check** — When invoked with a small `num_images` and short `timeout`, acts as a quick health check to verify the ingestion pipeline is working end-to-end.

## Packaging

Container image built by CodeBuild and pushed to ECR. Uses `aws-lambda-powertools` for structured logging and `opensearch-py` for AOSS access.

## Environment Variables

| Variable | Description |
|---|---|
| `RETRIEVAL_BUCKET_NAME` | S3 bucket for image uploads (triggers ingestion pipeline) |
| `AOSS_ENDPOINT` | OpenSearch Serverless collection endpoint |
| `AOSS_INDEX_NAME` | OpenSearch index name for image vectors |

## Payload Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `action` | string | — | Set to `"purge"` to delete all documents. Omit for load test mode. |
| `num_images` | int | `1000` | Number of random images to generate and upload |
| `poll_interval` | int | `5` | Seconds between monitoring polls |
| `timeout` | int | `600` | Max seconds to wait for ingestion (must be <= 870) |
| `image_size` | int | `64` | Width/height in pixels for generated PNGs |
| `cleanup` | bool | `true` | Delete test artifacts (S3 + AOSS) after completion |
