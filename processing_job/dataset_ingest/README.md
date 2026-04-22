> **Navigation:** [← Main README](../../README.md) | [Operations Guide — V-RAG Pipeline](../../docs/OPERATIONS.md#v-rag-pipeline) | [processing_job/README.md](../README.md)

# Dataset Ingest Setup Container

Standalone SageMaker Processing Job that seeds the retrieval ingestion pipeline with images and generates video prompts via a Strands Agent. Supports multiple dataset sources (Unsplash Lite, Open Images V7) — see the [Operations Guide — Dataset Sources](../../docs/OPERATIONS.md#dataset-sources) for switching between them.

## What It Does

1. Downloads the Unsplash Lite parquet dataset (~25k rows) from HuggingFace
2. Parses each row to extract image URLs and descriptions
3. Downloads images from the Unsplash CDN (concurrent thread pool)
4. Uploads images to the retrieval ingestion S3 bucket under `images/{photo_id}.jpg`
5. S3 upload triggers the existing S3 → SQS → Lambda → Bedrock embedding → OpenSearch Serverless ingestion pipeline
6. Generates video prompts from image descriptions using parallel Strands Agents (Bedrock Amazon Nova Lite, configurable worker count)
7. Writes `inputs_i2v.json` (VisualEntry array) to the SageMaker output channel
8. Verifies indexed document count in OpenSearch Serverless

## CLI Usage

```bash
# Normal mode: download, upload, generate prompts, verify
python3 main.py --run

# Cleanup mode: delete all images from S3 and documents from AOSS
python3 main.py --cleanup
```

## Environment Variables

| Variable | Description |
|---|---|
| `RETRIEVAL_BUCKET_NAME` | S3 bucket for image uploads (triggers ingestion pipeline) |
| `AOSS_ENDPOINT_SSM` | SSM parameter name for the AOSS endpoint (e.g. `/{prefix}/retrieval/aoss-endpoint`). Injected by CDK. |
| `AOSS_ENDPOINT` | Direct AOSS endpoint override. If set, skips SSM lookup. Not injected by CDK — for local testing only. |
| `AOSS_INDEX_NAME` | OpenSearch Serverless index name |
| `DATASET_URL` | HuggingFace dataset identifier (e.g. `1aurent/unsplash-lite`) |
| `DATASET_SCRIPT` | Dataset loader script filename (e.g. `unsplash.py`) |
| `NUM_PROMPTS` | Number of video prompts to generate |
| `TEST_IMAGE_COUNT` | Max images to process (for small-scale testing) |
| `PROMPT_WORKERS` | Number of parallel threads for prompt generation (default: `10`) |

## Parquet Schema (Unsplash Lite)

| Column | Type | Description |
|---|---|---|
| `id` | `str` | Unsplash photo ID |
| `image_url` | `str` | CDN URL (`https://images.unsplash.com/photo-...`) |
| `description` | `str` | User-provided description (may be `"nan"`) |
| `ai` | `dict` | Nested dict with `description` key (AI-generated) |
| `keywords` | `list[dict]` | List of dicts with `keyword` field |

## Description Fallback Chain

For each row, the best available description is resolved in order:

1. `ai["description"]` — if present, not `None`, not `"nan"`, not empty
2. `description` — if not `"nan"`, not empty
3. Comma-joined `keywords` — if the keywords list is non-empty
4. `None` — row is skipped with a warning

## How It Fits in the Pipeline

This container is a **standalone setup job**, not part of the SageMaker Pipeline DAG. It is:

- Created inside `RetrievalConstruct` (in `DataStack`)
- Triggered independently via its own Lambda (`{prefix}-dataset_ingest-trigger`)
- Designed for one-time data seeding or refresh of the retrieval index
- Idempotent: re-running overwrites existing S3 keys and upserts AOSS documents

The uploaded images flow through the existing retrieval ingestion pipeline:

```
S3 (images/) → SQS → Ingest Lambda (reserved_concurrency=40) → Bedrock Embedding → OpenSearch Serverless
```

No rate limiting is needed — the SQS queue and Lambda concurrency provide natural backpressure.
