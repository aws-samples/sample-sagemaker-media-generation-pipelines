> **Navigation:** [← Main README](../README.md) | [Extending Guide](../docs/EXTENDING.md)

# lambdas/

Lambda functions that orchestrate the pipeline lifecycle. Each subdirectory is a self-contained Lambda with its own `index.py`, `requirements.txt`, and (for container-image Lambdas) a `Dockerfile` + `buildspec.yml`.

## Functions

- `trigger_pipeline/` — Starts a SageMaker Pipeline execution. Invoked manually or by automation. Reads `PIPELINE_NAME` from env and calls `StartPipelineExecution`.
- `trigger_processing_job/` — Starts a standalone SageMaker Processing Job (outside the pipeline). Reads the full job definition from env and creates a job with a timestamped name.
- `codebuild_trigger/` — CloudFormation custom resource handler that fires off CodeBuild projects (fire-and-forget) during CDK deploys. Used to build Docker images on stack create/update.
- `submit_a2i_review/` — Lists generated assets from an output S3 bucket and creates one A2I human loop per asset via the configured flow definition. Labels each asset with the model name and generation index (e.g. `wan22 #0`) when a model from the shared registry (`schema/models.yaml`) is detected in the filename, falling back to `Generation N` for assets without a recognized model. Fetches the display prompt from DynamoDB with a fallback chain (`prompt` → `tags`) and passes `lyrics` as a separate field to the A2I template, supporting different modalities (video prompts, image tags, audio lyrics). Receives `TASK_TITLE` and `TASK_DESCRIPTION` from the A2I config as environment variables for customizing the review UI per flow. Invoked as a Lambda step in the SageMaker Pipeline after generation steps.
- `process_a2i_results/` — Processes completed A2I human review results. Triggered via SNS when a human loop completes, writes review decisions to DynamoDB.
- `retrieval_ingest/` — Processes SQS messages from S3 image uploads. Reads the image from S3, generates an embedding via Bedrock (supports both Titan multimodal and Nova multimodal embedding models, auto-detected from the configured `EMBEDDING_MODEL_ID`), and indexes the document into the AOSS vector index. For Nova models, automatically detects image format (JPEG, PNG, GIF, WebP) from file headers. Container-image Lambda via CodeBuild/ECR.
- `retrieval_load_test/` — Load testing and verification utility for the retrieval pipeline. Supports document count checks, image upload + poll for indexing, and index purge. Container-image Lambda.
- `retrieval_query/` — Performs kNN similarity search against the AOSS vector index. Supports both Titan and Nova embedding models for query embedding. Returns top-k nearest neighbours. Container-image Lambda.

## Packaging

`trigger_pipeline` and `trigger_processing_job` are packaged as container images via CodeBuild (ECR). `codebuild_trigger` is a zip-based Lambda bundled by CDK directly. The three `retrieval_*` Lambdas are also container-image Lambdas built via CodeBuild/ECR.

Zip-deployed Lambdas automatically include the [`schema/`](../schema/README.md) package in their bundle, providing DynamoDB column name constants and the known model registry.
