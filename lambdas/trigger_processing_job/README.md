# trigger_processing_job

Starts a standalone SageMaker Processing Job outside the pipeline.

## How It Works

Reads the full job definition from `PROCESSING_JOB_DEFINITION` (JSON string) and creates a processing job with a timestamped name (`{JOB_NAME}-YYYY-MM-DD-HH-MM-SS`). Useful for running individual steps without triggering the full pipeline.

## Packaging

Container image built by CodeBuild and pushed to ECR. Uses `aws-lambda-powertools` for structured logging.

## Environment Variables

- `PROCESSING_JOB_DEFINITION` — JSON string containing the complete SageMaker Processing Job definition (resources, app spec, inputs, outputs, role, network config)
- `JOB_NAME` — Base name for the processing job (timestamp is appended)
