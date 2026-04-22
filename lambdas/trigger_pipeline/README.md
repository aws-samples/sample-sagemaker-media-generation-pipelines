# trigger_pipeline

Starts a SageMaker Pipeline execution.

## How It Works

Reads `PIPELINE_NAME` from environment variables and calls `StartPipelineExecution`. Returns the execution ARN. Can be invoked manually via the AWS console/CLI or by automation.

## Packaging

Container image built by CodeBuild and pushed to ECR. Uses `aws-lambda-powertools` for structured logging.

## Environment Variables

- `PIPELINE_NAME` — Name of the SageMaker Pipeline to execute
