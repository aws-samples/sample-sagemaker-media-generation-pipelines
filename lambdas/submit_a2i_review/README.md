# submit_a2i_review

Submits generated assets to Amazon A2I for human review. Groups assets by `input_id` and starts one human loop per input, showing all assets side-by-side so the reviewer picks the best one.

## How It Works

1. Lists generated assets from the output S3 bucket matching the configured `MEDIA_TYPE` extensions
2. Parses each filename to extract `input_id`, `model`, and `generation_index` using the shared model registry (`schema/models.yaml`)
3. Groups assets by `input_id`
4. Fetches the display prompt from DynamoDB with a fallback chain (`prompt` → `tags`), plus `lyrics` as a separate field for audio modalities
5. Starts one A2I human loop per group via the configured flow definition
6. Updates existing DynamoDB records (created by `log_outputs.py`) with A2I metadata (`review_loop_name`, `review_status: pending`)

Asset labels use the model name and generation index (e.g. `wan22 #0`) when a model from the shared registry is detected in the filename, falling back to `Generation N` otherwise.

## Packaging

Container image built by CodeBuild and pushed to ECR. Uses `aws-lambda-powertools` for structured logging.

## Environment Variables

| Variable | Description |
|---|---|
| `FLOW_DEFINITION_ARN` | ARN of the A2I flow definition |
| `SOURCE_BUCKET` | S3 bucket containing generated assets |
| `DYNAMODB_TABLE_NAME` | DynamoDB table for tracking reviews |
| `MEDIA_TYPE` | Type of media (`image`, `video`, `audio`) |
| `UPSTREAM_STEP` | Name of the upstream generation step (e.g. `t2v`, `i2v`) |
| `TASK_TITLE` | Custom title for the review task UI |
| `TASK_DESCRIPTION` | Custom description for the review task UI |
