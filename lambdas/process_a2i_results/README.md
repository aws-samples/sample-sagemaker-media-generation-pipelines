# process_a2i_results

Processes completed A2I human review results. Triggered when a human loop completes, parses the reviewer's selection, and updates DynamoDB records with the outcome.

## How It Works

1. Receives events from either EventBridge ("SageMaker A2I HumanLoop Status Change") or SNS (legacy workteam notification)
2. Looks up all DynamoDB records matching the `review_loop_name` via the `gsi-review-loop-name` GSI
3. Downloads and parses the A2I output JSON from S3
4. Extracts the selected asset using majority vote when multiple reviewers are involved
5. Updates each record: `selected=True` on the winner, `selected=False` on the rest
6. Sets `selected_flag="true"` only on the winner so the sparse `gsi-selected-flag` GSI contains only selected items

## Packaging

Container image built by CodeBuild and pushed to ECR. Uses `aws-lambda-powertools` for structured logging.

## Environment Variables

| Variable | Description |
|---|---|
| `DYNAMODB_TABLE_NAME` | DynamoDB table for tracking reviews |
