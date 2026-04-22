"""
Lambda that processes completed A2I human review results.

Triggered by SNS notification when a human loop completes.
Reads the review output from S3, finds which asset was selected,
and updates the EXISTING DynamoDB records with selected=True/False.

Environment variables:
    DYNAMODB_TABLE_NAME: DynamoDB table for tracking reviews
"""

import json
import os
from datetime import datetime, timezone

import boto3
from aws_lambda_powertools import Logger
from boto3.dynamodb.conditions import Key

from schema.columns import COL

logger = Logger()

s3 = boto3.client("s3")
a2i_runtime = boto3.client("sagemaker-a2i-runtime")
dynamodb = boto3.resource("dynamodb")

TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]
REVIEW_LOOP_GSI = "gsi-review-loop-name"


def _parse_a2i_output(output_s3_uri: str) -> dict:
    """Download and parse the A2I JSON output from S3."""
    parts = output_s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    response = s3.get_object(Bucket=bucket, Key=key)
    result: dict = json.loads(response["Body"].read().decode("utf-8"))
    return result


def _extract_selected(a2i_output: dict) -> str | None:
    """Extract the selected asset filename from A2I output.

    The Crowd HTML radio button produces answerContent.selected as a dict
    of {filename: true/false} for each option, e.g.:
        {"test_001_g0.png": true, "test_001_g1.png": false}

    With multiple workers, majority vote wins.
    """
    answers = a2i_output.get("humanAnswers", [])
    votes: dict[str, int] = {}
    for answer in answers:
        content = answer.get("answerContent", {})
        selected = content.get("selected", "")
        if isinstance(selected, dict):
            # Radio button dict: find the key with value True
            for filename, is_selected in selected.items():
                if is_selected:
                    votes[filename] = votes.get(filename, 0) + 1
        elif isinstance(selected, str) and selected:
            # Simple string value (fallback)
            votes[selected] = votes.get(selected, 0) + 1
    if not votes:
        return None
    return max(votes, key=lambda k: votes[k])


def _find_records_by_loop_name(table, loop_name: str) -> list[dict]:
    """Find all DynamoDB records that have this review_loop_name.

    Uses the gsi-review-loop-name GSI for an efficient query instead of
    a full table scan. Paginates to handle large result sets.
    """
    items: list[dict] = []
    query_kwargs: dict = {
        "IndexName": REVIEW_LOOP_GSI,
        "KeyConditionExpression": Key(COL.REVIEW_LOOP_NAME).eq(loop_name),
    }
    while True:
        response = table.query(**query_kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        query_kwargs["ExclusiveStartKey"] = last_key
    return items


def lambda_handler(event: dict, context: object) -> dict:
    """Process A2I review completion — mark the selected asset.

    Supports two trigger sources:
    - EventBridge: "SageMaker A2I HumanLoop Status Change" events (recommended)
    - SNS: Legacy workteam notification subscription
    """
    table = dynamodb.Table(TABLE_NAME)
    now = datetime.now(timezone.utc).isoformat()
    processed = []

    # Normalise incoming events into a list of {humanLoopName, humanLoopOutput}
    loop_events = []

    if "detail-type" in event and event.get("detail-type") == "SageMaker A2I HumanLoop Status Change":
        # EventBridge event — single human loop completion
        detail = event.get("detail", {})
        loop_events.append(
            {
                "humanLoopName": detail.get("humanLoopName", ""),
                "humanLoopStatus": detail.get("humanLoopStatus", ""),
                "outputS3Uri": detail.get("humanLoopOutput", {}).get("outputS3Uri", ""),
            }
        )
    elif "Records" in event:
        # SNS event — parse each record
        for record in event.get("Records", []):
            sns_message = json.loads(record["Sns"]["Message"])
            loop_events.append(
                {
                    "humanLoopName": sns_message.get("humanLoopName", ""),
                    "humanLoopStatus": "",
                    "outputS3Uri": "",
                }
            )
    else:
        logger.error("Unrecognised event format: %s", json.dumps(event)[:500])
        return {"processed": [], "count": 0}

    for loop_evt in loop_events:
        loop_name = loop_evt["humanLoopName"]
        if not loop_name:
            continue

        logger.info("Processing completed review", loop_name=loop_name)

        db_records = _find_records_by_loop_name(table, loop_name)
        if not db_records:
            logger.error("No DynamoDB records found for loop_name=%s", loop_name)
            continue

        # Use output URI from event if available, otherwise describe the loop
        output_s3_uri = loop_evt.get("outputS3Uri", "")
        status = loop_evt.get("humanLoopStatus", "")

        if not output_s3_uri or not status:
            try:
                loop_info = a2i_runtime.describe_human_loop(HumanLoopName=loop_name)
                status = status or loop_info.get("HumanLoopStatus", "Unknown")
                output_s3_uri = output_s3_uri or loop_info.get("HumanLoopOutput", {}).get("OutputS3Uri", "")
            except Exception:
                logger.exception("Failed to describe human loop", loop_name=loop_name)
                continue

        selected_asset = None
        if output_s3_uri:
            try:
                a2i_output = _parse_a2i_output(output_s3_uri)
                selected_asset = _extract_selected(a2i_output)
            except Exception:
                logger.exception("Failed to parse A2I output", loop_name=loop_name, output_s3_uri=output_s3_uri)

        logger.info("Selected asset: %s", selected_asset)

        # Update each record: selected=True on winner, False on rest.
        # Write selected_flag="true" only on the winner so the sparse
        # gsi-selected-flag GSI contains only selected items.
        for rec in db_records:
            pk, sk = rec[COL.ID], rec[COL.STEP]
            is_selected = rec.get(COL.FILENAME) == selected_asset

            if is_selected:
                table.update_item(
                    Key={COL.ID: pk, COL.STEP: sk},
                    UpdateExpression=(
                        f"SET {COL.REVIEW_STATUS} = :status, {COL.REVIEW_COMPLETED_AT} = :now, "
                        f"{COL.SELECTED} = :sel, {COL.SELECTED_FLAG} = :flag"
                    ),
                    ExpressionAttributeValues={
                        ":status": status.lower(),
                        ":now": now,
                        ":sel": True,
                        ":flag": "true",
                    },
                )
            else:
                # Remove selected_flag so this item drops out of the sparse GSI
                table.update_item(
                    Key={COL.ID: pk, COL.STEP: sk},
                    UpdateExpression=(
                        f"SET {COL.REVIEW_STATUS} = :status, {COL.REVIEW_COMPLETED_AT} = :now, "
                        f"{COL.SELECTED} = :sel "
                        f"REMOVE {COL.SELECTED_FLAG}"
                    ),
                    ExpressionAttributeValues={
                        ":status": status.lower(),
                        ":now": now,
                        ":sel": False,
                    },
                )
            logger.info("Updated id=%s step=%s selected=%s", pk, sk, is_selected)

        processed.append(loop_name)

    return {"processed": processed, "count": len(processed)}
