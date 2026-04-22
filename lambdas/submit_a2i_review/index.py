"""
Lambda that submits generated assets to A2I for human review.

Groups assets by input_id and starts ONE human loop per input_id,
showing all assets side by side so the reviewer picks the best one.
Updates the EXISTING DynamoDB records (created by log_outputs.py)
with A2I metadata — does NOT create new records.

Environment variables:
    FLOW_DEFINITION_ARN: ARN of the A2I flow definition
    SOURCE_BUCKET: S3 bucket containing generated assets
    DYNAMODB_TABLE_NAME: DynamoDB table for tracking reviews
    MEDIA_TYPE: Type of media (image, video, audio)
    UPSTREAM_STEP: Name of the upstream generation step (e.g. t2v, i2v)
"""

import json
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import boto3
from aws_lambda_powertools import Logger

from schema.columns import COL
from schema.models import KNOWN_MODELS

logger = Logger()

s3 = boto3.client("s3")
a2i_runtime = boto3.client("sagemaker-a2i-runtime")
dynamodb = boto3.resource("dynamodb")

FLOW_DEFINITION_ARN = os.environ["FLOW_DEFINITION_ARN"]
SOURCE_BUCKET = os.environ["SOURCE_BUCKET"]
TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]
MEDIA_TYPE = os.environ.get("MEDIA_TYPE", "video")
UPSTREAM_STEP = os.environ.get("UPSTREAM_STEP", "")
TASK_TITLE = os.environ.get("TASK_TITLE", "")
TASK_DESCRIPTION = os.environ.get("TASK_DESCRIPTION", "")

MEDIA_EXTENSIONS = {
    "video": (".mp4", ".webm", ".mkv"),
    "image": (".png", ".jpg", ".jpeg", ".webp"),
    "audio": (".mp3", ".wav", ".flac", ".ogg"),
}


def _parse_asset_name(asset_name: str) -> dict:
    """Extract input_id, model, and generation_index from an asset filename.

    Filenames follow: {input_id}_{model}[_g{N}]_{frame}_.ext
    e.g. honey-pancakes_ltx23_00001_.mp4
         honey-pancakes_ltx23_g1_00001_.mp4
    """
    name = asset_name.rsplit(".", 1)[0] if "." in asset_name else asset_name
    for model in KNOWN_MODELS:
        pattern = rf"^(.+?)_{re.escape(model)}(?:_g(\d+))?_"
        m = re.match(pattern, name)
        if m:
            return {
                "input_id": m.group(1),
                "model": model,
                "generation_index": int(m.group(2)) if m.group(2) else 0,
            }
    return {"input_id": name, "model": "", "generation_index": 0}


def _build_sort_key(step_name: str, model: str, gen_idx: int) -> str:
    """Build the same sort key that log_outputs.py uses.

    Pattern: {step}#{model}#g{N} (always includes generation index).
    """
    if model and gen_idx is not None:
        return f"{step_name}#{model}#g{gen_idx}"
    elif model:
        return f"{step_name}#{model}"
    return step_name


def _list_assets(bucket: str, prefix: str) -> list[str]:
    """List S3 keys matching the media type extensions under prefix."""
    extensions = MEDIA_EXTENSIONS.get(MEDIA_TYPE, MEDIA_EXTENSIONS["video"])
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith(extensions):
                keys.append(obj["Key"])
    return keys


def lambda_handler(event: dict, context: object) -> dict:
    """Submit assets for A2I human review, grouped by input_id."""
    execution_id = event.get("execution_id", "")
    prefix = event.get("prefix", "")
    if execution_id and not prefix:
        prefix = f"{execution_id}/"

    logger.info(
        "Listing assets", bucket=SOURCE_BUCKET, prefix=prefix, media_type=MEDIA_TYPE, upstream_step=UPSTREAM_STEP
    )

    asset_keys = _list_assets(SOURCE_BUCKET, prefix)
    logger.info("Found %d assets to review", len(asset_keys))
    if not asset_keys:
        return {"human_loops": [], "count": 0}

    # Group assets by input_id
    groups: dict[str, list[dict]] = defaultdict(list)
    for key in asset_keys:
        asset_name = key.rsplit("/", 1)[-1]
        parsed = _parse_asset_name(asset_name)
        input_id = parsed["input_id"] or str(uuid.uuid4())
        groups[input_id].append(
            {
                "s3_key": key,
                "asset_name": asset_name,
                "model": parsed["model"],
                "generation_index": parsed["generation_index"],
            }
        )

    table = dynamodb.Table(TABLE_NAME)
    now = datetime.now(timezone.utc).isoformat()
    human_loops = []

    for input_id, assets in groups.items():
        loop_name = f"review-{uuid.uuid4().hex[:12]}"

        # Build the asset list for the template — pass S3 URIs so the
        # Crowd HTML template can use the | grant_read_access Liquid filter
        # to generate short-lived access tokens for the worker UI.
        template_assets = []
        for asset in assets:
            s3_uri = f"s3://{SOURCE_BUCKET}/{asset['s3_key']}"
            gen = asset["generation_index"]
            model = asset.get("model", "")
            label = f"{model} #{gen}" if model else f"Generation {gen}"
            template_assets.append(
                {
                    "s3_uri": s3_uri,
                    "name": asset["asset_name"],
                    "label": label,
                }
            )

        # Fetch the full prompt from DynamoDB (stored by log_outputs.py)
        full_prompt = ""
        first_asset = assets[0]
        sort_key = _build_sort_key(
            UPSTREAM_STEP,
            first_asset["model"],
            first_asset["generation_index"],
        )
        try:
            resp = table.get_item(Key={"id": input_id, "step": sort_key})
            item = resp.get("Item", {})
            full_prompt = item.get(COL.PROMPT, "") or item.get(COL.TAGS, "")
            full_lyrics = item.get(COL.LYRICS, "")
        except Exception as exc:
            logger.warning("Could not fetch prompt for id=%s step=%s: %s", input_id, sort_key, exc)

        prompt_display = full_prompt or input_id

        input_content = {
            "taskTitle": TASK_TITLE or f"Select best {MEDIA_TYPE}: {input_id}",
            "taskDescription": TASK_DESCRIPTION
            or f"Compare these {len(assets)} AI-generated {MEDIA_TYPE}(s) and select the best one.",
            "prompt": prompt_display,
            "lyrics": full_lyrics,
            "assets": template_assets,
        }

        logger.info("Starting human loop", loop_name=loop_name, input_id=input_id, asset_count=len(assets))

        a2i_runtime.start_human_loop(
            HumanLoopName=loop_name,
            FlowDefinitionArn=FLOW_DEFINITION_ARN,
            HumanLoopInput={"InputContent": json.dumps(input_content)},
        )

        # Update existing DynamoDB records with A2I metadata
        for asset in assets:
            sort_key = _build_sort_key(
                UPSTREAM_STEP,
                asset["model"],
                asset["generation_index"],
            )
            logger.info("Updating existing record id=%s step=%s with A2I info", input_id, sort_key)
            table.update_item(
                Key={COL.ID: input_id, COL.STEP: sort_key},
                UpdateExpression=(
                    f"SET {COL.REVIEW_LOOP_NAME} = :hln, "
                    f"{COL.REVIEW_STATUS} = :st, "
                    f"{COL.SELECTED} = :sel, "
                    f"{COL.REVIEW_SUBMITTED_AT} = :now, "
                    f"{COL.REVIEW_FLOW_ARN} = :fda"
                ),
                ExpressionAttributeValues={
                    ":hln": loop_name,
                    ":st": "pending",
                    ":sel": False,
                    ":now": now,
                    ":fda": FLOW_DEFINITION_ARN,
                },
            )

        human_loops.append(loop_name)

    logger.info("Submitted %d human loops for %d input_ids", len(human_loops), len(groups))
    return {"human_loops": human_loops, "count": len(human_loops)}
