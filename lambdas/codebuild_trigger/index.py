"""Custom resource handler that starts CodeBuild projects (fire-and-forget).

This Lambda is invoked by CDK's cr.Provider framework, NOT directly by
CloudFormation. It must return a dict with PhysicalResourceId (not call
cfnresponse or send HTTP responses to ResponseURL).
"""

import os

import boto3
from aws_lambda_powertools import Logger

logger = Logger()
codebuild = boto3.client("codebuild")


def handler(event: dict, context: object) -> dict:
    """Start all CodeBuild projects without waiting for completion.

    Args:
        event: cr.Provider on_event event with RequestType and ResourceProperties.
        context: AWS Lambda context object.

    Returns:
        Dict with PhysicalResourceId for cr.Provider framework.
    """
    request_type: str = event.get("RequestType", "Create")
    logical_id: str = event.get("LogicalResourceId", "codebuild-trigger")

    if request_type == "Delete":
        return {"PhysicalResourceId": event.get("PhysicalResourceId", logical_id)}

    project_names: list[str] = os.environ["PROJECT_NAMES"].split(",")
    logger.info("Starting CodeBuild projects", count=len(project_names), projects=project_names)

    started: list[str] = []
    for name in project_names:
        resp = codebuild.start_build(projectName=name)
        build_id: str = resp["build"]["id"]
        logger.info("Started build", project=name, build_id=build_id)
        started.append(build_id)

    logger.info("All builds started (completing asynchronously)")
    return {
        "PhysicalResourceId": logical_id,
        "Data": {
            "BuildCount": str(len(started)),
            "BuildIds": ",".join(started),
        },
    }
