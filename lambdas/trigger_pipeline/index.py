"""
Lambda function for triggering SageMaker Pipeline execution.

This module provides a Lambda function that starts a SageMaker Pipeline
execution using the pipeline name from environment variables.
"""

import logging
import os

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sagemaker_client = boto3.client("sagemaker")


def lambda_handler(event: dict, context: object) -> dict:
    """
    Start a SageMaker Pipeline execution.

    Reads the PIPELINE_NAME environment variable and calls
    StartPipelineExecution. Returns the execution ARN.

    Args:
        event: Lambda event payload (unused).
        context: Lambda context object (unused).

    Returns:
        dict with PipelineExecutionArn.
    """
    pipeline_name: str = os.environ["PIPELINE_NAME"]
    logger.info("Triggering pipeline execution for: %s", pipeline_name)

    try:
        response: dict = sagemaker_client.start_pipeline_execution(PipelineName=pipeline_name)
        execution_arn: str = response["PipelineExecutionArn"]
        logger.info("Pipeline execution started: %s", execution_arn)
        return {"PipelineExecutionArn": execution_arn}
    except Exception:
        logger.error("Failed to start pipeline execution: %s", pipeline_name)
        raise
