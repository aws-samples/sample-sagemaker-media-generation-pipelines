"""
Lambda function for triggering SageMaker processing jobs.

This module provides a Lambda function that creates and starts SageMaker
processing jobs. It reads the job configuration from environment variables
 and creates jobs with unique names based on timestamps.
"""

from datetime import datetime
import json
import os

import boto3
from aws_lambda_powertools import Logger

# Initialize logger for structured logging
logger = Logger()

# Load processing job definition from environment variable
processing_job_definition = json.loads(os.getenv('PROCESSING_JOB_DEFINITION'))


def lambda_handler(event, context):
    """
    AWS Lambda handler function for triggering SageMaker processing jobs.
    
    This function creates a SageMaker processing job with a unique name based
    on the current timestamp. It uses the job definition stored in environment
    variables to configure the processing job parameters.
    
    The function performs the following steps:
    1. Generate a unique job name using current timestamp
    2. Extract job arguments from the environment configuration
    3. Create a SageMaker processing job using the AWS SDK
    4. Return the response from the SageMaker API
    
    Args:
        event: AWS Lambda event object (not used in this implementation)
        context: AWS Lambda context object (not used in this implementation)
        
    Returns:
        dict: Response from the SageMaker CreateProcessingJob API call
        
    Raises:
        ClientError: If the SageMaker API call fails
        KeyError: If required environment variables are missing
        json.JSONDecodeError: If the processing job definition is malformed
        
    Environment Variables:
        PROCESSING_JOB_DEFINITION: JSON string containing the complete job definition
        JOB_NAME: Base name for the processing job (timestamp will be appended)
        
    Example:
        The function expects environment variables like:
        - PROCESSING_JOB_DEFINITION: '{"Arguments": {...}, "Name": "ComfyUI", ...}'
        - JOB_NAME: "ComfyUI"
        
        It will create a job with name like: "ComfyUI-2024-08-27-17-30-45"
    """
    # Generate unique job name with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_name = f"{os.getenv('JOB_NAME')}-{current_time}".replace("_", "-")

    # Log job information for debugging
    logger.info(f"Processing Job Name: {job_name}")
    logger.info(f"Processing Job Definition: {processing_job_definition}")

    # Initialize SageMaker client
    sagemaker_client = boto3.client('sagemaker')

    # Extract arguments from the job definition
    args = processing_job_definition['Arguments']

    # Create the SageMaker processing job
    response = sagemaker_client.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingResources=args['ProcessingResources'],
        AppSpecification=args['AppSpecification'],
        ProcessingInputs=args['ProcessingInputs'],
        ProcessingOutputConfig=args['ProcessingOutputConfig'],
        Environment=args['Environment'],
        RoleArn=args['RoleArn'],
        StoppingCondition=args['StoppingCondition'],
        NetworkConfig=args['NetworkConfig']
    )

    return response
