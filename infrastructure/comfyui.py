"""
Main ComfyUI infrastructure stack

This module defines the ComfyUIStack which orchestrates the creation of
Lambda triggers and SageMaker processing jobs and for ComfyUI.
It integrates with the SecurityStack and DataStack
to provide a complete serverless video processing pipeline.
"""

from aws_cdk import Stack

from constructs import Construct

from config.config import ContainerConfig
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack
from project_constructs.processing_job.main import ReusableProcessingJob


class ComfyUiSmStack(Stack):
    """
    CDK Stack that provides the main ComfyUI Stack.
    
    This stack creates and configures:
    - SageMaker processing job running ComfyUI
    - Lambda function for SageMaker processing job triggering
    - Integration with input/output S3 buckets

    """

    def __init__(self, scope: Construct, construct_id: str, security_stack: SecurityStack,
                 data_stack: DataStack, cfg: ContainerConfig, **kwargs) -> None:
        """
        Initialize the ComfyUiStack with processing job infrastructure.
        
        Args:
            scope: The parent construct (typically the CDK App)
            construct_id: Unique identifier for this stack
            security_stack: SecurityStack instance providing VPC, KMS, and security resources
            data_stack: DataStack instance providing S3 buckets for input/output
            cfg: ContainerConfig with processing job configuration parameters
            **kwargs: Additional keyword arguments passed to the parent Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        # Create reusable processing job for ComfyUI
        self.processing_job = ReusableProcessingJob(
            self, f"{construct_id}-ProcessingJob",
            job_name="ComfyUI",
            input_buckets={},
            output_bucket=data_stack.output_bucket,
            security_stack=security_stack,
            lambda_trigger=True,
            cfg=cfg,
        )
