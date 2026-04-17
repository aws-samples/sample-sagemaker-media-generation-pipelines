"""
Data infrastructure stack for the ComfyUI

This module defines the DataStack which creates and configures S3 buckets
for storing output images. It integrates with the SecurityStack to provide
encrypted storage with proper access controls and logging.
"""

from aws_cdk import (
    Stack,
    aws_s3 as s3,
    RemovalPolicy
)

from constructs import Construct
from infrastructure.security import SecurityStack
from project_constructs.s3 import BucketTemplate


class DataStack(Stack):
    """
    CDK Stack that provides data storage infrastructure for the ComfyUI application.
    
    This stack creates and configures:
    - Input S3 bucket for storing input data
    - Output S3 bucket for storing output data
    - Logging S3 bucket for access logs
    - KMS encryption for all buckets
    - Lifecycle policies for cost optimization
    
    The stack follows AWS best practices including:
    - Encryption at rest using KMS
    - Access logging for audit trails
    - Lifecycle policies for cost management
    - Block public access for security
    - SSL enforcement for data in transit
    
    Attributes:
        logs_bucket: S3 bucket for storing access logs
        input_bucket: BucketTemplate for input data storage
        output_bucket: BucketTemplate for output data storage
    """

    def __init__(self, scope: Construct, construct_id: str, security_stack: SecurityStack, **kwargs) -> None:
        """
        Initialize the DataStack with S3 buckets for data storage.
        
        Args:
            scope: The parent construct (typically the CDK App)
            construct_id: Unique identifier for this stack
            security_stack: SecurityStack instance providing KMS key and other security resources
            **kwargs: Additional keyword arguments passed to the parent Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        # Create logging bucket for S3 access logs
        self.logs_bucket = s3.Bucket(
            self, f"{construct_id}-LoggingBucket",
            bucket_name=f"{self.account}-{self.region}-{construct_id}-logging-bucket".lower(),
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY, # deleted upon stack destroy
            auto_delete_objects=True
        )

        # Create input bucket for videos to be processed
        self.input_bucket = BucketTemplate(
            self, construct_id=f"{construct_id}-InputBucket",
            bucket_name=f"{self.account}-{self.region}-{construct_id}-input-bucket".lower(),
            kms_key=security_stack.kms_key,
            logging_bucket=self.logs_bucket
        )

        # Create output bucket for processed videos
        self.output_bucket = BucketTemplate(
            self, construct_id=f"{construct_id}-OutputBucket",
            bucket_name=f"{self.account}-{self.region}-{construct_id}-output-bucket".lower(),
            kms_key=security_stack.kms_key,
            logging_bucket=self.logs_bucket
        )
