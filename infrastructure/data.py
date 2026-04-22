"""
Data infrastructure stack for the ComfyUI

This module defines the DataStack which creates and configures S3 buckets
for storing output images. It integrates with the SecurityStack to provide
encrypted storage with proper access controls and logging.
"""

from aws_cdk import RemovalPolicy, Stack
from aws_cdk import aws_s3 as s3
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from config.config import DynamoDBConfig, PipelineConfig, RetrievalConfig
from infrastructure.security import SecurityStack
from project_constructs.dynamodb import DynamoDbTemplate, GsiConfig
from project_constructs.retrieval import RetrievalConstruct
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

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        security_stack: SecurityStack,
        dynamodb_config: DynamoDBConfig,
        pipeline_config: PipelineConfig,
        prefix: str = "dev",
        retrieval_config: RetrievalConfig | None = None,
        ecr_prefix: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the DataStack with S3 buckets for data storage.

        Args:
            scope: The parent construct (typically the CDK App)
            construct_id: Unique identifier for this stack
            security_stack: SecurityStack instance providing KMS key and other security resources
            dynamodb_config: DynamoDB table key schema configuration
            pipeline_config: Full pipeline config (used to create A2I output buckets)
            prefix: Resource name prefix for multi-environment support (default: "dev")
            **kwargs: Additional keyword arguments passed to the parent Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        logger.info(f"Creating DataStack with prefix: {prefix}")

        # Create logging bucket for S3 access logs
        self.logs_bucket = s3.Bucket(
            self,
            f"{prefix}-LoggingBucket",
            bucket_name=f"{self.account}-{self.region}-{prefix}-logging-bucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,  # deleted upon stack destroy
            auto_delete_objects=True,
        )

        # Create input bucket for videos to be processed
        self.input_bucket = BucketTemplate(
            self,
            construct_id=f"{prefix}-InputBucket",
            bucket_name=f"{self.account}-{self.region}-{prefix}-input-bucket",
            kms_key=security_stack.kms_key,
            logging_bucket=self.logs_bucket,
        )

        # Create output bucket for processed videos
        self.output_bucket = BucketTemplate(
            self,
            construct_id=f"{prefix}-OutputBucket",
            bucket_name=f"{self.account}-{self.region}-{prefix}-output-bucket",
            kms_key=security_stack.kms_key,
            logging_bucket=self.logs_bucket,
        )

        # Create models bucket for storing downloaded model files
        self.models_bucket = BucketTemplate(
            self,
            construct_id=f"{prefix}-ModelsBucket",
            bucket_name=f"{self.account}-{self.region}-{prefix}-models-bucket",
            kms_key=security_stack.kms_key,
            logging_bucket=self.logs_bucket,
        )
        logger.info("Created models bucket")

        # Create DynamoDB table for pipeline results
        self.dynamodb_table = DynamoDbTemplate(
            self,
            f"{prefix}-ResultsTable",
            table_name="results",
            kms_key=security_stack.kms_key,
            partition_key=dynamodb_config.partition_key,
            sort_key=dynamodb_config.sort_key,
            gsi_name=f"gsi-{dynamodb_config.sort_key}",
            gsi_partition_key=dynamodb_config.sort_key,
            global_secondary_indexes=[
                GsiConfig(
                    index_name="gsi-review-loop-name",
                    partition_key="review_loop_name",
                ),
                GsiConfig(
                    index_name="gsi-selected-flag",
                    partition_key="selected_flag",
                    sort_key="step",
                ),
            ],
        )
        logger.info("Created DynamoDB results table")

        # Create A2I output buckets (one per active A2I config entry).
        # These live in DataStack so they survive A2IStack delete/recreate cycles.
        referenced_a2i_names = {ls_cfg.a2i_name for ls_cfg in pipeline_config.lambda_steps.values() if ls_cfg.a2i_name}
        active_a2i_names = referenced_a2i_names & set(pipeline_config.a2i.keys())

        self.a2i_output_buckets: dict[str, BucketTemplate] = {}
        for a2i_name in sorted(active_a2i_names):
            safe_name = a2i_name.replace("_", "-")
            bucket = BucketTemplate(
                self,
                f"{prefix}-{a2i_name}-A2IOutputBucket",
                bucket_name=f"{self.account}-{self.region}-{prefix}-{safe_name}-a2i-output-bucket",
                kms_key=security_stack.kms_key,
                logging_bucket=self.logs_bucket,
            )
            self.a2i_output_buckets[a2i_name] = bucket
            logger.info("Created A2I output bucket for: {}", a2i_name)

        # Conditionally create retrieval construct (OpenSearch + ingest pipeline)
        self.retrieval: RetrievalConstruct | None = None
        if retrieval_config is not None:
            self.retrieval = RetrievalConstruct(
                self,
                f"{prefix}-Retrieval",
                logging_bucket=self.logs_bucket,
                retrieval_config=retrieval_config,
                prefix=prefix,
                ecr_prefix=ecr_prefix,
                setup_configs=pipeline_config.setup,
                output_bucket=self.input_bucket,
                security_stack=security_stack,
            )
            logger.info("Created RetrievalConstruct inside DataStack")

            # Suppress CDK Nag for the auto-created S3 BucketNotificationsHandler
            nag_path = f"/{self.stack_name}/BucketNotificationsHandler050a0587b7544547bf325f094a3db834/Role"
            NagSuppressions.add_resource_suppressions_by_path(
                self,
                f"{nag_path}/DefaultPolicy/Resource",
                suppressions=[
                    {
                        "id": "AwsSolutions-IAM5",
                        "reason": "CDK-managed S3 bucket notifications handler requires wildcard permissions",
                    },
                ],
            )
            NagSuppressions.add_resource_suppressions_by_path(
                self,
                f"{nag_path}/Resource",
                suppressions=[
                    {
                        "id": "AwsSolutions-IAM4",
                        "reason": "CDK-managed S3 bucket notifications handler uses AWS managed policy",
                    },
                ],
            )
