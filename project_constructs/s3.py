"""
Reusable S3 bucket construct with security best practices.

This module provides a BucketTemplate construct that creates S3 buckets
with comprehensive security configurations including KMS encryption,
access logging, lifecycle policies, and IAM policies for different
access patterns.
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_s3 as s3,
)
from cdk_nag import NagSuppressions
from constructs import Construct


class BucketTemplate(Construct):
    """
    A reusable S3 bucket construct with security best practices.

    This construct creates an S3 bucket with:
    - KMS encryption at rest
    - Access logging to a separate bucket
    - Lifecycle policies for cost optimization
    - Block public access for security
    - SSL enforcement for data in transit
    - Versioning enabled for data protection
    - Pre-configured IAM policies for different access patterns

    The construct also creates three IAM managed policies:
    - Read-only policy for reading objects and listing buckets
    - Write-only policy for putting objects
    - Read-write policy for full bucket access

    Attributes:
        bucket: The S3 bucket resource
        read_policy: IAM policy for read-only access
        write_policy: IAM policy for write-only access
        read_write_policy: IAM policy for read-write access
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        bucket_name: str,
        kms_key: kms.Key | None = None,
        logging_bucket: s3.Bucket | None = None,
    ) -> None:
        """
        Initialize the BucketTemplate with security configurations.

        Args:
            scope: The parent construct
            construct_id: Unique identifier for this construct
            bucket_name: Name for the S3 bucket (must be globally unique)
            kms_key: KMS key for bucket encryption (optional, uses S3-managed encryption when None)
            logging_bucket: S3 bucket for storing access logs (optional)
        """
        super().__init__(scope, construct_id)

        # Determine encryption settings
        if kms_key is not None:
            encryption = s3.BucketEncryption.KMS
            encryption_key = kms_key
        else:
            encryption = s3.BucketEncryption.S3_MANAGED
            encryption_key = None

        # Create S3 bucket with comprehensive security configuration
        bucket_kwargs: dict = {}
        if logging_bucket is not None:
            bucket_kwargs["server_access_logs_bucket"] = logging_bucket
            bucket_kwargs["server_access_logs_prefix"] = bucket_name

        self.bucket = s3.Bucket(
            self,
            construct_id,
            bucket_name=bucket_name,
            encryption=encryption,
            bucket_key_enabled=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            encryption_key=encryption_key,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    expiration=Duration.days(90),
                    noncurrent_version_expiration=Duration.days(7),
                )
            ],
            **bucket_kwargs,
        )

        # Create IAM policy for read-only access
        self.read_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-S3ReadOnlyPolicy",
            managed_policy_name=f"{construct_id}-S3ReadOnlyPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "s3:GetObject",
                        "s3:ListBucket",
                        "s3:ListObjectsV2",
                    ],
                    resources=[
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                )
            ],
        )

        # Create IAM policy for write-only access
        self.write_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-S3WriteOnlyPolicy",
            managed_policy_name=f"{construct_id}-S3WriteOnlyPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "s3:PutObject",
                    ],
                    resources=[f"arn:aws:s3:::{bucket_name}/*"],
                )
            ],
        )

        # Create IAM policy for read-write access
        self.read_write_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-S3ReadWritePolicy",
            managed_policy_name=f"{construct_id}-S3ReadWritePolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket",
                        "s3:ListObjectsV2",
                    ],
                    resources=[
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                )
            ],
        )

        # Suppress CDK Nag warnings for wildcard permissions (required for S3 operations)
        NagSuppressions.add_resource_suppressions(
            self.read_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Star permission is needed to read any file",
                }
            ],
        )

        NagSuppressions.add_resource_suppressions(
            self.write_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Star permission is needed to write any file",
                }
            ],
        )

        NagSuppressions.add_resource_suppressions(
            self.read_write_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Star permission is needed to write any file",
                }
            ],
        )
