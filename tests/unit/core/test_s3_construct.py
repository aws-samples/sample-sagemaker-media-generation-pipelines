"""
Unit tests for the BucketTemplate construct.

Tests verify that the S3 bucket construct creates resources with the
expected security configurations, encryption, lifecycle policies,
and IAM managed policies.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import (
    RemovalPolicy,
    assertions,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_s3 as s3,
)

from project_constructs.s3 import BucketTemplate

pytestmark = pytest.mark.core


def _create_bucket_stack() -> tuple[cdk.Stack, BucketTemplate]:
    """Helper to create a stack with a BucketTemplate for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))

    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)
    logging_bucket = s3.Bucket(
        stack,
        "TestLoggingBucket",
        encryption=s3.BucketEncryption.S3_MANAGED,
        removal_policy=RemovalPolicy.DESTROY,
    )

    bucket = BucketTemplate(
        stack,
        "TestBucket",
        bucket_name="test-bucket-name",
        kms_key=kms_key,
        logging_bucket=logging_bucket,
    )
    return stack, bucket


class TestBucketTemplateResources:
    """Tests for BucketTemplate resource creation."""

    def test_creates_s3_bucket(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::S3::Bucket", 2)  # logging + main bucket

    def test_bucket_has_kms_encryption(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketName": "test-bucket-name",
                "BucketEncryption": assertions.Match.object_like(
                    {
                        "ServerSideEncryptionConfiguration": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "BucketKeyEnabled": True,
                                        "ServerSideEncryptionByDefault": assertions.Match.object_like(
                                            {
                                                "SSEAlgorithm": "aws:kms",
                                            }
                                        ),
                                    }
                                )
                            ]
                        )
                    }
                ),
            },
        )

    def test_bucket_blocks_public_access(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketName": "test-bucket-name",
                "PublicAccessBlockConfiguration": {
                    "BlockPublicAcls": True,
                    "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True,
                    "RestrictPublicBuckets": True,
                },
            },
        )

    def test_bucket_has_versioning_enabled(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketName": "test-bucket-name",
                "VersioningConfiguration": {"Status": "Enabled"},
            },
        )

    def test_bucket_has_lifecycle_rules(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketName": "test-bucket-name",
                "LifecycleConfiguration": assertions.Match.object_like(
                    {
                        "Rules": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "ExpirationInDays": 90,
                                        "NoncurrentVersionExpiration": assertions.Match.object_like(
                                            {
                                                "NoncurrentDays": 7,
                                            }
                                        ),
                                        "Status": "Enabled",
                                    }
                                )
                            ]
                        )
                    }
                ),
            },
        )

    def test_bucket_has_access_logging(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::S3::Bucket",
            {
                "BucketName": "test-bucket-name",
                "LoggingConfiguration": assertions.Match.object_like(
                    {
                        "LogFilePrefix": "test-bucket-name",
                    }
                ),
            },
        )

    def test_bucket_has_destroy_removal_policy(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource(
            "AWS::S3::Bucket",
            {
                "Properties": assertions.Match.object_like(
                    {
                        "BucketName": "test-bucket-name",
                    }
                ),
                "UpdateReplacePolicy": "Delete",
                "DeletionPolicy": "Delete",
            },
        )


class TestBucketTemplateManagedPolicies:
    """Tests for BucketTemplate IAM managed policies."""

    def test_creates_three_managed_policies(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::IAM::ManagedPolicy", 3)

    def test_read_policy_has_correct_actions(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "TestBucket-S3ReadOnlyPolicy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": ["s3:GetObject", "s3:ListBucket", "s3:ListObjectsV2"],
                                        "Effect": "Allow",
                                        "Resource": [
                                            "arn:aws:s3:::test-bucket-name",
                                            "arn:aws:s3:::test-bucket-name/*",
                                        ],
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )

    def test_write_policy_has_correct_actions(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "TestBucket-S3WriteOnlyPolicy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": "s3:PutObject",
                                        "Effect": "Allow",
                                        "Resource": "arn:aws:s3:::test-bucket-name/*",
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )

    def test_read_write_policy_has_correct_actions(self):
        stack, _ = _create_bucket_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "TestBucket-S3ReadWritePolicy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": [
                                            "s3:GetObject",
                                            "s3:PutObject",
                                            "s3:DeleteObject",
                                            "s3:ListBucket",
                                            "s3:ListObjectsV2",
                                        ],
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )

    def test_policies_exposed_as_attributes(self):
        _, bucket = _create_bucket_stack()
        assert bucket.read_policy is not None
        assert bucket.write_policy is not None
        assert bucket.read_write_policy is not None

    def test_bucket_exposed_as_attribute(self):
        _, bucket = _create_bucket_stack()
        assert bucket.bucket is not None
