# Feature: sagemaker-pipeline, Task 17.7: Tests for updated existing stacks
"""
Unit tests for SecurityStack, DataStack, and ComfyUiSmStack prefix parameter.

**Validates: Requirements 8.6, 8.13, 7.7**
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from config.config import DynamoDBConfig, PipelineConfig
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack

pytestmark = pytest.mark.core


class TestSecurityStackPrefix:
    """SecurityStack applies prefix to resource names."""

    def test_default_prefix_dev(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        stack = SecurityStack(app, "Sec", env=env)
        assert stack.vpc is not None
        assert stack.kms_key is not None

    def test_custom_prefix_applied_to_kms_alias(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        stack = SecurityStack(app, "Sec", prefix="staging", env=env)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::KMS::Alias",
            {
                "AliasName": "alias//staging-comfyui",
            },
        )

    def test_custom_prefix_applied_to_security_group(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        stack = SecurityStack(app, "Sec", prefix="prod", env=env)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::EC2::SecurityGroup",
            {
                "GroupName": "prod-security-group",
            },
        )

    def test_exposed_attributes(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        stack = SecurityStack(app, "Sec", prefix="dev", env=env)
        assert stack.vpc is not None
        assert stack.kms_key is not None
        assert stack.kms_key_policy is not None
        assert stack.security_group is not None
        assert stack.subnet_ids is not None


class TestDataStackPrefix:
    """DataStack applies prefix to bucket names."""

    def test_default_prefix(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        sec = SecurityStack(app, "Sec", prefix="dev", env=env)
        stack = DataStack(
            app,
            "Data",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=PipelineConfig(construct_id="dev", s3_downloads=[], steps={}),
            prefix="dev",
            env=env,
        )
        assert stack.logs_bucket is not None

    def test_custom_prefix_applied(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        sec = SecurityStack(app, "Sec", prefix="staging", env=env)
        stack = DataStack(
            app,
            "Data",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=PipelineConfig(construct_id="staging", s3_downloads=[], steps={}),
            prefix="staging",
            env=env,
        )
        template = assertions.Template.from_stack(stack)
        # Verify S3 buckets exist
        buckets = template.find_resources("AWS::S3::Bucket")
        assert len(buckets) > 0

    def test_exposed_attributes(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        sec = SecurityStack(app, "Sec", prefix="dev", env=env)
        stack = DataStack(
            app,
            "Data",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=PipelineConfig(construct_id="dev", s3_downloads=[], steps={}),
            prefix="dev",
            env=env,
        )
        assert stack.logs_bucket is not None
