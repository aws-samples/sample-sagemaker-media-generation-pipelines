"""
Unit tests for the A2IHumanReview construct.

Tests verify that the A2I construct creates the expected resources:
SNS topic, Cognito user pool, workteam, IAM role, custom resources
for HumanTaskUi and FlowDefinition, and managed policies.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import (
    RemovalPolicy,
    assertions,
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

from config.config import A2IConfig
from project_constructs.a2i.main import A2IHumanReview
from project_constructs.s3 import BucketTemplate

pytestmark = pytest.mark.core


def _create_a2i_stack() -> tuple[cdk.Stack, A2IHumanReview]:
    """Helper to create a stack with an A2IHumanReview for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))

    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)
    logging_bucket = s3.Bucket(
        stack,
        "LogBucket",
        encryption=s3.BucketEncryption.S3_MANAGED,
        removal_policy=RemovalPolicy.DESTROY,
    )

    input_bucket = BucketTemplate(
        stack,
        "InputBucket",
        bucket_name="test-input-bucket",
        kms_key=kms_key,
        logging_bucket=logging_bucket,
    )
    output_bucket = BucketTemplate(
        stack,
        "OutputBucket",
        bucket_name="test-output-bucket",
        kms_key=kms_key,
        logging_bucket=logging_bucket,
    )

    kms_key_policy = iam.ManagedPolicy(
        stack,
        "KmsPolicy",
        statements=[
            iam.PolicyStatement(
                actions=["kms:Decrypt", "kms:GenerateDataKey"],
                resources=[kms_key.key_arn],
            ),
        ],
    )

    cfg = A2IConfig(
        media_type="video",
        task_title="Review videos",
        task_description="Review generated videos",
        task_count=1,
        task_timeout_seconds=3600,
    )

    a2i = A2IHumanReview(
        stack,
        "TestA2I",
        flow_definition_name="test-flow",
        input_bucket=input_bucket,
        output_bucket=output_bucket,
        kms_key=kms_key,
        kms_key_policy=kms_key_policy,
        cfg=cfg,
    )
    return stack, a2i


class TestA2IResources:
    """Tests for A2IHumanReview resource creation."""

    def test_creates_sns_topic(self):
        stack, _ = _create_a2i_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SNS::Topic",
            {
                "TopicName": "test-flow-notifications",
            },
        )

    def test_creates_cognito_user_pool(self):
        stack, _ = _create_a2i_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPool", 1)

    def test_creates_workteam(self):
        stack, _ = _create_a2i_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::SageMaker::Workteam", 1)

    def test_creates_iam_role_for_flow_definition(self):
        stack, _ = _create_a2i_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "RoleName": "test-flow-a2i-role",
                "AssumeRolePolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )

    def test_creates_custom_resources(self):
        """A2I uses AwsCustomResource for HumanTaskUi and FlowDefinition."""
        stack, _ = _create_a2i_stack()
        template = assertions.Template.from_stack(stack)
        # AwsCustomResource creates AWS::CloudFormation::CustomResource
        resources = template.find_resources("Custom::AWS")
        assert len(resources) >= 2, "Expected at least 2 custom resources (HumanTaskUi + FlowDefinition)"


class TestA2IAttributes:
    """Tests for A2IHumanReview exposed attributes."""

    def test_flow_definition_arn_set(self):
        _, a2i = _create_a2i_stack()
        assert a2i.flow_definition_arn is not None
        assert "flow-definition" in a2i.flow_definition_arn

    def test_notification_topic_set(self):
        _, a2i = _create_a2i_stack()
        assert a2i.notification_topic is not None

    def test_cognito_set(self):
        _, a2i = _create_a2i_stack()
        assert a2i.cognito is not None
        assert a2i.cognito_user_pool_id is not None
        assert a2i.cognito_client_id is not None

    def test_workteam_set(self):
        _, a2i = _create_a2i_stack()
        assert a2i.workteam is not None

    def test_role_set(self):
        _, a2i = _create_a2i_stack()
        assert a2i.role is not None
