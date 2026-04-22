"""
Unit tests for the SnsTopicTemplate construct.

Tests verify that the SNS topic construct creates resources with
KMS encryption, service principal resource policies, and IAM
managed publish policies.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_kms as kms

from project_constructs.sns import SnsTopicTemplate

pytestmark = pytest.mark.core


def _create_sns_stack(
    service_principals: list[str] | None = None,
) -> tuple[cdk.Stack, SnsTopicTemplate]:
    """Helper to create a stack with an SnsTopicTemplate for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)

    topic = SnsTopicTemplate(
        stack,
        "TestTopic",
        topic_name="test-topic",
        kms_key=kms_key,
        service_principals=service_principals,
    )
    return stack, topic


class TestSnsTopicResources:
    """Tests for SnsTopicTemplate resource creation."""

    def test_creates_sns_topic(self):
        stack, _ = _create_sns_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::SNS::Topic", 1)

    def test_topic_has_kms_encryption(self):
        stack, _ = _create_sns_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SNS::Topic",
            {
                "TopicName": "test-topic",
                "KmsMasterKeyId": assertions.Match.any_value(),
            },
        )

    def test_topic_has_name(self):
        stack, _ = _create_sns_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SNS::Topic",
            {
                "TopicName": "test-topic",
            },
        )


class TestSnsTopicServicePrincipals:
    """Tests for SNS topic resource policy with service principals."""

    def test_no_resource_policy_without_principals(self):
        stack, _ = _create_sns_stack(service_principals=None)
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::SNS::TopicPolicy", 0)

    def test_resource_policy_with_sagemaker_principal(self):
        stack, _ = _create_sns_stack(service_principals=["sagemaker.amazonaws.com"])
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SNS::TopicPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": "sns:Publish",
                                        "Effect": "Allow",
                                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )


class TestSnsTopicIamPolicies:
    """Tests for SnsTopicTemplate IAM managed policies."""

    def test_creates_publish_managed_policy(self):
        stack, _ = _create_sns_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": "sns:Publish",
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )

    def test_publish_policy_exposed_as_attribute(self):
        _, topic = _create_sns_stack()
        assert topic.publish_policy is not None

    def test_topic_exposed_as_attribute(self):
        _, topic = _create_sns_stack()
        assert topic.topic is not None
