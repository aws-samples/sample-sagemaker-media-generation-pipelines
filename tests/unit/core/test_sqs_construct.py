"""
Unit tests for the SqsQueueTemplate construct.

Tests verify that the SQS queue construct creates resources with
KMS encryption, a dead-letter queue, and IAM managed policies for
sending and consuming messages.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_kms as kms

from project_constructs.sqs import SqsQueueTemplate

pytestmark = pytest.mark.core


def _create_sqs_stack(
    kms_key_provided: bool = True,
    visibility_timeout_seconds: int = 960,
    max_receive_count: int = 3,
) -> tuple[cdk.Stack, SqsQueueTemplate]:
    """Helper to create a stack with an SqsQueueTemplate for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))

    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True) if kms_key_provided else None

    queue = SqsQueueTemplate(
        stack,
        "TestQueue",
        queue_name="test-queue",
        kms_key=kms_key,
        visibility_timeout_seconds=visibility_timeout_seconds,
        max_receive_count=max_receive_count,
    )
    return stack, queue


class TestSqsQueueResources:
    """Tests for SqsQueueTemplate resource creation."""

    def test_creates_main_queue_and_dlq(self):
        stack, _ = _create_sqs_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::SQS::Queue", 2)  # main + DLQ

    def test_main_queue_has_correct_name(self):
        stack, _ = _create_sqs_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SQS::Queue",
            {
                "QueueName": "test-queue",
            },
        )

    def test_dlq_has_correct_name(self):
        stack, _ = _create_sqs_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SQS::Queue",
            {
                "QueueName": "test-queue-dlq",
            },
        )

    def test_main_queue_has_dlq_configured(self):
        stack, _ = _create_sqs_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SQS::Queue",
            {
                "QueueName": "test-queue",
                "RedrivePolicy": assertions.Match.object_like(
                    {
                        "maxReceiveCount": 3,
                    }
                ),
            },
        )


class TestSqsQueueEncryption:
    """Tests for SqsQueueTemplate KMS encryption."""

    def test_queue_has_kms_encryption(self):
        stack, _ = _create_sqs_stack(kms_key_provided=True)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SQS::Queue",
            {
                "QueueName": "test-queue",
                "KmsMasterKeyId": assertions.Match.any_value(),
            },
        )

    def test_dlq_has_kms_encryption(self):
        stack, _ = _create_sqs_stack(kms_key_provided=True)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SQS::Queue",
            {
                "QueueName": "test-queue-dlq",
                "KmsMasterKeyId": assertions.Match.any_value(),
            },
        )

    def test_queue_uses_sqs_managed_encryption_without_kms(self):
        stack, _ = _create_sqs_stack(kms_key_provided=False)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SQS::Queue",
            {
                "QueueName": "test-queue",
                "SqsManagedSseEnabled": True,
            },
        )


class TestSqsQueueIamPolicies:
    """Tests for SqsQueueTemplate IAM managed policies."""

    def test_creates_send_and_consume_policies(self):
        stack, _ = _create_sqs_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::IAM::ManagedPolicy", 2)

    def test_send_policy_has_correct_actions(self):
        stack, _ = _create_sqs_stack()
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
                                        "Action": "sqs:SendMessage",
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )

    def test_consume_policy_has_correct_actions(self):
        stack, _ = _create_sqs_stack()
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
                                        "Action": [
                                            "sqs:ReceiveMessage",
                                            "sqs:DeleteMessage",
                                            "sqs:GetQueueAttributes",
                                            "sqs:ChangeMessageVisibility",
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
        _, queue = _create_sqs_stack()
        assert queue.send_policy is not None
        assert queue.consume_policy is not None

    def test_queue_and_dlq_exposed_as_attributes(self):
        _, queue = _create_sqs_stack()
        assert queue.queue is not None
        assert queue.dlq is not None
