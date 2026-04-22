"""
Unit tests for the WorkteamTemplate construct.

Tests verify that the SageMaker private workteam construct creates
a CfnWorkteam backed by a Cognito user pool group with SNS notifications.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_kms as kms

from project_constructs.cognito import CognitoWorkforceTemplate
from project_constructs.sns import SnsTopicTemplate
from project_constructs.workteam import WorkteamTemplate

pytestmark = pytest.mark.core


def _create_workteam_stack() -> tuple[cdk.Stack, WorkteamTemplate]:
    """Helper to create a stack with a WorkteamTemplate for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)

    cognito = CognitoWorkforceTemplate(
        stack,
        "TestCognito",
        pool_name="test-reviewers",
        group_name="test-group",
        domain_prefix="test-workforce",
    )

    notification_topic = SnsTopicTemplate(
        stack,
        "TestTopic",
        topic_name="test-notifications",
        kms_key=kms_key,
        service_principals=["sagemaker.amazonaws.com"],
    )

    workteam = WorkteamTemplate(
        stack,
        "TestWorkteam",
        workteam_name="test-team",
        cognito=cognito,
        notification_topic=notification_topic,
    )
    return stack, workteam


class TestWorkteamResources:
    """Tests for WorkteamTemplate resource creation."""

    def test_creates_workteam(self):
        stack, _ = _create_workteam_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::SageMaker::Workteam", 1)

    def test_workteam_has_correct_name(self):
        stack, _ = _create_workteam_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SageMaker::Workteam",
            {
                "WorkteamName": "test-team",
            },
        )

    def test_workteam_has_cognito_member_definition(self):
        stack, _ = _create_workteam_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SageMaker::Workteam",
            {
                "MemberDefinitions": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "CognitoMemberDefinition": assertions.Match.object_like(
                                    {
                                        "CognitoUserGroup": "test-group",
                                    }
                                ),
                            }
                        )
                    ]
                ),
            },
        )

    def test_workteam_has_notification_config(self):
        stack, _ = _create_workteam_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SageMaker::Workteam",
            {
                "NotificationConfiguration": assertions.Match.object_like(
                    {
                        "NotificationTopicArn": assertions.Match.any_value(),
                    }
                ),
            },
        )


class TestWorkteamAttributes:
    """Tests for WorkteamTemplate exposed attributes."""

    def test_workteam_exposed_as_attribute(self):
        _, workteam = _create_workteam_stack()
        assert workteam.workteam is not None

    def test_workteam_name_set(self):
        _, workteam = _create_workteam_stack()
        assert workteam.workteam_name == "test-team"

    def test_workteam_arn_contains_name(self):
        _, workteam = _create_workteam_stack()
        assert "test-team" in workteam.workteam_arn
        assert "private-crowd" in workteam.workteam_arn
