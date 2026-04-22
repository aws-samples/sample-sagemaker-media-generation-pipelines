"""
Unit tests for the SsmParameter construct.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_kms as kms
from aws_cdk import aws_lambda as lambda_

from project_constructs.ssm_parameter import SsmParameter

pytestmark = pytest.mark.core


def _mock_lambda_asset(*a, **kw):
    return lambda_.Code.from_inline("def handler(event, context): pass")


def _create_stack_with_param(kms_key=None):
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
    key = None
    if kms_key:
        key = kms.Key(stack, "TestKey")
    param = SsmParameter(
        stack,
        "TestParam",
        parameter_name="/test/my-parameter",
        string_value='{"key": "value"}',
        description="Test parameter",
        kms_key=key,
    )
    return stack, param


class TestSsmParameter:
    def test_creates_ssm_parameter(self):
        stack, _ = _create_stack_with_param()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::SSM::Parameter", 1)

    def test_parameter_has_correct_properties(self):
        stack, _ = _create_stack_with_param()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SSM::Parameter",
            {
                "Name": "/test/my-parameter",
                "Type": "String",
                "Value": '{"key": "value"}',
                "Tier": "Advanced",
            },
        )

    def test_creates_read_policy(self):
        stack, param = _create_stack_with_param()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-my-parameter-ssm-read-policy",
            },
        )

    def test_creates_write_policy(self):
        stack, param = _create_stack_with_param()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-my-parameter-ssm-write-policy",
            },
        )

    def test_read_policy_has_get_actions(self):
        stack, _ = _create_stack_with_param()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-my-parameter-ssm-read-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": ["ssm:GetParameter", "ssm:GetParameters"],
                                        "Effect": "Allow",
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_write_policy_has_put_action(self):
        stack, _ = _create_stack_with_param()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-my-parameter-ssm-write-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": "ssm:PutParameter",
                                        "Effect": "Allow",
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_kms_key_adds_decrypt_to_read_policy(self):
        stack, _ = _create_stack_with_param(kms_key=True)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-my-parameter-ssm-read-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": "kms:Decrypt",
                                        "Effect": "Allow",
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_kms_key_adds_encrypt_to_write_policy(self):
        stack, _ = _create_stack_with_param(kms_key=True)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-my-parameter-ssm-write-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": ["kms:Encrypt", "kms:GenerateDataKey"],
                                        "Effect": "Allow",
                                    }
                                ),
                            ]
                        ),
                    }
                ),
            },
        )

    def test_no_kms_key_omits_kms_from_read_policy(self):
        stack, _ = _create_stack_with_param()
        template = assertions.Template.from_stack(stack)
        # Only 1 statement (ssm:Get*) — no kms statement
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-my-parameter-ssm-read-policy",
                "PolicyDocument": {
                    "Statement": [
                        assertions.Match.object_like(
                            {
                                "Action": ["ssm:GetParameter", "ssm:GetParameters"],
                            }
                        ),
                    ],
                    "Version": "2012-10-17",
                },
            },
        )

    def test_exposes_parameter_attribute(self):
        _, param = _create_stack_with_param()
        assert param.parameter is not None
        assert param.read_policy is not None
        assert param.write_policy is not None
