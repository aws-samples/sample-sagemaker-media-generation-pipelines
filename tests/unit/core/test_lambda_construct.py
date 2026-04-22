"""
Unit tests for the LambdaTemplate construct.

Tests verify that the Lambda function construct creates resources with
the expected VPC integration, IAM roles, CloudWatch logging, and
security configurations.

Note: The LambdaTemplate uses Code.from_asset with Docker bundling,
so we mock the bundling to avoid requiring Docker during tests.
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import (
    assertions,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_lambda as lambda_,
)

from project_constructs.lambda_function import LambdaTemplate
from tests.unit.core.conftest import _mock_from_asset

pytestmark = pytest.mark.core


def _create_lambda_stack(
    env_vars: dict = None,
    kms_key_enabled: bool = False,
    timeout: int = 600,
    memory_size: int = 2048,
) -> tuple[cdk.Stack, LambdaTemplate]:
    """Helper to create a stack with a LambdaTemplate for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))

    vpc = ec2.Vpc(stack, "TestVpc", max_azs=2)
    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True) if kms_key_enabled else None

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        lambda_template = LambdaTemplate(
            stack,
            "TestLambda",
            function_name="test-function",
            lambda_path="trigger_processing_job",
            description="Test lambda function",
            vpc=vpc,
            timeout=timeout,
            memory_size=memory_size,
            env_vars=env_vars,
            kms_key=kms_key,
        )
    return stack, lambda_template


class TestLambdaTemplateResources:
    """Tests for LambdaTemplate resource creation."""

    def test_creates_lambda_function(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Lambda::Function", 1)

    def test_lambda_uses_python_313_runtime(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "Runtime": "python3.13",
            },
        )

    def test_lambda_uses_arm64_architecture(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "Architectures": ["arm64"],
            },
        )

    def test_lambda_has_correct_handler(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "Handler": "index.lambda_handler",
            },
        )

    def test_lambda_has_correct_timeout(self):
        stack, _ = _create_lambda_stack(timeout=300)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "Timeout": 300,
            },
        )

    def test_lambda_has_correct_memory_size(self):
        stack, _ = _create_lambda_stack(memory_size=1024)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "MemorySize": 1024,
            },
        )

    def test_lambda_has_description(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "Description": "Test lambda function",
            },
        )

    def test_lambda_runs_in_vpc(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "VpcConfig": assertions.Match.object_like(
                    {
                        "SubnetIds": assertions.Match.any_value(),
                        "SecurityGroupIds": assertions.Match.any_value(),
                    }
                ),
            },
        )

    def test_lambda_has_default_timeout(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "Timeout": 600,
            },
        )

    def test_lambda_has_default_memory(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "MemorySize": 2048,
            },
        )


class TestLambdaTemplateIAM:
    """Tests for LambdaTemplate IAM role configuration."""

    def test_creates_iam_role(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "AssumeRolePolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": "sts:AssumeRole",
                                        "Effect": "Allow",
                                        "Principal": {"Service": "lambda.amazonaws.com"},
                                    }
                                )
                            ]
                        ),
                    }
                ),
            },
        )

    def test_role_has_basic_execution_policy(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "ManagedPolicyArns": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Fn::Join": [
                                    "",
                                    [
                                        "arn:",
                                        {"Ref": "AWS::Partition"},
                                        ":iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
                                    ],
                                ],
                            }
                        ),
                    ]
                ),
            },
        )

    def test_role_has_vpc_access_policy(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "ManagedPolicyArns": assertions.Match.array_with(
                    [
                        assertions.Match.object_like(
                            {
                                "Fn::Join": [
                                    "",
                                    [
                                        "arn:",
                                        {"Ref": "AWS::Partition"},
                                        ":iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole",
                                    ],
                                ],
                            }
                        ),
                    ]
                ),
            },
        )


class TestLambdaTemplateLogging:
    """Tests for LambdaTemplate CloudWatch logging."""

    def test_creates_log_group(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Logs::LogGroup",
            {
                "LogGroupName": "/aws/lambda/test-function-Lambda",
                "RetentionInDays": 30,
            },
        )

    def test_log_group_has_destroy_removal_policy(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource(
            "AWS::Logs::LogGroup",
            {
                "Properties": assertions.Match.object_like(
                    {
                        "LogGroupName": "/aws/lambda/test-function-Lambda",
                    }
                ),
                "UpdateReplacePolicy": "Delete",
                "DeletionPolicy": "Delete",
            },
        )


class TestLambdaTemplateEnvironmentVariables:
    """Tests for LambdaTemplate environment variable handling."""

    def test_lambda_with_env_vars(self):
        stack, _ = _create_lambda_stack(env_vars={"MY_VAR": "my_value"})
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "Environment": assertions.Match.object_like(
                    {
                        "Variables": assertions.Match.object_like(
                            {
                                "MY_VAR": "my_value",
                            }
                        ),
                    }
                ),
            },
        )

    def test_lambda_without_env_vars(self):
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Lambda::Function", 1)


class TestLambdaTemplateKMS:
    """Tests for LambdaTemplate KMS encryption."""

    def test_lambda_with_kms_encryption(self):
        stack, _ = _create_lambda_stack(kms_key_enabled=True)
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "KmsKeyArn": assertions.Match.any_value(),
            },
        )

    def test_lambda_without_kms_encryption(self):
        stack, _ = _create_lambda_stack(kms_key_enabled=False)
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Lambda::Function", 1)


class TestLambdaTemplateAttributes:
    """Tests for LambdaTemplate exposed attributes."""

    def test_function_role_exposed(self):
        _, lambda_template = _create_lambda_stack()
        assert lambda_template.function_role is not None

    def test_log_group_exposed(self):
        _, lambda_template = _create_lambda_stack()
        assert lambda_template.log_group is not None

    def test_lambda_function_exposed(self):
        _, lambda_template = _create_lambda_stack()
        assert lambda_template.lambda_function is not None


class TestLambdaTemplateEfsFilesystem:
    """Tests for LambdaTemplate optional EFS filesystem parameter."""

    def test_lambda_with_efs_filesystem(self):
        """When filesystem is provided, Lambda has FileSystemConfigs."""
        from aws_cdk import aws_efs as efs

        app = cdk.App()
        stack = cdk.Stack(app, "EfsTestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
        vpc = ec2.Vpc(stack, "Vpc", max_azs=2)
        sg = ec2.SecurityGroup(stack, "SG", vpc=vpc)
        fs = efs.FileSystem(stack, "Efs", vpc=vpc, security_group=sg)
        ap = efs.AccessPoint(stack, "AP", file_system=fs, path="/mnt/data")
        efs_fs = lambda_.FileSystem.from_efs_access_point(ap, "/mnt/efs")

        with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
            LambdaTemplate(
                stack,
                "EfsLambda",
                function_name="efs-test",
                lambda_path="trigger_processing_job",
                description="EFS test",
                vpc=vpc,
                filesystem=efs_fs,
            )

        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {
                "FileSystemConfigs": assertions.Match.any_value(),
            },
        )

    def test_lambda_without_efs_filesystem(self):
        """When filesystem is omitted, Lambda has no FileSystemConfigs."""
        stack, _ = _create_lambda_stack()
        template = assertions.Template.from_stack(stack)
        lambdas = template.find_resources("AWS::Lambda::Function")
        for _, res in lambdas.items():
            assert "FileSystemConfigs" not in res.get("Properties", {})
