"""
Unit tests for the CodeBuildProject construct.

Tests verify CodeBuild project creation with VPC integration, KMS encryption,
CloudWatch logging, privileged mode, and exposed managed policies for logs
and VPC access.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import (
    assertions,
)
from aws_cdk import (
    aws_codebuild as codebuild,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_s3 as s3,
)

from project_constructs.codebuild import CodeBuildProject

pytestmark = pytest.mark.core


def _create_codebuild_construct_stack(
    env_vars: dict[str, codebuild.BuildEnvironmentVariable] | None = None,
) -> tuple[cdk.Stack, CodeBuildProject, assertions.Template]:
    """Helper to create a stack with a CodeBuildProject for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
    vpc = ec2.Vpc(stack, "TestVpc", max_azs=2)
    sg = ec2.SecurityGroup(stack, "TestSg", vpc=vpc)
    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)
    source_bucket = s3.Bucket(stack, "SourceBucket")
    source = codebuild.Source.s3(bucket=source_bucket, path="source.zip")

    cb = CodeBuildProject(
        stack,
        "TestCB",
        source=source,
        vpc=vpc,
        subnet_selection=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
        security_group=sg,
        kms_key=kms_key,
        environment_variables=env_vars,
    )
    template = assertions.Template.from_stack(stack)
    return stack, cb, template


class TestCodeBuildProjectResources:
    """Tests for CodeBuild project resource creation."""

    def test_creates_codebuild_project(self):
        _, _, template = _create_codebuild_construct_stack()
        template.resource_count_is("AWS::CodeBuild::Project", 1)

    def test_privileged_mode_enabled(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {"Environment": assertions.Match.object_like({"PrivilegedMode": True})},
        )

    def test_project_in_vpc(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {"VpcConfig": assertions.Match.object_like({"VpcId": assertions.Match.any_value()})},
        )

    def test_project_has_kms_encryption(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {"EncryptionKey": assertions.Match.any_value()},
        )

    def test_project_uses_standard_7_image(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "Environment": assertions.Match.object_like(
                    {"Image": assertions.Match.string_like_regexp(".*standard.*7\\.0")}
                )
            },
        )

    def test_project_uses_buildspec_from_source(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {"Source": assertions.Match.object_like({"BuildSpec": "buildspec.yml"})},
        )

    def test_project_has_timeout(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {"TimeoutInMinutes": 60},
        )

    def test_project_has_cloudwatch_logging(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "LogsConfig": assertions.Match.object_like(
                    {"CloudWatchLogs": assertions.Match.object_like({"Status": "ENABLED"})}
                )
            },
        )

    def test_creates_log_group(self):
        _, _, template = _create_codebuild_construct_stack()
        template.resource_count_is("AWS::Logs::LogGroup", 1)

    def test_log_group_retention_one_month(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 30})

    def test_log_group_removal_policy_destroy(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource(
            "AWS::Logs::LogGroup",
            {"UpdateReplacePolicy": "Delete", "DeletionPolicy": "Delete"},
        )


class TestCodeBuildProjectIAM:
    """Tests for CodeBuild project IAM configuration."""

    def test_creates_iam_role(self):
        _, _, template = _create_codebuild_construct_stack()
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
                                        "Principal": {"Service": "codebuild.amazonaws.com"},
                                    }
                                )
                            ]
                        )
                    }
                ),
            },
        )

    def test_logs_policy_created(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": assertions.Match.array_with(["logs:CreateLogGroup"]),
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        )
                    }
                )
            },
        )

    def test_vpc_policy_created(self):
        _, _, template = _create_codebuild_construct_stack()
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": assertions.Match.array_with(["ec2:CreateNetworkInterface"]),
                                        "Effect": "Allow",
                                        "Resource": "*",
                                    }
                                )
                            ]
                        )
                    }
                )
            },
        )


class TestCodeBuildProjectAttributes:
    """Tests for CodeBuildProject exposed attributes."""

    def test_project_exposed(self):
        _, cb, _ = _create_codebuild_construct_stack()
        assert cb.project is not None

    def test_role_exposed(self):
        _, cb, _ = _create_codebuild_construct_stack()
        assert cb.role is not None

    def test_logs_policy_exposed(self):
        _, cb, _ = _create_codebuild_construct_stack()
        assert cb.logs_policy is not None

    def test_vpc_policy_exposed(self):
        _, cb, _ = _create_codebuild_construct_stack()
        assert cb.vpc_policy is not None


class TestCodeBuildProjectEnvVars:
    """Tests for CodeBuild project environment variables."""

    def test_project_with_env_vars(self):
        env_vars = {"MY_VAR": codebuild.BuildEnvironmentVariable(value="my_value")}
        _, _, template = _create_codebuild_construct_stack(env_vars=env_vars)
        template.has_resource_properties(
            "AWS::CodeBuild::Project",
            {
                "Environment": assertions.Match.object_like(
                    {
                        "EnvironmentVariables": assertions.Match.array_with(
                            [assertions.Match.object_like({"Name": "MY_VAR", "Value": "my_value"})]
                        )
                    }
                )
            },
        )
