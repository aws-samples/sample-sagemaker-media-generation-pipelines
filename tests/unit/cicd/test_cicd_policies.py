"""Unit tests for infrastructure/cicd_pipeline/policies.py.

Verifies each IAM policy factory returns a ManagedPolicy with the
correct statement structure (Effect, Action, Resource).
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infrastructure.cicd_pipeline.policies import (
    container_build_policy,
    model_download_policy,
    trigger_pipeline_policy,
    upload_input_policy,
)

pytestmark = pytest.mark.cicd


def _synth_policy(policy_fn, **kwargs) -> assertions.Template:
    """Synthesize a stack containing a single policy and return its Template."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack")
    policy_fn(stack, "test", **kwargs)
    return assertions.Template.from_stack(stack)


class TestContainerBuildPolicy:
    @pytest.fixture(scope="class")
    def template(self):
        return _synth_policy(container_build_policy)

    def test_creates_managed_policy(self, template) -> None:
        template.resource_count_is("AWS::IAM::ManagedPolicy", 1)

    def test_allows_codebuild_actions(self, template) -> None:
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Effect": "Allow",
                                    "Action": assertions.Match.array_with(
                                        [
                                            "codebuild:ListProjects",
                                            "codebuild:StartBuild",
                                            "codebuild:BatchGetBuilds",
                                        ]
                                    ),
                                }
                            )
                        ]
                    )
                }
            },
        )


class TestModelDownloadPolicy:
    @pytest.fixture(scope="class")
    def template(self):
        return _synth_policy(
            model_download_policy,
            bucket_arn="arn:aws:s3:::my-bucket",
        )

    def test_creates_managed_policy(self, template) -> None:
        template.resource_count_is("AWS::IAM::ManagedPolicy", 1)

    def test_allows_s3_actions(self, template) -> None:
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Effect": "Allow",
                                    "Action": assertions.Match.array_with(["s3:GetObject", "s3:PutObject"]),
                                }
                            )
                        ]
                    )
                }
            },
        )

    def test_allows_kms_actions(self, template) -> None:
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Effect": "Allow",
                                    "Action": assertions.Match.array_with(["kms:GenerateDataKey", "kms:Decrypt"]),
                                }
                            )
                        ]
                    )
                }
            },
        )


class TestUploadInputPolicy:
    @pytest.fixture(scope="class")
    def template(self):
        return _synth_policy(
            upload_input_policy,
            bucket_arn="arn:aws:s3:::input-bucket",
        )

    def test_creates_managed_policy(self, template) -> None:
        template.resource_count_is("AWS::IAM::ManagedPolicy", 1)

    def test_allows_s3_put(self, template) -> None:
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Effect": "Allow",
                                    "Action": assertions.Match.array_with(["s3:PutObject"]),
                                }
                            )
                        ]
                    )
                }
            },
        )


class TestTriggerPipelinePolicy:
    @pytest.fixture(scope="class")
    def template(self):
        return _synth_policy(
            trigger_pipeline_policy,
            region="us-east-1",
            account="123456789012",
            cfg_prefix="dev",
        )

    def test_creates_managed_policy(self, template) -> None:
        template.resource_count_is("AWS::IAM::ManagedPolicy", 1)

    def test_allows_lambda_invoke(self, template) -> None:
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Effect": "Allow",
                                    "Action": "lambda:InvokeFunction",
                                }
                            )
                        ]
                    )
                }
            },
        )

    def test_allows_cfn_describe_stacks(self, template) -> None:
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with(
                        [
                            assertions.Match.object_like(
                                {
                                    "Effect": "Allow",
                                    "Action": "cloudformation:DescribeStacks",
                                }
                            )
                        ]
                    )
                }
            },
        )
