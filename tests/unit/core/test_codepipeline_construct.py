"""
Unit tests for the CodePipelineTemplate construct.

Tests verify CodePipeline resource creation with KMS encryption,
artifact bucket wiring, stage assembly, and exposed managed policies
for pipeline orchestration.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import (
    assertions,
)
from aws_cdk import (
    aws_codepipeline as codepipeline,
)
from aws_cdk import (
    aws_codepipeline_actions as codepipeline_actions,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_s3 as s3,
)

from project_constructs.codepipeline import CodePipelineTemplate

pytestmark = pytest.mark.core


def _create_codepipeline_construct_stack() -> tuple[cdk.Stack, CodePipelineTemplate, assertions.Template]:
    """Helper to create a stack with a CodePipelineTemplate for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)
    artifact_bucket = s3.Bucket(
        stack,
        "ArtifactBucket",
        encryption=s3.BucketEncryption.KMS,
        encryption_key=kms_key,
    )

    source_output = codepipeline.Artifact("SourceOutput")

    stages = [
        codepipeline.StageProps(
            stage_name="Source",
            actions=[
                codepipeline_actions.S3SourceAction(
                    action_name="S3Source",
                    bucket=artifact_bucket,
                    bucket_key="source.zip",
                    output=source_output,
                ),
            ],
        ),
        codepipeline.StageProps(
            stage_name="Approve",
            actions=[
                codepipeline_actions.ManualApprovalAction(
                    action_name="ManualApproval",
                ),
            ],
        ),
    ]

    cp = CodePipelineTemplate(
        stack,
        "TestCP",
        pipeline_name="test-pipeline",
        artifact_bucket=artifact_bucket,
        kms_key=kms_key,
        stages=stages,
    )
    template = assertions.Template.from_stack(stack)
    return stack, cp, template


class TestCodePipelineTemplateResources:
    """Tests for CodePipelineTemplate resource creation."""

    def test_creates_codepipeline(self):
        _, _, template = _create_codepipeline_construct_stack()
        template.resource_count_is("AWS::CodePipeline::Pipeline", 1)

    def test_pipeline_has_artifact_bucket(self):
        _, _, template = _create_codepipeline_construct_stack()
        template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {
                "ArtifactStore": assertions.Match.object_like(
                    {
                        "Location": assertions.Match.any_value(),
                        "Type": "S3",
                    }
                )
            },
        )

    def test_pipeline_has_encryption_key(self):
        _, _, template = _create_codepipeline_construct_stack()
        template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {
                "ArtifactStore": assertions.Match.object_like(
                    {"EncryptionKey": assertions.Match.object_like({"Type": "KMS"})}
                )
            },
        )

    def test_pipeline_has_two_stages(self):
        _, _, template = _create_codepipeline_construct_stack()
        template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {
                "Stages": assertions.Match.array_with(
                    [
                        assertions.Match.object_like({"Name": "Source"}),
                        assertions.Match.object_like({"Name": "Approve"}),
                    ]
                )
            },
        )

    def test_pipeline_has_destroy_removal_policy(self):
        _, _, template = _create_codepipeline_construct_stack()
        template.has_resource(
            "AWS::CodePipeline::Pipeline",
            {
                "UpdateReplacePolicy": "Delete",
                "DeletionPolicy": "Delete",
            },
        )

    def test_pipeline_does_not_restart_on_update(self):
        _, _, template = _create_codepipeline_construct_stack()
        template.has_resource_properties(
            "AWS::CodePipeline::Pipeline",
            {"RestartExecutionOnUpdate": False},
        )


class TestCodePipelineTemplateIAM:
    """Tests for CodePipelineTemplate IAM configuration."""

    def test_creates_pipeline_managed_policy(self):
        _, _, template = _create_codepipeline_construct_stack()
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": assertions.Match.array_with(
                                            [
                                                "codepipeline:StartPipelineExecution",
                                                "codepipeline:GetPipelineState",
                                            ]
                                        ),
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        )
                    }
                )
            },
        )


class TestCodePipelineTemplateAttributes:
    """Tests for CodePipelineTemplate exposed attributes."""

    def test_pipeline_exposed(self):
        _, cp, _ = _create_codepipeline_construct_stack()
        assert cp.pipeline is not None

    def test_pipeline_policy_exposed(self):
        _, cp, _ = _create_codepipeline_construct_stack()
        assert cp.pipeline_policy is not None
