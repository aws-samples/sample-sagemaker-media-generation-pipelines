"""
Reusable CodePipeline construct with pre-built IAM policies.

This module provides a CodePipelineTemplate construct that creates an AWS
CodePipeline resource with KMS encryption, an artifact bucket, pre-built
stages, and a managed policy for CodePipeline orchestration permissions.
"""

from aws_cdk import (
    RemovalPolicy,
)
from aws_cdk import (
    aws_codepipeline as codepipeline,
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
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger


class CodePipelineTemplate(Construct):
    """
    Reusable CodePipeline construct with pre-built IAM policies.

    Attributes:
        pipeline: The AWS CodePipeline resource.
        pipeline_policy: ManagedPolicy for CodePipeline orchestration.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        pipeline_name: str,
        artifact_bucket: s3.IBucket,
        kms_key: kms.Key,
        stages: list[codepipeline.StageProps],
    ) -> None:
        """
        Initialize the CodePipelineTemplate.

        Args:
            scope: The parent construct.
            construct_id: Unique identifier for this construct.
            pipeline_name: Name for the CodePipeline.
            artifact_bucket: S3 bucket for storing pipeline artifacts.
            kms_key: KMS key for pipeline encryption.
            stages: List of pre-built stage definitions for the pipeline.
        """
        super().__init__(scope, construct_id)

        self.pipeline = codepipeline.Pipeline(
            self,
            f"{construct_id}-Pipeline",
            pipeline_name=pipeline_name,
            pipeline_type=codepipeline.PipelineType.V2,
            execution_mode=codepipeline.ExecutionMode.PARALLEL,
            artifact_bucket=artifact_bucket,
            stages=stages,
            restart_execution_on_update=False,
        )
        self.pipeline.apply_removal_policy(RemovalPolicy.DESTROY)

        self.pipeline_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-PipelinePolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "codepipeline:StartPipelineExecution",
                        "codepipeline:StopPipelineExecution",
                        "codepipeline:GetPipelineExecution",
                        "codepipeline:GetPipelineState",
                        "codepipeline:GetPipeline",
                        "codepipeline:ListPipelineExecutions",
                    ],
                    resources=[self.pipeline.pipeline_arn],
                ),
            ],
        )

        logger.info("Created CodePipeline: {}", pipeline_name)

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Pipeline execution role needs wildcard for S3 artifact access and stage actions",
                },
                {
                    "id": "AwsSolutions-CB4",
                    "reason": "Pipeline artifact encryption is handled by the KMS-encrypted artifact bucket",
                },
            ],
            apply_to_children=True,
        )
