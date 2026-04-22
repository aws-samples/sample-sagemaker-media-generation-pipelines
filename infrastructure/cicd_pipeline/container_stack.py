"""Container build pipeline stack.

Wraps the existing ``create_container_pipeline`` function in its own
CDK Stack to keep the CiCdPipelineStack template under the 1MB limit.
Shares the artifact bucket and KMS key from CiCdPipelineStack.
"""

from aws_cdk import AssetHashType, Stack
from aws_cdk import aws_s3_assets as s3_assets
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from config.config import CicdConfig
from infrastructure.cicd_pipeline.container_pipeline import create_container_pipeline
from infrastructure.cicd_pipeline.stack import CiCdPipelineStack
from infrastructure.security import SecurityStack
from project_constructs.s3 import BucketTemplate


class ContainerPipelineStack(Stack):
    """CDK Stack for the dedicated container build pipeline."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        security_stack: SecurityStack,
        cicd_config: CicdConfig,
        artifact_bucket: BucketTemplate,
        prefix: str = "dev",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        logger.info("Creating ContainerPipelineStack with prefix: {}", prefix)

        account = Stack.of(self).account
        region = Stack.of(self).region

        import aws_cdk.aws_codebuild as codebuild

        env = {
            "AWS_ACCOUNT_ID": codebuild.BuildEnvironmentVariable(value=account),
            "REGION": codebuild.BuildEnvironmentVariable(value=region),
        }

        container_source_asset = s3_assets.Asset(
            self,
            f"{prefix}-container-source-asset",
            path=".",
            exclude=cicd_config.source_excludes,
            asset_hash=CiCdPipelineStack._dir_content_hash(
                ["processing_job", "schema", ".pre-commit-config.yaml", "pyproject.toml", "Makefile"],
                excludes=cicd_config.source_excludes,
            ),
            asset_hash_type=AssetHashType.CUSTOM,
        )

        self.container_pipeline = create_container_pipeline(
            scope=self,
            prefix=prefix,
            cicd_config=cicd_config,
            security_stack=security_stack,
            source_asset=container_source_asset,
            artifact_bucket=artifact_bucket,
            env=env,
        )

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CodeBuild roles use AWS managed policies for admin deploy access",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodeBuild roles need wildcard for VPC, S3, CodeBuild, Lambda, KMS access",
                },
                {"id": "AwsSolutions-CB4", "reason": "CodeBuild projects use KMS-encrypted pipeline artifact bucket"},
            ],
            apply_to_children=True,
        )
