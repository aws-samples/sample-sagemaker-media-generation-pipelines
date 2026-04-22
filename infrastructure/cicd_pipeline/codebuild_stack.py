"""
CodeBuild infrastructure stack.

This module defines the CodeBuildStack which creates per-step ECR
repositories and CodeBuild projects for building container images.
Container builds are triggered by the CI/CD pipeline's ContainerBuild
stage rather than a custom resource.
"""

from aws_cdk import (
    Stack,
)
from aws_cdk import (
    aws_codebuild as codebuild,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_s3_assets as s3_assets,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from infrastructure.security import SecurityStack
from project_constructs.codebuild import CodeBuildProject
from project_constructs.ecr import EcrRepository


class CodeBuildStack(Stack):
    """
    CDK Stack that creates per-step ECR repositories and CodeBuild projects.

    Supports two categories of builds:
    - Processing job steps: source from processing_job/{step_name}/
    - Lambda functions: source from lambdas/{lambda_name}/

    Attributes:
        ecr_repositories: ECR repo per step, keyed by step name.
        codebuild_projects: CodeBuild project per step, keyed by step name.
        lambda_ecr_repositories: ECR repo per Lambda, keyed by Lambda name.
        lambda_codebuild_projects: CodeBuild project per Lambda, keyed by Lambda name.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        security_stack: SecurityStack,
        step_names: list[str],
        lambda_names: list[str] | None = None,
        prefix: str = "dev",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if lambda_names is None:
            lambda_names = []

        logger.info("Creating CodeBuildStack with prefix: {}, steps: {}, lambdas: {}", prefix, step_names, lambda_names)

        self.ecr_repositories: dict[str, EcrRepository] = {}
        self.codebuild_projects: dict[str, CodeBuildProject] = {}
        self.lambda_ecr_repositories: dict[str, EcrRepository] = {}
        self.lambda_codebuild_projects: dict[str, CodeBuildProject] = {}

        # --- Processing job steps ---
        for step_name in step_names:
            self._create_ecr_and_codebuild(
                security_stack=security_stack,
                name=step_name,
                source_path="processing_job",
                prefix=prefix,
                repo_dict=self.ecr_repositories,
                project_dict=self.codebuild_projects,
                category="processing",
                extra_env={
                    "STEP_NAME": codebuild.BuildEnvironmentVariable(
                        value=step_name,
                    ),
                },
            )

        # --- Lambda functions ---
        for lambda_name in lambda_names:
            self._create_ecr_and_codebuild(
                security_stack=security_stack,
                name=lambda_name,
                source_path=f"lambdas/{lambda_name}",
                prefix=prefix,
                repo_dict=self.lambda_ecr_repositories,
                project_dict=self.lambda_codebuild_projects,
                category="lambda",
            )

        # --- Cross-account ECR pull policy for SageMaker DLC base images ---
        # Some processing steps (e.g. vbench) use SageMaker DLC images from
        # the AWS-managed account 763104351884. All processing CodeBuild roles
        # need pull access to that registry.
        sagemaker_dlc_account = "763104351884"
        sagemaker_dlc_ecr_pull_policy = iam.ManagedPolicy(
            self,
            f"{prefix}-SageMakerDlcEcrPullPolicy",
            managed_policy_name=f"{prefix}-sagemaker-dlc-ecr-pull-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "ecr:BatchCheckLayerAvailability",
                    ],
                    resources=[
                        f"arn:aws:ecr:{Stack.of(self).region}:{sagemaker_dlc_account}:repository/*",
                    ],
                )
            ],
        )
        for cb_project in self.codebuild_projects.values():
            cb_project.role.add_managed_policy(sagemaker_dlc_ecr_pull_policy)

        # CDK Nag suppressions
        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "CodeBuild roles use AWS managed policies",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodeBuild roles need wildcard for VPC, S3 asset, and ECR auth access",
                },
            ],
            apply_to_children=True,
        )

    def _create_ecr_and_codebuild(
        self,
        security_stack: SecurityStack,
        name: str,
        source_path: str,
        prefix: str,
        repo_dict: dict[str, EcrRepository],
        project_dict: dict[str, CodeBuildProject],
        category: str = "processing",
        extra_env: dict[str, codebuild.BuildEnvironmentVariable] | None = None,
    ) -> None:
        """Create an ECR repo and CodeBuild project for a given source directory.

        Naming convention:
            ECR repo:  {prefix}/{category}/{clean_name}   e.g. dev/processing/step-1
            Construct: {prefix}-{category}-{clean_name}-*  e.g. dev-processing-step-1-EcrRepo
        """
        clean_name = name.replace("_", "-")
        repo_name = f"{prefix}/{category}/{clean_name}"
        construct_tag = f"{prefix}-{category}-{clean_name}"

        ecr_repo = EcrRepository(
            self,
            f"{construct_tag}-EcrRepo",
            repository_name=repo_name,
            kms_key=security_stack.kms_key,
        )
        repo_dict[name] = ecr_repo
        logger.info("Created ECR repository: {}", repo_name)

        asset = s3_assets.Asset(
            self,
            f"{construct_tag}-Source",
            path=source_path,
        )

        cb_source = codebuild.Source.s3(
            bucket=asset.bucket,
            path=asset.s3_object_key,
        )

        # Use larger compute for processing (Docker builds), small for Lambda
        compute = codebuild.ComputeType.LARGE if category == "processing" else codebuild.ComputeType.SMALL

        env_vars = {
            "ECR_REPO_URI": codebuild.BuildEnvironmentVariable(
                value=ecr_repo.repository.repository_uri,
            ),
            "AWS_DEFAULT_REGION": codebuild.BuildEnvironmentVariable(
                value=Stack.of(self).region,
            ),
        }
        if extra_env:
            env_vars.update(extra_env)

        cb_project = CodeBuildProject(
            self,
            f"{construct_tag}-CodeBuild",
            source=cb_source,
            vpc=security_stack.vpc,
            subnet_selection=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_group=security_stack.security_group,
            kms_key=security_stack.kms_key,
            compute_type=compute,
            environment_variables=env_vars,
        )
        project_dict[name] = cb_project
        logger.info("Created CodeBuild project: {}", construct_tag)

        cb_project.role.add_managed_policy(ecr_repo.ecr_push_policy)
        cb_project.role.add_managed_policy(ecr_repo.ecr_auth_policy)
        cb_project.role.add_managed_policy(security_stack.kms_key_policy)
        cb_project.role.add_managed_policy(cb_project.logs_policy)
        cb_project.role.add_managed_policy(cb_project.vpc_policy)
        asset.grant_read(cb_project.role)
        logger.debug("Attached IAM policies for: {}", construct_tag)
