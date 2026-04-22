"""
Reusable Lambda function construct with security best practices.

This module provides a LambdaTemplate construct that creates Lambda functions
with comprehensive security configurations including VPC integration,
KMS encryption, proper IAM roles, and CloudWatch logging.

Supports two deployment modes:
- Zip deployment: Code.from_asset() with Docker bundling (default)
- Container image: DockerImageFunction from ECR (when ecr_image_uri is provided)
"""

import subprocess
from os import path
from pathlib import Path

import jsii
from aws_cdk import (
    BundlingOptions,
    Duration,
    ILocalBundling,
    RemovalPolicy,
    Size,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_ecr as ecr,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_logs as logs,
)
from cdk_nag import NagSuppressions
from constructs import Construct


@jsii.implements(ILocalBundling)
class PipLocalBundling:
    """Bundle Lambda code locally using pip, avoiding Docker dependency."""

    def __init__(self, source_path: str) -> None:
        self._source_path = source_path

    def try_bundle(self, output_dir: str, *, image=None, **kwargs) -> bool:
        """Install requirements and copy source to output_dir."""
        req_file = path.join(self._source_path, "requirements.txt")
        if path.exists(req_file):
            subprocess.check_call(
                ["pip", "install", "-r", req_file, "-t", output_dir, "-q"],
            )
        # Copy source files
        subprocess.check_call(
            ["bash", "-c", f"cp -a {self._source_path}/. {output_dir}/"],
        )
        # Copy schema package for column name constants
        schema_dir = path.join(Path(__file__).parents[0], "../schema")
        if path.isdir(schema_dir):
            subprocess.check_call(
                ["bash", "-c", f"cp -a {schema_dir} {output_dir}/schema"],
            )
        return True


class LambdaTemplate(Construct):
    """
    A reusable Lambda function construct with security best practices.

    Supports two deployment modes:
    - Zip: uses Code.from_asset() with Docker bundling (default, when ecr_image_uri is None)
    - Container image: uses DockerImageFunction from ECR (when ecr_image_uri is provided)

    Attributes:
        function_role: IAM role for the Lambda function
        log_group: CloudWatch log group for function logs
        lambda_function: The Lambda function resource
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        function_name: str,
        lambda_path: str,
        description: str,
        vpc: ec2.Vpc | None = None,
        timeout: int = 600,
        memory_size: int = 2048,
        ephemeral_storage_size: int | None = None,
        env_vars: dict[str, str] | None = None,
        kms_key: kms.Key | None = None,
        filesystem: lambda_.FileSystem | None = None,
        ecr_image_uri: str | None = None,
        ecr_repository: ecr.IRepository | None = None,
        ecr_pull_policy: iam.ManagedPolicy | None = None,
        ecr_auth_policy: iam.ManagedPolicy | None = None,
        reserved_concurrent_executions: int | None = None,
    ) -> None:
        """
        Initialize the LambdaTemplate with security configurations.

        Args:
            scope: The parent construct
            construct_id: Unique identifier for this construct
            function_name: Name for the Lambda function
            lambda_path: Path to the Lambda function code directory
            description: Description for the Lambda function
            vpc: VPC for network isolation
            timeout: Function timeout in seconds (default: 600)
            memory_size: Memory allocation in MB (default: 2048)
            env_vars: Dictionary of environment variables (optional)
            kms_key: KMS key for environment variable encryption (optional)
            filesystem: EFS filesystem configuration for Lambda (optional)
            ecr_image_uri: ECR image URI for container-image Lambda (optional).
                When provided, creates a DockerImageFunction instead of a zip Function.
            ecr_repository: ECR repository object for container-image Lambda (optional).
                Required when ecr_image_uri is provided.
            ephemeral_storage_size: Ephemeral /tmp storage in MB (512-10240, optional)
            ecr_pull_policy: ManagedPolicy for ECR pull access (optional, used with ecr_image_uri)
            ecr_auth_policy: ManagedPolicy for ECR auth access (optional, used with ecr_image_uri)
        """
        super().__init__(scope, construct_id)

        # Create IAM role for Lambda function with basic execution permissions
        managed_policies = [
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole"),
        ]
        if vpc is not None:
            managed_policies.append(
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole")
            )
        self.function_role = iam.Role(
            self,
            f"{function_name}-Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=managed_policies,
        )

        # Attach ECR policies if provided
        if ecr_pull_policy is not None:
            self.function_role.add_managed_policy(ecr_pull_policy)
        if ecr_auth_policy is not None:
            self.function_role.add_managed_policy(ecr_auth_policy)

        # Grant KMS decrypt permissions if KMS key is provided
        if kms_key:
            kms_key.grant_decrypt(self.function_role)

        # Create CloudWatch log group with retention policy
        self.log_group = logs.LogGroup(
            self,
            f"{function_name}-LogGroup",
            log_group_name=f"/aws/lambda/{function_name}-Lambda",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Build optional keyword arguments shared by both function types
        lambda_kwargs: dict = {}
        if filesystem is not None:
            lambda_kwargs["filesystem"] = filesystem
        if ephemeral_storage_size is not None:
            lambda_kwargs["ephemeral_storage_size"] = Size.mebibytes(ephemeral_storage_size)
        if reserved_concurrent_executions is not None:
            lambda_kwargs["reserved_concurrent_executions"] = reserved_concurrent_executions

        # VPC kwargs — only included when vpc is provided
        vpc_kwargs: dict = {}
        if vpc is not None:
            vpc_kwargs["vpc"] = vpc
            vpc_kwargs["vpc_subnets"] = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)

        if ecr_image_uri is not None and ecr_repository is not None:
            # Container image mode: DockerImageFunction from ECR
            # Use X86_64 because CodeBuild builds x86_64 images by default
            self.lambda_function: lambda_.Function | lambda_.DockerImageFunction = lambda_.DockerImageFunction(
                self,
                f"{function_name}-Lambda",
                function_name=f"{function_name}-Lambda",
                code=lambda_.DockerImageCode.from_ecr(
                    repository=ecr_repository,
                    tag_or_digest="latest",
                ),
                description=description,
                role=self.function_role,
                log_group=self.log_group,
                architecture=lambda_.Architecture.X86_64,
                memory_size=memory_size,
                timeout=Duration.seconds(timeout),
                environment_encryption=kms_key,
                **vpc_kwargs,
                **lambda_kwargs,
            )
        else:
            # Zip deployment mode: local pip install, Docker fallback
            lambda_source = path.join(Path(__file__).parents[0], f"../lambdas/{lambda_path}")
            self.lambda_function = lambda_.Function(
                self,
                f"{function_name}-Lambda",
                function_name=f"{function_name}-Lambda",
                runtime=lambda_.Runtime.PYTHON_3_13,
                handler="index.lambda_handler",
                description=description,
                role=self.function_role,
                log_group=self.log_group,
                architecture=lambda_.Architecture.ARM_64,
                memory_size=memory_size,
                code=lambda_.Code.from_asset(
                    path=lambda_source,
                    bundling=BundlingOptions(
                        image=lambda_.Runtime.PYTHON_3_13.bundling_image,
                        command=[
                            "bash",
                            "-c",
                            "pip install -r requirements.txt -t /asset-output && cp -au . /asset-output",
                        ],
                        local=PipLocalBundling(lambda_source),
                    ),
                ),
                timeout=Duration.seconds(timeout),
                environment_encryption=kms_key,
                **vpc_kwargs,
                **lambda_kwargs,
            )

        # Add environment variables if provided
        if env_vars:
            for env_key, env_value in env_vars.items():
                self.lambda_function.add_environment(key=env_key, value=env_value)

        # Suppress CDK Nag warnings for managed policies (required for Lambda execution)
        NagSuppressions.add_resource_suppressions(
            self.function_role,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "Managed policy for Lambda Execution and VPC access",
                }
            ],
        )

        # Suppress L1 for zip-deployed Lambdas — PYTHON_3_14 not yet available in aws-cdk-lib
        NagSuppressions.add_resource_suppressions(
            self.lambda_function,
            suppressions=[
                {
                    "id": "AwsSolutions-L1",
                    "reason": "PYTHON_3_14 not available in aws-cdk-lib 2.250; using latest supported runtime (3.13)",
                }
            ],
        )
