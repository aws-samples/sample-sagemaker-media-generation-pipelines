"""
Reusable Lambda function construct with security best practices.

This module provides a LambdaTemplate construct that creates Lambda functions
with comprehensive security configurations including VPC integration,
KMS encryption, proper IAM roles, and CloudWatch logging.
"""

from os import path
from pathlib import Path
from aws_cdk import (
    Duration,
    BundlingOptions,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_kms as kms,
    aws_logs as logs,
    RemovalPolicy,
)
from cdk_nag import NagSuppressions
from constructs import Construct


class LambdaTemplate(Construct):
    """
    A reusable Lambda function construct with security best practices.
    
    This construct creates a Lambda function with:
    - VPC integration for network isolation
    - KMS encryption for environment variables
    - Proper IAM role with least privilege access
    - CloudWatch logging with retention policies
    - ARM64 architecture for cost optimization
    - Automatic dependency bundling
    
    The construct follows AWS security best practices including:
    - Network isolation using VPC private subnets
    - Encryption at rest and in transit
    - Least privilege IAM permissions
    - Comprehensive logging and monitoring
    
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
        vpc: ec2.Vpc,
        timeout: int = 600,
        memory_size: int = 2048,
        env_vars: dict = None,
        kms_key: kms.Key = None,
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
        """
        super().__init__(scope, construct_id)

        # Create IAM role for Lambda function with basic execution permissions
        self.function_role = iam.Role(
            self,
            f"{construct_id}-{function_name}-Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaVPCAccessExecutionRole"
                ),
            ],
        )

        # Grant KMS decrypt permissions if KMS key is provided
        if kms_key:
            kms_key.grant_decrypt(self.function_role)

        # Create CloudWatch log group with retention policy
        self.log_group = logs.LogGroup(
            self,
            f"{construct_id}-LogGroup",
            log_group_name=f"/aws/lambda/{function_name}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY, # deleted upon stack destroy
        )

        # Create Lambda function with security configurations
        self.lambda_function = lambda_.Function(
            self,
            f"{construct_id}-{function_name}-Lambda",
            function_name=f"{construct_id}-{function_name}-Lambda",
            runtime=lambda_.Runtime.PYTHON_3_13,
            handler="index.lambda_handler",
            description=description,
            role=self.function_role,
            architecture=lambda_.Architecture.ARM_64,
            memory_size=memory_size,
            code=lambda_.Code.from_asset(
                path=path.join(Path(__file__).parents[0], f"../lambdas/{lambda_path}"),
                bundling=BundlingOptions(
                    image=lambda_.Runtime.PYTHON_3_13.bundling_image,
                    command=[
                        "bash",
                        "-c",
                        "pip install -r requirements.txt -t /asset-output "
                        "&& cp -au . /asset-output",
                    ],
                ),
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ),
            timeout=Duration.seconds(timeout),
            environment_encryption=kms_key,
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
                    "reason": "Managed policy for Lambda Execution anv VPC access",
                }
            ],
        )
