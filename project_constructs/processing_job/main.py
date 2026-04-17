"""
Reusable SageMaker Processing Job construct for video upscaling.

This module provides a comprehensive construct for creating SageMaker
processing jobs with Docker containers, Lambda triggers, and proper
IAM permissions. It's specifically designed for video upscaling tasks
but can be adapted for other processing workloads.
"""

import os

from aws_cdk import (
    Stack,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
)

from cdk_nag import NagSuppressions
from constructs import Construct

from config.config import ContainerConfig
from project_constructs.lambda_function import LambdaTemplate
from infrastructure.security import SecurityStack
from project_constructs.processing_job.model import (
    ProcessingDefinition,
    Arguments,
    ProcessingResources,
    ClusterConfig,
    AppSpecification,
    ProcessingInput,
    S3Input,
    ProcessingOutputConfig,
    Output,
    S3Output,
    StoppingCondition,
    NetworkConfig,
    VpcConfig,
)
from project_constructs.s3 import BucketTemplate


class ReusableProcessingJob(Construct):
    """
    A reusable construct for creating SageMaker processing jobs with Lambda triggers.
    
    This construct creates a complete processing job infrastructure including:
    - Docker image asset from local Dockerfile
    - IAM role with necessary permissions for SageMaker, S3, ECR, and VPC access
    - Processing job definition with configurable parameters
    - Optional Lambda trigger for automatic job execution
    - Comprehensive logging and monitoring setup
    
    The construct is designed for video upscaling workloads but can be adapted
    for other processing tasks by modifying the container configuration and
    processing logic.
    
    Key Features:
    - VPC integration for network isolation
    - KMS encryption for data at rest
    - Comprehensive IAM permissions with least privilege
    - CDK Nag compliance for security best practices
    - Configurable instance types and resource allocation
    - Support for multiple input buckets and single output bucket
    
    Attributes:
        image: Docker image asset for the processing container
        role: IAM role for the SageMaker processing job
        definition_model: Pydantic model of the processing job definition
        definition: Dictionary representation of the processing job
        trigger_lambda: Optional Lambda function for triggering jobs
        managed_policy: IAM policy with processing job permissions
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        job_name: str,
        input_buckets: dict[str, BucketTemplate],
        output_bucket: BucketTemplate,
        security_stack: SecurityStack,
        lambda_trigger: bool,
        cfg: ContainerConfig,
        environment=None,
        **kwargs,
    ) -> None:
        """
        Initialize the ReusableProcessingJob construct.
        
        Args:
            scope: The parent construct
            construct_id: Unique identifier for this construct
            job_name: Name for the processing job (used in resource naming)
            input_buckets: Dictionary mapping input names to S3 bucket templates
            output_bucket: S3 bucket template for storing processing results
            security_stack: SecurityStack providing VPC, KMS, and security resources
            lambda_trigger: Whether to create a Lambda trigger for the job
            cfg: Container configuration with instance type, arguments, etc.
            environment: Optional environment variables for the container
            **kwargs: Additional keyword arguments
        """
        super().__init__(scope, construct_id)

        # Initialize environment variables if not provided
        if environment is None:
            environment = {}

        # Store references to AWS account and region
        self.region = Stack.of(self).region
        self.account = Stack.of(self).account
        self.security_stack = security_stack
        self.input_buckets = input_buckets
        self.output_bucket = output_bucket

        # Create Docker image asset from local processing_job directory
        self.image = ecr_assets.DockerImageAsset(
            self,
            f"{construct_id}-ModelImage",
            directory=os.path.join("processing_job"),
            file="Dockerfile",
            build_args={"DOCKER_BUILDKIT": "1"},
        )

        # Create IAM role for SageMaker processing job
        self.role = iam.Role(
            self,
            f"{construct_id}-SageMakerRole",
            role_name=f"{construct_id}-sagemaker-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )

        # Add necessary permissions to the role
        self._add_permissions(construct_id)

        # Create processing inputs for each input bucket
        processing_inputs = [
            ProcessingInput(
                InputName=input_name,
                S3Input=S3Input(
                    S3Uri=f"s3://{bucket.bucket.bucket_name}/",
                    LocalPath=f"/opt/ml/processing/input/{input_name}/",
                    S3DataType="S3Prefix",
                    S3InputMode="File",
                ),
            )
            for input_name, bucket in self.input_buckets.items()
        ]

        LOCAL_OUTPUT_DIR = "/opt/ml/processing/output/"

        # Create processing job definition using Pydantic models
        self.definition_model = ProcessingDefinition(
            Name=job_name,
            Type="Processing",
            Arguments=Arguments(
                ProcessingResources=ProcessingResources(
                    ClusterConfig=ClusterConfig(
                        InstanceCount=cfg.InstanceCount,
                        InstanceType=cfg.InstanceType,
                        VolumeSizeInGB=cfg.VolumeSizeInGB,
                    )
                ),
                AppSpecification=AppSpecification(
                    ImageUri=self.image.image_uri,
                    ContainerEntrypoint=cfg.ContainerEntrypoint,
                    ContainerArguments=cfg.ContainerArguments,
                ),
                RoleArn=self.role.role_arn,
                ProcessingInputs=processing_inputs,
                ProcessingOutputConfig=ProcessingOutputConfig(
                    Outputs=[
                        Output(
                            OutputName="output",
                            S3Output=S3Output(
                                S3Uri=f"s3://{output_bucket.bucket.bucket_name}/output/",
                                LocalPath=LOCAL_OUTPUT_DIR,
                                S3UploadMode="Continuous", #['Continuous', 'EndOfJob']
                            ),
                        )
                    ]
                ),
                Environment=environment | {"LOCAL_OUTPUT_DIR": LOCAL_OUTPUT_DIR},
                StoppingCondition=StoppingCondition(
                    MaxRuntimeInSeconds=86_400
                ),
                NetworkConfig=NetworkConfig(
                    EnableInterContainerTrafficEncryption=True,
                    EnableNetworkIsolation=False,
                    VpcConfig=VpcConfig(
                        SecurityGroupIds=[
                            self.security_stack.security_group.security_group_id
                        ],
                        Subnets=self.security_stack.subnet_ids,
                    ),
                ),
            ),
            DependsOn=[],
        )

        # Initialize definition dictionary
        self.definition: dict = {}
        self.update_definition()

        # Create Lambda trigger if requested
        if lambda_trigger:
            self._create_lambda_trigger(construct_id, job_name)

    def update_definition(self) -> None:
        """
        Update the dictionary representation of the processing job definition.
        
        This method converts the Pydantic model to a dictionary that can be
        used by the Lambda trigger function to create SageMaker processing jobs.
        """
        self.definition = self.definition_model.model_dump()

    def _create_lambda_trigger(self, construct_id: str, job_name: str) -> None:
        """
        Create Lambda function for triggering processing jobs.
        
        Args:
            construct_id: Construct identifier for resource naming
            job_name: Name of the processing job
        """
        # Create Lambda function for triggering processing jobs
        self.trigger_lambda = LambdaTemplate(
            self, f"{construct_id}-Trigger",
            function_name=f"{job_name}-trigger",
            lambda_path="trigger_processing_job",
            description=f"Lambda used to trigger {job_name} processing job",
            vpc=self.security_stack.vpc,
            env_vars={
                "PROCESSING_JOB_DEFINITION": self.definition_model.model_dump_json(),
                "JOB_NAME": job_name,
            }
        )

        # Create IAM policy for Lambda to trigger SageMaker jobs
        self._trigger_policy = iam.ManagedPolicy(
            self, f"{construct_id}-TriggerLambdaPolicy",
            managed_policy_name=f"{self.node.id}-TriggerLambdaPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["sagemaker:CreateProcessingJob"],
                    resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:processing-job/*"]
                ),
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[self.role.role_arn]
                )
            ]
        )

        # Attach policy to Lambda role
        self.trigger_lambda.function_role.add_managed_policy(self._trigger_policy)

        # Suppress CDK Nag warnings for wildcard permissions
        NagSuppressions.add_resource_suppressions(
            self._trigger_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Lambda function needs to create SageMaker processing jobs with dynamic names",
                    "appliesTo": [f"Resource::arn:aws:sagemaker:{self.region}:{self.account}:processing-job/*"]
                }
            ],
        )

    def _add_permissions(self, construct_id: str) -> None:
        """
        Add comprehensive IAM permissions to the processing job role.
        
        This method creates and attaches IAM policies for:
        - SageMaker processing job operations
        - CloudWatch metrics and logging
        - EC2 network interface management for VPC
        - ECR image access
        - S3 bucket access through bucket-specific policies
        - KMS key access for encryption
        
        Args:
            construct_id: Construct identifier for resource naming
        """
        # Create comprehensive IAM policy for processing job
        self.managed_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-ProcessingJobPolicy",
            managed_policy_name=f"{self.node.id}-ProcessingJobPolicy",
            statements=[
                # SageMaker processing job permissions
                iam.PolicyStatement(
                    actions=[
                        "sagemaker:CreateProcessingJob",
                        "sagemaker:DescribeProcessingJob",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{Stack.of(self).region}:{Stack.of(self).account}:processing-job/*"
                    ],
                ),
                # CloudWatch metrics permissions
                iam.PolicyStatement(
                    actions=["cloudwatch:GetMetricData", "cloudwatch:ListMetrics"],
                    resources=["*"], # required for cloudwatch
                ),
                # EC2 VPC permissions for network interface management
                iam.PolicyStatement(
                    actions=[
                        "ec2:CreateNetworkInterface",
                        "ec2:CreateNetworkInterfacePermission",
                        "ec2:DeleteNetworkInterface",
                        "ec2:DeleteNetworkInterfacePermission",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DescribeVpcs",
                        "ec2:DescribeDhcpOptions",
                        "ec2:DescribeSubnets",
                        "ec2:DescribeSecurityGroups",
                    ],
                    resources=["*"], # required for networking
                ),
                # CloudWatch Logs permissions
                iam.PolicyStatement(
                    actions=[
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:CreateLogGroup",
                        "logs:DescribeLogStreams",
                    ],
                    resources=[
                        f"arn:aws:logs:{Stack.of(self).region}:{Stack.of(self).account}:log-group:/aws/sagemaker/ProcessingJobs:*",
                        f"arn:aws:logs:{Stack.of(self).region}:{Stack.of(self).account}:log-group:/aws/sagemaker/ProcessingJobs:*:log-stream:*",
                    ],
                ),
                # CloudWatch metrics publishing permissions
                iam.PolicyStatement(
                    actions=["cloudwatch:PutMetricData"],
                    resources=["*"], # required for cloudwatch
                    conditions={
                        "StringEquals": {
                            "cloudwatch:namespace": "/aws/sagemaker/ProcessingJobs"
                        }
                    },
                ),
                # ECR permissions for Docker image access
                iam.PolicyStatement(
                    actions=[
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "ecr:BatchCheckLayerAvailability",
                    ],
                    resources=[self.image.repository.repository_arn],
                ),
                # ECR authentication permissions
                iam.PolicyStatement(
                    actions=["ecr:GetAuthorizationToken"], 
                    resources=["*"]
                ),
            ],
        )

        # Attach S3 bucket policies for input and output access
        for _, bucket in self.input_buckets.items():
            self.role.add_managed_policy(bucket.read_policy)
        self.role.add_managed_policy(self.output_bucket.write_policy)
        
        # Attach KMS and processing job policies
        self.role.add_managed_policy(self.security_stack.kms_key_policy)
        self.role.add_managed_policy(self.managed_policy)

        # Suppress CDK Nag warnings for necessary wildcard permissions
        NagSuppressions.add_resource_suppressions(
            self.managed_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "SageMaker Processing Job role requires broad permissions to access AWS services",
                    "appliesTo": [
                        "Resource::*",
                        f"Resource::arn:aws:sagemaker:{Stack.of(self).region}:{Stack.of(self).account}:processing-job/*",
                        f"Resource::arn:aws:logs:{Stack.of(self).region}:{Stack.of(self).account}:log-group:/aws/sagemaker/ProcessingJobs:*",
                        f"Resource::arn:aws:logs:{Stack.of(self).region}:{Stack.of(self).account}:log-group:/aws/sagemaker/ProcessingJobs:*:log-stream:*",
                    ],
                }
            ],
        )
