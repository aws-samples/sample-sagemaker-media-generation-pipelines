"""
Reusable CodeBuild project construct with VPC integration.

This module provides a CodeBuildProject construct that creates a CodeBuild
project with VPC networking, KMS encryption, CloudWatch logging, and
exposes managed policies for logs and VPC access.
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
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
    aws_kms as kms,
)
from aws_cdk import (
    aws_logs as logs,
)
from cdk_nag import NagSuppressions
from constructs import Construct


class CodeBuildProject(Construct):
    """
    Reusable CodeBuild project with managed policies.

    Attributes:
        project: The CodeBuild project resource.
        role: The CodeBuild service role.
        logs_policy: ManagedPolicy for CloudWatch Logs.
        vpc_policy: ManagedPolicy for VPC networking.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        source: codebuild.ISource,
        vpc: ec2.Vpc,
        subnet_selection: ec2.SubnetSelection,
        security_group: ec2.SecurityGroup,
        kms_key: kms.Key,
        environment_variables: dict[str, codebuild.BuildEnvironmentVariable] | None = None,
        compute_type: codebuild.ComputeType = codebuild.ComputeType.SMALL,
        timeout_minutes: int = 60,
    ) -> None:
        super().__init__(scope, construct_id)

        self.role = iam.Role(
            self,
            f"{construct_id}-Role",
            assumed_by=iam.ServicePrincipal("codebuild.amazonaws.com"),
        )

        log_group = logs.LogGroup(
            self,
            f"{construct_id}-LogGroup",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
            encryption_key=kms_key,
        )

        self.project = codebuild.Project(
            self,
            f"{construct_id}-Project",
            source=source,
            role=self.role,
            vpc=vpc,
            subnet_selection=subnet_selection,
            security_groups=[security_group],
            encryption_key=kms_key,
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0,
                compute_type=compute_type,
                privileged=True,
            ),
            environment_variables=environment_variables or {},
            build_spec=codebuild.BuildSpec.from_source_filename("buildspec.yml"),
            timeout=Duration.minutes(timeout_minutes),
            logging=codebuild.LoggingOptions(
                cloud_watch=codebuild.CloudWatchLoggingOptions(
                    log_group=log_group,
                    enabled=True,
                )
            ),
        )

        # Logs policy
        self.logs_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-LogsPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                    ],
                    resources=[log_group.log_group_arn, f"{log_group.log_group_arn}:*"],
                )
            ],
        )

        # VPC policy
        self.vpc_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-VpcPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "ec2:CreateNetworkInterface",
                        "ec2:DescribeNetworkInterfaces",
                        "ec2:DeleteNetworkInterface",
                        "ec2:DescribeSubnets",
                        "ec2:DescribeSecurityGroups",
                        "ec2:DescribeDhcpOptions",
                        "ec2:DescribeVpcs",
                        "ec2:CreateNetworkInterfacePermission",
                    ],
                    resources=["*"],
                )
            ],
        )

        NagSuppressions.add_resource_suppressions(
            self.vpc_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ec2:Describe* and CreateNetworkInterface actions require Resource: *",
                }
            ],
        )
        NagSuppressions.add_resource_suppressions(
            self.project,
            suppressions=[
                {
                    "id": "AwsSolutions-CB4",
                    "reason": "Using S3 managed encryption for CDK asset bucket source",
                }
            ],
        )
        NagSuppressions.add_resource_suppressions(
            self.role,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "CodeBuild role needs wildcard for VPC and S3 asset access",
                }
            ],
        )
