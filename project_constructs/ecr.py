"""
Reusable ECR repository construct with security defaults.

This module provides an EcrRepository construct that creates an ECR
repository with image scanning, lifecycle rules, KMS encryption,
and exposes managed policies for push, pull, and auth operations.
"""

from aws_cdk import (
    RemovalPolicy,
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
from cdk_nag import NagSuppressions
from constructs import Construct


class EcrRepository(Construct):
    """
    Reusable ECR repository with managed policies.

    Attributes:
        repository: The ECR repository resource.
        ecr_push_policy: ManagedPolicy for push operations.
        ecr_pull_policy: ManagedPolicy for pull operations.
        ecr_auth_policy: ManagedPolicy for ECR authentication.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        repository_name: str,
        kms_key: kms.Key,
    ) -> None:
        super().__init__(scope, construct_id)

        # Policy names can't contain '/', so flatten for naming
        policy_safe_name = repository_name.replace("/", "-")

        self.repository = ecr.Repository(
            self,
            f"{construct_id}-Repo",
            repository_name=repository_name,
            encryption_key=kms_key,
            image_scan_on_push=True,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
        )
        self.repository.add_lifecycle_rule(max_image_count=10)

        # Push policy
        self.ecr_push_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-EcrPushPolicy",
            managed_policy_name=f"{policy_safe_name}-ecr-push-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "ecr:PutImage",
                        "ecr:InitiateLayerUpload",
                        "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload",
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:DescribeImages",
                        "ecr:BatchGetImage",
                    ],
                    resources=[self.repository.repository_arn],
                )
            ],
        )

        # Pull policy
        self.ecr_pull_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-EcrPullPolicy",
            managed_policy_name=f"{policy_safe_name}-ecr-pull-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage",
                        "ecr:BatchCheckLayerAvailability",
                    ],
                    resources=[self.repository.repository_arn],
                )
            ],
        )

        # Auth policy (requires Resource: * — API limitation)
        self.ecr_auth_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-EcrAuthPolicy",
            managed_policy_name=f"{policy_safe_name}-ecr-auth-policy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["ecr:GetAuthorizationToken"],
                    resources=["*"],
                )
            ],
        )
        NagSuppressions.add_resource_suppressions(
            self.ecr_auth_policy,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "ecr:GetAuthorizationToken does not support resource-level permissions",
                }
            ],
        )
