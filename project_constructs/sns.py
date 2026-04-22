"""
Reusable SNS topic construct with KMS encryption and SageMaker publish access.

This module provides an SnsTopicTemplate construct that creates an SNS topic
with KMS encryption, a resource policy allowing SageMaker to publish, and
a managed policy for granting publish access to IAM principals.
"""

from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_sns as sns,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger


class SnsTopicTemplate(Construct):
    """
    Reusable SNS topic with KMS encryption and managed publish policy.

    Attributes:
        topic: The SNS topic resource.
        publish_policy: ManagedPolicy granting sns:Publish on this topic.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        topic_name: str,
        kms_key: kms.Key,
        service_principals: list[str] | None = None,
    ) -> None:
        """
        Initialize the SnsTopicTemplate.

        Args:
            scope: The parent construct.
            construct_id: Unique identifier for this construct.
            topic_name: Name for the SNS topic.
            kms_key: KMS key for topic encryption.
            service_principals: AWS service principals allowed to publish
                (e.g. ["sagemaker.amazonaws.com"]). Defaults to empty.
        """
        super().__init__(scope, construct_id)

        self.topic = sns.Topic(
            self,
            f"{construct_id}-Topic",
            topic_name=topic_name,
            master_key=kms_key,
        )

        # Allow specified service principals to publish
        for principal in service_principals or []:
            self.topic.add_to_resource_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[iam.ServicePrincipal(principal)],
                    actions=["sns:Publish"],
                    resources=[self.topic.topic_arn],
                )
            )

        # Managed policy for granting publish access to IAM roles
        self.publish_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-PublishPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["sns:Publish"],
                    resources=[self.topic.topic_arn],
                ),
            ],
        )

        logger.info("Created SNS topic: {}", topic_name)

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-SNS2",
                    "reason": "Topic uses KMS encryption via master_key parameter",
                },
            ],
            apply_to_children=True,
        )
