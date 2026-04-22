"""
Reusable SQS queue construct with KMS encryption and DLQ.

This module provides an SqsQueueTemplate construct that creates an SQS queue
with KMS encryption, a dead-letter queue, and managed policies for sending
and consuming messages.
"""

from aws_cdk import (
    Duration,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_sqs as sqs,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger


class SqsQueueTemplate(Construct):
    """
    Reusable SQS queue with KMS encryption, DLQ, and managed policies.

    Attributes:
        queue: The main SQS queue resource.
        dlq: The dead-letter queue resource.
        send_policy: ManagedPolicy granting sqs:SendMessage on the main queue.
        consume_policy: ManagedPolicy granting sqs:ReceiveMessage, DeleteMessage, etc.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        queue_name: str,
        kms_key: kms.Key | None = None,
        visibility_timeout_seconds: int = 960,
        max_receive_count: int = 3,
    ) -> None:
        """
        Initialize the SqsQueueTemplate.

        Args:
            scope: The parent construct.
            construct_id: Unique identifier for this construct.
            queue_name: Name for the SQS queue.
            kms_key: KMS key for queue encryption (optional, uses SQS-managed encryption when None).
            visibility_timeout_seconds: Visibility timeout in seconds (default: 960).
                Should exceed Lambda timeout to prevent double-processing.
            max_receive_count: Max receive count before message moves to DLQ (default: 3).
        """
        super().__init__(scope, construct_id)

        # Determine encryption settings
        if kms_key is not None:
            enc = sqs.QueueEncryption.KMS
            enc_key = kms_key
        else:
            enc = sqs.QueueEncryption.SQS_MANAGED
            enc_key = None

        # Create dead-letter queue
        self.dlq = sqs.Queue(
            self,
            f"{construct_id}-DLQ",
            queue_name=f"{queue_name}-dlq",
            encryption=enc,
            encryption_master_key=enc_key,
            retention_period=Duration.days(14),
        )

        # Create main queue with DLQ
        self.queue = sqs.Queue(
            self,
            f"{construct_id}-Queue",
            queue_name=queue_name,
            encryption=enc,
            encryption_master_key=enc_key,
            visibility_timeout=Duration.seconds(visibility_timeout_seconds),
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=max_receive_count,
                queue=self.dlq,
            ),
        )

        # Managed policy for sending messages
        self.send_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-SendPolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["sqs:SendMessage"],
                    resources=[self.queue.queue_arn],
                ),
            ],
        )

        # Managed policy for consuming messages
        self.consume_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-ConsumePolicy",
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=[
                        "sqs:ReceiveMessage",
                        "sqs:DeleteMessage",
                        "sqs:GetQueueAttributes",
                        "sqs:ChangeMessageVisibility",
                    ],
                    resources=[self.queue.queue_arn],
                ),
            ],
        )

        logger.info("Created SQS queue: {}", queue_name)

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-SQS3",
                    "reason": "DLQ is explicitly configured via DeadLetterQueue property",
                },
            ],
            apply_to_children=True,
        )
