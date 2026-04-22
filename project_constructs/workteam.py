"""
Reusable SageMaker private workteam construct.

This module provides a WorkteamTemplate construct that creates a SageMaker
CfnWorkteam backed by a Cognito User Pool group with SNS notifications.
"""

from aws_cdk import (
    Stack,
)
from aws_cdk import (
    aws_sagemaker as sagemaker,
)
from constructs import Construct
from loguru import logger

from project_constructs.cognito import CognitoWorkforceTemplate
from project_constructs.sns import SnsTopicTemplate


class WorkteamTemplate(Construct):
    """
    Reusable SageMaker private workteam construct.

    Attributes:
        workteam: The CfnWorkteam resource.
        workteam_arn: ARN of the workteam.
        workteam_name: Name of the workteam.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        workteam_name: str,
        cognito: CognitoWorkforceTemplate,
        notification_topic: SnsTopicTemplate,
    ) -> None:
        """
        Initialize the WorkteamTemplate.

        Args:
            scope: The parent construct.
            construct_id: Unique identifier for this construct.
            workteam_name: Name for the SageMaker workteam.
            cognito: CognitoWorkforceTemplate providing pool/client/group.
            notification_topic: SnsTopicTemplate for task notifications.
        """
        super().__init__(scope, construct_id)

        region = Stack.of(self).region
        account = Stack.of(self).account

        self.workteam_name = workteam_name

        topic_arn = notification_topic.topic.topic_arn
        if topic_arn is None:
            msg = "notification_topic must have a resolved topic_arn"
            raise ValueError(msg)

        self.workteam = sagemaker.CfnWorkteam(
            self,
            f"{construct_id}-Workteam",
            workteam_name=workteam_name,
            description=f"Review team for {workteam_name}",
            member_definitions=[
                sagemaker.CfnWorkteam.MemberDefinitionProperty(
                    cognito_member_definition=sagemaker.CfnWorkteam.CognitoMemberDefinitionProperty(
                        cognito_client_id=cognito.client_id,
                        cognito_user_group=cognito.user_pool_group.group_name,  # type: ignore[arg-type]
                        cognito_user_pool=cognito.user_pool_id,
                    ),
                ),
            ],
            notification_configuration=sagemaker.CfnWorkteam.NotificationConfigurationProperty(
                notification_topic_arn=topic_arn,
            ),
        )
        # Workteam depends on the Cognito group existing first
        self.workteam.add_dependency(cognito.user_pool_group)

        self.workteam_arn = f"arn:aws:sagemaker:{region}:{account}:workteam/private-crowd/{workteam_name}"

        logger.info("Created SageMaker workteam: {}", workteam_name)
