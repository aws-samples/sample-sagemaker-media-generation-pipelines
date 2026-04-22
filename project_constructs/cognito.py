"""
Reusable Cognito workforce construct for A2I private workteams.

This module provides a CognitoWorkforceTemplate construct that creates
(or imports) a Cognito User Pool with client, domain, and reviewer group
for use with SageMaker private workteams.
"""

from aws_cdk import (
    RemovalPolicy,
)
from aws_cdk import (
    aws_cognito as cognito,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger


class CognitoWorkforceTemplate(Construct):
    """
    Reusable Cognito workforce construct.

    Either imports an existing User Pool (when existing IDs are provided)
    or creates a new one with client, domain, and reviewer group.

    Attributes:
        user_pool: The Cognito User Pool (imported or created).
        user_pool_id: The User Pool ID string.
        client_id: The User Pool Client ID string.
        user_pool_group: The CfnUserPoolGroup for reviewers.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        pool_name: str,
        group_name: str,
        existing_user_pool_id: str = "",
        existing_client_id: str = "",
        domain_prefix: str = "",
    ) -> None:
        """
        Initialize the CognitoWorkforceTemplate.

        Args:
            scope: The parent construct.
            construct_id: Unique identifier for this construct.
            pool_name: Name for the Cognito User Pool (used when creating new).
            group_name: Name for the reviewer group.
            existing_user_pool_id: Existing pool ID to import (skip creation).
            existing_client_id: Existing client ID to import (skip creation).
            domain_prefix: Cognito domain prefix (used when creating new pool).
        """
        super().__init__(scope, construct_id)

        if existing_user_pool_id and existing_client_id:
            logger.info(
                "Reusing existing Cognito User Pool: {} / client: {}",
                existing_user_pool_id,
                existing_client_id,
            )
            self.user_pool = cognito.UserPool.from_user_pool_id(
                self,
                f"{construct_id}-UserPool",
                existing_user_pool_id,
            )
            self.user_pool_id = existing_user_pool_id
            self.client_id = existing_client_id
        else:
            logger.info("Creating new Cognito User Pool: {}", pool_name)
            self.user_pool = cognito.UserPool(
                self,
                f"{construct_id}-UserPool",
                user_pool_name=pool_name,
                self_sign_up_enabled=False,
                removal_policy=RemovalPolicy.DESTROY,
                password_policy=cognito.PasswordPolicy(
                    min_length=8,
                    require_uppercase=True,
                    require_lowercase=True,
                    require_digits=True,
                    require_symbols=True,
                ),
                mfa=cognito.Mfa.OPTIONAL,
                mfa_second_factor=cognito.MfaSecondFactor(sms=True, otp=True),
                feature_plan=cognito.FeaturePlan.PLUS,
                advanced_security_mode=cognito.AdvancedSecurityMode.ENFORCED,
            )

            user_pool_client = self.user_pool.add_client(
                f"{construct_id}-UserPoolClient",
                user_pool_client_name=f"{pool_name}-client",
                generate_secret=True,
                auth_flows=cognito.AuthFlow(user_password=True, user_srp=True),
            )

            self.user_pool.add_domain(
                f"{construct_id}-UserPoolDomain",
                cognito_domain=cognito.CognitoDomainOptions(
                    domain_prefix=domain_prefix or f"{pool_name}-workforce",
                ),
            )

            self.user_pool_id = self.user_pool.user_pool_id
            self.client_id = user_pool_client.user_pool_client_id

        # Reviewer group (always created — idempotent via group_name)
        self.user_pool_group = cognito.CfnUserPoolGroup(
            self,
            f"{construct_id}-UserPoolGroup",
            user_pool_id=self.user_pool_id,
            group_name=group_name,
            description=f"Reviewer group for {pool_name}",
        )

        logger.info("Cognito workforce ready: pool={}, group={}", self.user_pool_id, group_name)

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-COG2",
                    "reason": "MFA is optional for internal dev review workforce",
                },
            ],
            apply_to_children=True,
        )
