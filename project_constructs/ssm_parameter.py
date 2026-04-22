"""
Reusable SSM Parameter Store construct.

This module provides an SsmParameter construct that creates an SSM
parameter with KMS encryption and exposes managed policies for
read and write access.
"""

from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from constructs import Construct


class SsmParameter(Construct):
    """
    Reusable SSM Parameter with managed policies.

    Attributes:
        parameter: The SSM parameter resource.
        read_policy: ManagedPolicy for reading the parameter.
        write_policy: ManagedPolicy for writing the parameter.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        parameter_name: str,
        string_value: str,
        description: str = "",
        kms_key: kms.Key | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        self.parameter = ssm.StringParameter(
            self,
            f"{construct_id}-Parameter",
            parameter_name=parameter_name,
            string_value=string_value,
            description=description,
            tier=ssm.ParameterTier.ADVANCED,
        )

        # Policy name safe version (no slashes)
        policy_safe_name = parameter_name.lstrip("/").replace("/", "-")

        # Read policy
        read_statements = [
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[self.parameter.parameter_arn],
            )
        ]
        if kms_key:
            read_statements.append(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["kms:Decrypt"],
                    resources=[kms_key.key_arn],
                )
            )

        self.read_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-SsmReadPolicy",
            managed_policy_name=f"{policy_safe_name}-ssm-read-policy",
            statements=read_statements,
        )

        # Write policy
        write_statements = [
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["ssm:PutParameter"],
                resources=[self.parameter.parameter_arn],
            )
        ]
        if kms_key:
            write_statements.append(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["kms:Encrypt", "kms:GenerateDataKey"],
                    resources=[kms_key.key_arn],
                )
            )

        self.write_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-SsmWritePolicy",
            managed_policy_name=f"{policy_safe_name}-ssm-write-policy",
            statements=write_statements,
        )
