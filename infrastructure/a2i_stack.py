"""
A2I (Augmented AI) infrastructure stack.

This module defines the A2IStack which creates A2I human review resources
as a separate stack from the pipeline. It creates flow definitions, task UIs,
workteams, and submit/process Lambdas for each active A2I config.

A2I output buckets are owned by DataStack so they survive A2IStack
delete/recreate cycles. This stack looks them up via
``data_stack.a2i_output_buckets``.

A2I resources are only created when BOTH conditions are met:
1. An a2i: config entry exists
2. A lambda_steps: entry references it via a2i_name

Cognito User Pool discovery:
SageMaker requires all private workteams in an account to share the same
Cognito User Pool Client. On first deploy the A2IHumanReview construct
creates a new pool. Subsequent deploys (same or different config) discover
the IDs automatically from existing workteams and pass them to every A2I
construct.
"""

import boto3
from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as events_targets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_sns_subscriptions as sns_subs,
)
from botocore.exceptions import ClientError
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from config.config import PipelineConfig
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack
from project_constructs.a2i import A2IHumanReview  # noqa: F401 - from __init__.py
from project_constructs.lambda_function import LambdaTemplate


def _discover_cognito_from_workteams() -> tuple[str, str] | None:
    """Discover Cognito User Pool and Client ID from existing SageMaker workteams.

    SageMaker requires all private workteams in an account to share the same
    Cognito User Pool Client. This queries existing workteams at synth time
    to extract the pool/client IDs for reuse.

    Returns:
        Tuple of (user_pool_id, client_id) if found, else None.
    """
    try:
        sm = boto3.client("sagemaker")
        resp = sm.list_workteams(SortBy="CreateDate", SortOrder="Descending", MaxResults=1)
        workteams = resp.get("Workteams", [])
        if not workteams:
            return None
        members = workteams[0].get("MemberDefinitions", [])
        for member in members:
            cognito_def = member.get("CognitoMemberDefinition")
            if cognito_def:
                pool_id = cognito_def.get("UserPool", "")
                client_id = cognito_def.get("ClientId", "")
                if pool_id and client_id:
                    logger.info(
                        "Discovered Cognito from workteam '{}': pool={}, client={}",
                        workteams[0].get("WorkteamName", "?"),
                        pool_id,
                        client_id,
                    )
                    return pool_id, client_id
    except ClientError as exc:
        logger.warning("Could not discover Cognito from workteams: {}", exc)
    return None


class A2IStack(Stack):
    """
    CDK Stack for A2I human review resources.

    Creates flow definitions, task UIs, workteams, submit/process Lambdas.
    Output buckets are owned by DataStack and looked up via
    ``data_stack.a2i_output_buckets``.

    Cognito IDs are auto-discovered from existing workteams at synth time.

    Attributes:
        a2i_constructs: Dict of A2I construct name -> A2IHumanReview.
        submit_lambdas: Dict of A2I name -> LambdaTemplate (submit Lambda).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        security_stack: SecurityStack,
        data_stack: DataStack,
        pipeline_config: PipelineConfig,
        prefix: str = "dev",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        logger.info("Creating A2IStack with prefix: {}", prefix)

        # Determine which A2I configs are active
        referenced_a2i_names = {ls_cfg.a2i_name for ls_cfg in pipeline_config.lambda_steps.values() if ls_cfg.a2i_name}
        active_a2i_names = referenced_a2i_names & set(pipeline_config.a2i.keys())

        self.a2i_constructs: dict[str, A2IHumanReview] = {}
        self.submit_lambdas: dict[str, LambdaTemplate] = {}

        # --- Discover existing Cognito IDs at synth time ---
        # SageMaker requires all workteams to share the same Cognito pool.
        # On first deploy no workteams exist so the first A2IHumanReview
        # construct creates a new pool. On subsequent deploys (same or
        # different config) we discover the IDs from existing workteams.
        discovered = _discover_cognito_from_workteams()
        cognito_user_pool_id = discovered[0] if discovered else ""
        cognito_client_id = discovered[1] if discovered else ""

        for a2i_name, a2i_cfg in pipeline_config.a2i.items():
            if a2i_name not in active_a2i_names:
                logger.info("Skipping A2I '{}' — needs both a2i config and lambda_step reference", a2i_name)
                continue

            # Output bucket for review results (owned by DataStack)
            a2i_output_bucket = data_stack.a2i_output_buckets[a2i_name]

            a2i_construct = A2IHumanReview(
                self,
                f"{prefix}-{a2i_name}-A2I",
                flow_definition_name=f"{prefix}-{a2i_name}",
                input_bucket=data_stack.input_bucket,
                output_bucket=a2i_output_bucket,
                kms_key=security_stack.kms_key,
                kms_key_policy=security_stack.kms_key_policy,
                cfg=a2i_cfg,
                existing_user_pool_id=cognito_user_pool_id,
                existing_client_id=cognito_client_id,
            )
            self.a2i_constructs[a2i_name] = a2i_construct

            # After the first construct creates a pool, capture its IDs
            # so subsequent constructs in this stack reuse the same pool.
            if not cognito_user_pool_id:
                cognito_user_pool_id = a2i_construct.cognito_user_pool_id
                cognito_client_id = a2i_construct.cognito_client_id

            logger.info("Created A2I human review for step: {}", a2i_name)

            CfnOutput(
                self,
                f"{prefix}-{a2i_name}-FlowDefinitionArn",
                value=a2i_construct.flow_definition_arn,
                description=f"A2I flow definition ARN for {a2i_name}",
            )

            # --- Submit Lambda: lists assets and starts human loops ---
            a2i_short = a2i_name.replace("_", "-")

            submit_lambda = LambdaTemplate(
                self,
                f"{prefix}-{a2i_name}-SubmitA2ILambda",
                function_name=f"{prefix}-{a2i_short}-submit",
                lambda_path="submit_a2i_review",
                description=f"Submits {a2i_cfg.media_type} assets to A2I for human review ({a2i_name})",
                vpc=security_stack.vpc,
                kms_key=security_stack.kms_key,
                env_vars={
                    "FLOW_DEFINITION_ARN": a2i_construct.flow_definition_arn,
                    "SOURCE_BUCKET": data_stack.input_bucket.bucket.bucket_name,
                    "DYNAMODB_TABLE_NAME": data_stack.dynamodb_table.table.table_name,
                    "MEDIA_TYPE": a2i_cfg.media_type,
                    "TASK_TITLE": a2i_cfg.task_title,
                    "TASK_DESCRIPTION": a2i_cfg.task_description,
                },
            )
            submit_lambda.function_role.add_managed_policy(data_stack.dynamodb_table.read_write_policy)
            submit_lambda.function_role.add_managed_policy(security_stack.kms_key_policy)
            submit_lambda.function_role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["sagemaker:StartHumanLoop"],
                    resources=["*"],
                )
            )

            self.submit_lambdas[a2i_name] = submit_lambda

            CfnOutput(
                self,
                f"{prefix}-{a2i_name}-SubmitLambdaName",
                value=submit_lambda.lambda_function.function_name,
                description=f"A2I submit Lambda for {a2i_name}",
            )
            logger.info("Created A2I submit Lambda for: {}", a2i_name)

            process_lambda = LambdaTemplate(
                self,
                f"{prefix}-{a2i_name}-ProcessA2ILambda",
                function_name=f"{prefix}-{a2i_short}-process",
                lambda_path="process_a2i_results",
                description=f"Processes completed A2I reviews and writes results to DynamoDB ({a2i_name})",
                vpc=security_stack.vpc,
                kms_key=security_stack.kms_key,
                env_vars={
                    "DYNAMODB_TABLE_NAME": data_stack.dynamodb_table.table.table_name,
                },
            )
            process_lambda.function_role.add_managed_policy(a2i_output_bucket.read_policy)
            process_lambda.function_role.add_managed_policy(data_stack.dynamodb_table.read_write_policy)
            process_lambda.function_role.add_managed_policy(security_stack.kms_key_policy)
            process_lambda.function_role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    actions=["sagemaker:DescribeHumanLoop"],
                    resources=["*"],
                )
            )

            a2i_construct.notification_topic.topic.add_subscription(
                sns_subs.LambdaSubscription(process_lambda.lambda_function)
            )

            events.Rule(
                self,
                f"{prefix}-{a2i_name}-A2ICompletionRule",
                rule_name=f"{prefix}-{a2i_short}-a2i-completion",
                description=f"Triggers process Lambda when A2I human loop completes for {a2i_name}",
                event_pattern=events.EventPattern(
                    source=["aws.sagemaker"],
                    detail_type=["SageMaker A2I HumanLoop Status Change"],
                    detail={
                        "flowDefinitionArn": [a2i_construct.flow_definition_arn],
                        "humanLoopStatus": ["Completed", "Failed", "Stopped"],
                    },
                ),
                targets=[events_targets.LambdaFunction(process_lambda.lambda_function)],
            )

            CfnOutput(
                self,
                f"{prefix}-{a2i_name}-ProcessLambdaName",
                value=process_lambda.lambda_function.function_name,
                description=f"A2I results processor Lambda for {a2i_name}",
            )
            logger.info("Created A2I process results Lambda for: {}", a2i_name)

        # --- Log discovered Cognito IDs ---
        if cognito_user_pool_id and self.a2i_constructs:
            CfnOutput(
                self,
                f"{prefix}-CognitoUserPoolId",
                value=cognito_user_pool_id,
                description="Cognito User Pool ID for A2I workforce",
            )
            CfnOutput(
                self,
                f"{prefix}-CognitoClientId",
                value=cognito_client_id,
                description="Cognito User Pool Client ID for A2I workforce",
            )
            logger.info(
                "Cognito IDs: pool={}, client={}",
                cognito_user_pool_id,
                cognito_client_id,
            )

        # --- CDK Nag Suppressions ---
        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "Lambda and A2I roles use AWS managed policies",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "A2I and Lambda roles require wildcard for dynamic resource names",
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "AwsCustomResource internal Lambda runtime is managed by CDK and cannot be overridden",
                },
            ],
            apply_to_children=True,
        )
