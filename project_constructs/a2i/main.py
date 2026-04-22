"""
Reusable Amazon A2I (Augmented AI) human review construct.

This module provides an A2IHumanReview construct that creates a complete
human-in-the-loop review workflow for validating generated media assets
(images, videos, or audio). It composes:

- SnsTopicTemplate for task completion notifications
- CognitoWorkforceTemplate for reviewer authentication
- WorkteamTemplate for the private workteam
- A Liquid HTML task template (HumanTaskUi) via AwsCustomResource
- A flow definition that routes media to human reviewers via AwsCustomResource
- IAM roles with least-privilege managed policies

Note: AWS::SageMaker::HumanTaskUi and AWS::SageMaker::FlowDefinition are NOT
valid CloudFormation resource types. We use AwsCustomResource (SDK calls) instead.

Template versioning: A short hash of the template content is appended to the
HumanTaskUi and FlowDefinition names so that any template change forces
CloudFormation to replace (delete + create) the resources automatically.

Both AwsCustomResource blocks define on_create AND on_update with the same
create SDK call (with ignore_error_codes_matching="ResourceInUse") so that
deploys are idempotent — if the resource already exists the call succeeds
silently, and if it doesn't it gets created.
"""

import hashlib
from pathlib import Path

from aws_cdk import (
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    custom_resources as cr,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from config.config import A2IConfig
from project_constructs.cognito import CognitoWorkforceTemplate
from project_constructs.s3 import BucketTemplate
from project_constructs.sns import SnsTopicTemplate
from project_constructs.workteam import WorkteamTemplate

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_TEMPLATE_FILES = {
    "image": _TEMPLATES_DIR / "image_review.html",
    "video": _TEMPLATES_DIR / "video_review.html",
    "audio": _TEMPLATES_DIR / "audio_review.html",
}

_TEMPLATE_FILES = {
    "image": _TEMPLATES_DIR / "image_review.html",
    "video": _TEMPLATES_DIR / "video_review.html",
    "audio": _TEMPLATES_DIR / "audio_review.html",
}


def _load_template(media_type: str) -> str:
    """Load a Liquid HTML task template from the templates directory."""
    template_path = _TEMPLATE_FILES[media_type]
    return template_path.read_text()


class A2IHumanReview(Construct):
    """
    Reusable construct for Amazon A2I human review workflows.

    Composes SnsTopicTemplate, CognitoWorkforceTemplate, and WorkteamTemplate
    to create a complete human-in-the-loop pipeline for reviewing generated
    media assets. Uses AwsCustomResource for HumanTaskUi and FlowDefinition
    since these don't have native CloudFormation resource types.

    Attributes:
        flow_definition_arn: ARN of the created A2I flow definition.
        notification_topic: SnsTopicTemplate for task completion notifications.
        cognito: CognitoWorkforceTemplate for reviewer authentication.
        workteam: WorkteamTemplate for the private workteam.
        role: IAM role used by the flow definition.
        cognito_user_pool_id: Cognito User Pool ID (for reuse by other constructs).
        cognito_client_id: Cognito Client ID (for reuse by other constructs).
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        flow_definition_name: str,
        input_bucket: BucketTemplate,
        output_bucket: BucketTemplate,
        kms_key: kms.Key,
        kms_key_policy: iam.ManagedPolicy,
        cfg: A2IConfig,
        existing_user_pool_id: str = "",
        existing_client_id: str = "",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id)

        region = Stack.of(self).region
        account = Stack.of(self).account

        # SageMaker resource names must match [a-z0-9](-*[a-z0-9])*
        safe_name = flow_definition_name.replace("_", "-")

        logger.info(
            "Creating A2IHumanReview: name={}, media_type={}",
            safe_name,
            cfg.media_type,
        )

        self.input_bucket = input_bucket
        self.output_bucket = output_bucket

        # --- SNS topic for task completion notifications ---
        self.notification_topic = SnsTopicTemplate(
            self,
            f"{construct_id}-NotificationTopic",
            topic_name=f"{safe_name}-notifications",
            kms_key=kms_key,
            service_principals=["sagemaker.amazonaws.com"],
        )

        # Allow SageMaker service to use the KMS key for encrypted SNS
        kms_key.grant_encrypt_decrypt(iam.ServicePrincipal("sagemaker.amazonaws.com"))

        # --- IAM role for the flow definition ---
        self.role = iam.Role(
            self,
            f"{construct_id}-FlowDefinitionRole",
            role_name=f"{safe_name}-a2i-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
        )

        # S3 read on input, write on output via managed policies
        self.role.add_managed_policy(input_bucket.read_policy)
        self.role.add_managed_policy(output_bucket.read_write_policy)
        self.role.add_managed_policy(kms_key_policy)

        # SNS publish via managed policy from the SnsTopicTemplate
        self.role.add_managed_policy(self.notification_topic.publish_policy)

        # --- Cognito workforce ---
        workteam_name = (cfg.workteam_name or f"{safe_name}-team").replace("_", "-")
        group_name = f"{safe_name}-reviewers"

        self.cognito = CognitoWorkforceTemplate(
            self,
            f"{construct_id}-Cognito",
            pool_name=f"{safe_name}-reviewers",
            group_name=group_name,
            existing_user_pool_id=existing_user_pool_id,
            existing_client_id=existing_client_id,
            domain_prefix=f"{safe_name}-workforce",
        )

        self.cognito_user_pool_id = self.cognito.user_pool_id
        self.cognito_client_id = self.cognito.client_id

        # --- Private Workteam ---
        self.workteam = WorkteamTemplate(
            self,
            f"{construct_id}-Workteam",
            workteam_name=workteam_name,
            cognito=self.cognito,
            notification_topic=self.notification_topic,
        )

        # --- Human Task UI via AwsCustomResource (SDK call) ---
        # Append a short content hash to the name so that any template change
        # forces CloudFormation to replace (delete old + create new) the resource.
        template_content = _load_template(cfg.media_type)
        tmpl_hash = hashlib.sha256(template_content.encode()).hexdigest()[:8]
        task_ui_name = f"{safe_name}-ui-{tmpl_hash}"

        _task_ui_create_call = cr.AwsSdkCall(
            service="SageMaker",
            action="createHumanTaskUi",
            parameters={
                "HumanTaskUiName": task_ui_name,
                "UiTemplate": {"Content": template_content},
            },
            physical_resource_id=cr.PhysicalResourceId.of(task_ui_name),
            ignore_error_codes_matching="ResourceInUse",
        )

        # AwsCustomResource creates its own internal Lambda + role.
        # We create a ManagedPolicy and attach it to the CR's grant_principal
        # instead of using from_statements() which creates inline policies.
        task_ui_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-HumanTaskUiPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "sagemaker:CreateHumanTaskUi",
                        "sagemaker:DeleteHumanTaskUi",
                        "sagemaker:DescribeHumanTaskUi",
                    ],
                    resources=["*"],
                ),
            ],
        )

        self.human_task_ui_cr = cr.AwsCustomResource(
            self,
            f"{construct_id}-HumanTaskUi",
            on_create=_task_ui_create_call,
            on_update=_task_ui_create_call,
            on_delete=cr.AwsSdkCall(
                service="SageMaker",
                action="deleteHumanTaskUi",
                parameters={"HumanTaskUiName": task_ui_name},
                ignore_error_codes_matching="ResourceNotFound",
            ),
            # Use ANY_RESOURCE with no extra permissions — we attach our own managed policy
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE,
            ),
        )
        self.human_task_ui_cr.grant_principal.add_managed_policy(task_ui_policy)  # type: ignore[attr-defined]

        self.human_task_ui_arn = f"arn:aws:sagemaker:{region}:{account}:human-task-ui/{task_ui_name}"

        # --- Flow Definition via AwsCustomResource (SDK call) ---
        # Include the same template hash so the flow definition is replaced
        # whenever the task UI changes (it references the task UI ARN).
        flow_def_name = f"{safe_name}-{tmpl_hash}"

        _flow_def_create_call = cr.AwsSdkCall(
            service="SageMaker",
            action="createFlowDefinition",
            parameters={
                "FlowDefinitionName": flow_def_name,
                "HumanLoopConfig": {
                    "HumanTaskUiArn": self.human_task_ui_arn,
                    "TaskTitle": cfg.task_title,
                    "TaskDescription": cfg.task_description,
                    "TaskCount": cfg.task_count,
                    "TaskTimeLimitInSeconds": cfg.task_timeout_seconds,
                    "WorkteamArn": self.workteam.workteam_arn,
                },
                "OutputConfig": {
                    "S3OutputPath": f"s3://{output_bucket.bucket.bucket_name}/a2i-results/",
                    "KmsKeyId": kms_key.key_arn,
                },
                "RoleArn": self.role.role_arn,
            },
            physical_resource_id=cr.PhysicalResourceId.of(flow_def_name),
            ignore_error_codes_matching="ResourceInUse",
        )

        # ManagedPolicy for the FlowDefinition custom resource Lambda
        flow_def_policy = iam.ManagedPolicy(
            self,
            f"{construct_id}-FlowDefinitionPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "sagemaker:CreateFlowDefinition",
                        "sagemaker:DeleteFlowDefinition",
                        "sagemaker:DescribeFlowDefinition",
                    ],
                    resources=["*"],
                ),
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[self.role.role_arn],
                ),
            ],
        )

        self.flow_definition_cr = cr.AwsCustomResource(
            self,
            f"{construct_id}-FlowDefinition",
            on_create=_flow_def_create_call,
            on_update=_flow_def_create_call,
            on_delete=cr.AwsSdkCall(
                service="SageMaker",
                action="deleteFlowDefinition",
                parameters={"FlowDefinitionName": flow_def_name},
                ignore_error_codes_matching="ResourceNotFound",
            ),
            policy=cr.AwsCustomResourcePolicy.from_sdk_calls(
                resources=cr.AwsCustomResourcePolicy.ANY_RESOURCE,
            ),
        )
        self.flow_definition_cr.grant_principal.add_managed_policy(flow_def_policy)  # type: ignore[attr-defined]

        # Ensure ordering: task UI, role, workteam, and output bucket fully created first
        self.flow_definition_cr.node.add_dependency(self.human_task_ui_cr)
        self.flow_definition_cr.node.add_dependency(self.workteam.workteam)
        self.flow_definition_cr.node.add_dependency(self.role)
        self.flow_definition_cr.node.add_dependency(output_bucket)

        self.flow_definition_arn = f"arn:aws:sagemaker:{region}:{account}:flow-definition/{flow_def_name}"

        logger.info("Created A2I flow definition: {}", safe_name)

        # --- CDK Nag suppressions ---
        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "A2I flow definition role uses bucket-scoped managed policies",
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "AwsCustomResource needs wildcard for SageMaker A2I resource creation",
                },
                {
                    "id": "AwsSolutions-L1",
                    "reason": "AwsCustomResource internal Lambda runtime is managed by CDK and cannot be overridden",
                },
            ],
            apply_to_children=True,
        )
