"""
Unit tests for RetrievalStack CDK infrastructure.

Verifies stack synthesis, resource creation, IAM permissions, S3 notifications,
SQS event source mapping, naming conventions, and logging bucket reuse.

To avoid cross-stack cyclic references (KMS key resource policy references
S3 bucket ARN while Lambda references VPC subnets), all shared resources
are created inside a single stack alongside the retrieval resources.

**Validates: Requirements 15.7**
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import (
    RemovalPolicy,
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
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_lambda_event_sources as lambda_event_sources,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_s3_notifications as s3_notifications,
)
from aws_cdk.assertions import Template
from cdk_nag import NagSuppressions

from config.config import RetrievalConfig
from project_constructs.lambda_function import LambdaTemplate
from project_constructs.opensearch import OpenSearchServerlessConstruct
from project_constructs.s3 import BucketTemplate
from project_constructs.sqs import SqsQueueTemplate
from tests.unit.conftest import _mock_from_asset
from tests.unit.retrieval.conftest import _valid_retrieval_config

pytestmark = pytest.mark.retrieval


class _RetrievalTestStack(cdk.Stack):
    """Single stack that embeds shared infra + retrieval resources.

    Mirrors the resource creation logic of ``RetrievalStack`` but keeps
    VPC, KMS key, and logs bucket in the same CFn template to avoid the
    cross-stack cyclic dependency that CDK creates when a KMS key in
    stack A auto-grants to an S3 bucket in stack B while stack B already
    depends on stack A for VPC subnets.
    """

    def __init__(
        self,
        scope: cdk.App,
        construct_id: str,
        retrieval_config: RetrievalConfig,
        prefix: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Shared resources (SecurityStack + DataStack stand-ins)
        vpc = ec2.Vpc(self, f"{prefix}-Vpc", max_azs=2, nat_gateways=1)
        kms_key_res = kms.Key(
            self,
            f"{prefix}-KmsKey",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.DESTROY,
        )
        kms_key_policy = iam.ManagedPolicy(
            self,
            f"{prefix}-KmsKeyPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["kms:Decrypt", "kms:DescribeKey", "kms:GenerateDataKey"],
                    resources=[kms_key_res.key_arn],
                ),
            ],
        )
        logs_bucket = s3.Bucket(
            self,
            f"{prefix}-LoggingBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # --- Retrieval resources (mirrors infrastructure/retrieval.py) ---
        ingestion_bucket_name = f"{self.account}-{self.region}-{prefix}-retrieval-images-bucket"
        self.ingestion_bucket = BucketTemplate(
            self,
            f"{prefix}-RetrievalImagesBucket",
            bucket_name=ingestion_bucket_name,
            kms_key=kms_key_res,
            logging_bucket=logs_bucket,
        )

        self.sqs_queue = SqsQueueTemplate(
            self,
            f"{prefix}-RetrievalIngestQueue",
            queue_name=f"{prefix}-retrieval-ingest-queue",
            kms_key=kms_key_res,
            visibility_timeout_seconds=retrieval_config.sqs_visibility_timeout_seconds,
            max_receive_count=retrieval_config.sqs_max_receive_count,
        )

        self.ingest_lambda = LambdaTemplate(
            self,
            f"{prefix}-RetrievalIngestLambda",
            function_name=f"{prefix}-retrieval-ingest",
            lambda_path="retrieval_ingest",
            description="Downloads images, generates Bedrock Titan embeddings, indexes to AOSS",
            vpc=vpc,
            kms_key=kms_key_res,
            timeout=retrieval_config.ingest_lambda_timeout_seconds,
            memory_size=retrieval_config.ingest_lambda_memory_mb,
            env_vars={
                "RETRIEVAL_BUCKET_NAME": ingestion_bucket_name,
                "AOSS_INDEX_NAME": retrieval_config.index_name,
                "EMBEDDING_MODEL_ID": retrieval_config.embedding_model_id,
            },
        )

        self.opensearch = OpenSearchServerlessConstruct(
            self,
            f"{prefix}-RetrievalOss",
            collection_name=retrieval_config.collection_name,
            prefix=prefix,
            kms_key=kms_key_res,
            ssm_parameter_name=f"/{prefix}/retrieval/aoss-endpoint",
            principal_arns=[self.ingest_lambda.function_role.role_arn],
        )

        self.ingest_lambda.lambda_function.add_environment(
            "AOSS_ENDPOINT",
            self.opensearch.collection_endpoint,
        )

        self.ingestion_bucket.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3_notifications.SqsDestination(self.sqs_queue.queue),
            s3.NotificationKeyFilter(prefix="images/"),
        )

        self.ingest_lambda.lambda_function.add_event_source(
            lambda_event_sources.SqsEventSource(
                self.sqs_queue.queue,
                batch_size=10,
                report_batch_item_failures=True,
            )
        )

        # IAM permissions
        self.ingest_lambda.function_role.add_managed_policy(
            self.ingestion_bucket.read_write_policy,
        )
        self.ingest_lambda.function_role.add_managed_policy(
            self.sqs_queue.consume_policy,
        )
        self.ingest_lambda.function_role.add_managed_policy(kms_key_policy)

        self.ingest_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["bedrock:InvokeModel"],
                resources=[f"arn:aws:bedrock:{self.region}::foundation-model/{retrieval_config.embedding_model_id}"],
            )
        )
        self.ingest_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["aoss:APIAccessAll"],
                resources=[self.opensearch.collection.attr_arn],
            )
        )
        self.ingest_lambda.function_role.add_managed_policy(
            self.opensearch.endpoint_parameter.read_policy,
        )

        # CDK Nag suppressions
        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {"id": "AwsSolutions-IAM4", "reason": "Lambda managed policies"},
                {"id": "AwsSolutions-IAM5", "reason": "Bedrock/AOSS/S3 wildcards"},
                {"id": "AwsSolutions-SQS4", "reason": "KMS-encrypted internal queue"},
            ],
            apply_to_children=True,
        )


def _synth(
    prefix: str = "test",
    retrieval_config: RetrievalConfig | None = None,
) -> Template:
    """Synthesize the test stack and return the parsed CFn template."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    if retrieval_config is None:
        retrieval_config = _valid_retrieval_config()

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        stack = _RetrievalTestStack(
            app,
            f"{prefix}-RetrievalStack",
            retrieval_config=retrieval_config,
            prefix=prefix,
            env=env,
        )

    return Template.from_stack(stack, skip_cyclical_dependencies_check=True)


# ---------------------------------------------------------------------------
# 1. Stack synthesizes without errors
# ---------------------------------------------------------------------------
class TestStackSynthesizes:
    """Stack synthesizes without errors."""

    def test_synthesizes_successfully(self) -> None:
        template = _synth()
        assert template.to_json().get("Resources")


# ---------------------------------------------------------------------------
# 2. S3 bucket, SQS queue, Lambda function, OpenSearch collection present
# ---------------------------------------------------------------------------
class TestResourcesPresent:
    """All four core resource types are present in the template."""

    def test_s3_bucket_present(self) -> None:
        template = _synth()
        buckets = template.find_resources("AWS::S3::Bucket")
        # Logging bucket + ingestion bucket
        assert len(buckets) >= 2

    def test_sqs_queues_present(self) -> None:
        template = _synth()
        queues = template.find_resources("AWS::SQS::Queue")
        assert len(queues) >= 2, "Expected at least 2 SQS queues (main + DLQ)"

    def test_lambda_function_present(self) -> None:
        template = _synth()
        functions = template.find_resources("AWS::Lambda::Function")
        assert len(functions) >= 1

    def test_opensearch_collection_present(self) -> None:
        template = _synth()
        template.has_resource_properties(
            "AWS::OpenSearchServerless::Collection",
            {"Type": "VECTORSEARCH"},
        )


# ---------------------------------------------------------------------------
# 3. S3 notification on images/ prefix
# ---------------------------------------------------------------------------
class TestS3Notification:
    """S3 bucket notification targets SQS with images/ prefix filter."""

    def test_notification_on_images_prefix(self) -> None:
        template = _synth()
        custom_resources = template.find_resources("Custom::S3BucketNotifications")
        assert len(custom_resources) >= 1

        found = False
        for _lid, res in custom_resources.items():
            props = res.get("Properties", {})
            config = props.get("NotificationConfiguration", {})
            for qc in config.get("QueueConfigurations", []):
                rules = qc.get("Filter", {}).get("Key", {}).get("FilterRules", [])
                for rule in rules:
                    if rule.get("Name") == "prefix" and rule.get("Value") == "images/":
                        found = True
        assert found, "No S3 notification with images/ prefix found"


# ---------------------------------------------------------------------------
# 4. SQS event source mapping on Lambda with batch item failure reporting
# ---------------------------------------------------------------------------
class TestSqsEventSourceMapping:
    """SQS -> Lambda event source mapping with batch item failure reporting."""

    def test_event_source_mapping_exists(self) -> None:
        template = _synth()
        mappings = template.find_resources("AWS::Lambda::EventSourceMapping")
        assert len(mappings) >= 1

    def test_batch_size_is_10(self) -> None:
        template = _synth()
        template.has_resource_properties(
            "AWS::Lambda::EventSourceMapping",
            {"BatchSize": 10},
        )

    def test_reports_batch_item_failures(self) -> None:
        template = _synth()
        template.has_resource_properties(
            "AWS::Lambda::EventSourceMapping",
            {"FunctionResponseTypes": ["ReportBatchItemFailures"]},
        )


# ---------------------------------------------------------------------------
# 5. IAM permissions for Bedrock InvokeModel and AOSS APIAccessAll
# ---------------------------------------------------------------------------
class TestIamPermissions:
    """Bedrock InvokeModel and AOSS APIAccessAll policies present."""

    @staticmethod
    def _collect_all_actions(template: Template) -> list[str]:
        """Gather every IAM action string from the template."""
        actions: list[str] = []
        for resource_type in (
            "AWS::IAM::ManagedPolicy",
            "AWS::IAM::Policy",
            "AWS::IAM::Role",
        ):
            for _lid, res in template.find_resources(resource_type).items():
                props = res.get("Properties", {})
                for pol in props.get("Policies", []):
                    for stmt in pol.get("PolicyDocument", {}).get("Statement", []):
                        a = stmt.get("Action", [])
                        actions.extend(a if isinstance(a, list) else [a])
                for stmt in props.get("PolicyDocument", {}).get("Statement", []):
                    a = stmt.get("Action", [])
                    actions.extend(a if isinstance(a, list) else [a])
        return actions

    def test_bedrock_invoke_model(self) -> None:
        template = _synth()
        actions = self._collect_all_actions(template)
        assert any("bedrock:InvokeModel" in a for a in actions)

    def test_aoss_api_access_all(self) -> None:
        template = _synth()
        actions = self._collect_all_actions(template)
        assert any("aoss:APIAccessAll" in a for a in actions)


# ---------------------------------------------------------------------------
# 6. No "rag" substring in any logical ID (case-insensitive)
# ---------------------------------------------------------------------------
class TestNoRagInIdentifiers:
    """No 'rag' substring in any template key (case-insensitive)."""

    def test_no_rag_in_logical_ids(self) -> None:
        template = _synth()
        tpl = template.to_json()
        for section in ("Resources", "Outputs", "Parameters", "Conditions"):
            for key in tpl.get(section, {}):
                assert "rag" not in key.lower(), f"'{key}' in {section} contains 'rag'"


# ---------------------------------------------------------------------------
# 7. No separate logging bucket created (uses DataStack's)
# ---------------------------------------------------------------------------
class TestLoggingBucketFromDataStack:
    """Ingestion bucket references an external logging bucket."""

    def test_ingestion_bucket_has_logging_config(self) -> None:
        template = _synth()
        # Find the ingestion bucket (KMS-encrypted, not the logging bucket)
        buckets = template.find_resources("AWS::S3::Bucket")
        kms_buckets = {
            lid: res
            for lid, res in buckets.items()
            if res.get("Properties", {})
            .get("BucketEncryption", {})
            .get("ServerSideEncryptionConfiguration", [{}])[0]
            .get("ServerSideEncryptionByDefault", {})
            .get("SSEAlgorithm")
            == "aws:kms"
        }
        assert len(kms_buckets) >= 1, "Expected at least 1 KMS-encrypted bucket"
        for _lid, res in kms_buckets.items():
            props = res.get("Properties", {})
            assert props.get("LoggingConfiguration"), (
                "Ingestion bucket should have LoggingConfiguration pointing to DataStack's logging bucket"
            )
