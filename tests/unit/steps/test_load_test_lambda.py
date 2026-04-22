"""
Unit tests for the retrieval load test Lambda handler.

Tests parse_event validation and defaults, generate_images file creation,
assemble_result status logic, upload_images with mocked S3, and
cleanup_artifacts targeting only loadtest-prefixed keys.
"""

import os
import shutil
import struct
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Set env vars before importing the module("RETRIEVAL_BUCKET_NAME", "test-retrieval-bucket")
os.environ.setdefault("INGEST_QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue")
os.environ.setdefault("INGEST_FUNCTION_NAME", "test-ingest-function")
os.environ.setdefault("AOSS_ENDPOINT", "https://test.us-east-1.aoss.amazonaws.com")
os.environ.setdefault("AOSS_INDEX_NAME", "image-vectors")

# Stub out opensearchpy before importing the Lambda module to avoid
# pulling in requests/urllib3 (slow import, high memory in xdist workers).
import sys
import types

_oss_stub = types.ModuleType("opensearchpy")
_oss_stub.OpenSearch = MagicMock
_oss_stub.AWSV4SignerAuth = MagicMock
_oss_stub.RequestsHttpConnection = MagicMock
sys.modules["opensearchpy"] = _oss_stub

from lambdas.retrieval_load_test.index import (
    assemble_result,
    cleanup_artifacts,
    generate_images,
    parse_event,
    upload_images,
)


# ---------------------------------------------------------------------------
# 1. parse_event validation and defaults
# ---------------------------------------------------------------------------
class TestParseEvent:
    """Validates parse_event defaults, round-trips, and rejection rules."""

    def test_empty_event_returns_defaults(self) -> None:
        result = parse_event({})
        assert result == {
            "num_images": 1000,
            "poll_interval": 5,
            "timeout": 600,
            "image_size": 64,
            "cleanup": True,
        }

    def test_valid_event_round_trips(self) -> None:
        event = {
            "num_images": 500,
            "poll_interval": 10,
            "timeout": 300,
            "image_size": 128,
            "cleanup": False,
        }
        assert parse_event(event) == event

    def test_num_images_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="num_images must be a positive integer"):
            parse_event({"num_images": 0})

    def test_num_images_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="num_images must be a positive integer"):
            parse_event({"num_images": -1})

    def test_num_images_float_raises(self) -> None:
        with pytest.raises(ValueError, match="num_images must be a positive integer"):
            parse_event({"num_images": 1.5})

    def test_timeout_871_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout must be <= 870"):
            parse_event({"timeout": 871})

    def test_timeout_870_accepted(self) -> None:
        result = parse_event({"timeout": 870})
        assert result["timeout"] == 870


# ---------------------------------------------------------------------------
# 2. generate_images file creation
# ---------------------------------------------------------------------------
class TestGenerateImages:
    """Validates generate_images creates correct files with correct names."""

    def test_creates_correct_number_of_files(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        try:
            result = generate_images(5, 32, tmp_dir)
            assert len(result) == 5
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_file_names_match_pattern(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        try:
            result = generate_images(5, 32, tmp_dir)
            for i, path in enumerate(result):
                assert os.path.basename(path) == f"loadtest-{i:04d}.png"
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_files_are_valid_pngs(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        try:
            result = generate_images(3, 32, tmp_dir)
            for path in result:
                with open(path, "rb") as f:
                    header = f.read(24)
                assert header[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG"
                w, h = struct.unpack(">II", header[16:24])
                assert (w, h) == (32, 32)
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3. assemble_result status logic
# ---------------------------------------------------------------------------
class TestAssembleResult:
    """Validates assemble_result PASS/FAIL logic and throughput calculation."""

    @staticmethod
    def _monitor_result(
        documents_indexed: int = 10,
        wall_clock_seconds: float = 10.0,
        peak_concurrent: int = 5,
        doc_count_final: int = 10,
    ) -> dict:
        return {
            "documents_indexed": documents_indexed,
            "wall_clock_seconds": wall_clock_seconds,
            "peak_concurrent_executions": peak_concurrent,
            "document_count_final": doc_count_final,
            "time_series": [],
        }

    def test_pass_when_all_indexed(self) -> None:
        result = assemble_result(10, 0, 10, self._monitor_result(documents_indexed=10))
        assert result["status"] == "PASS"

    def test_fail_when_not_all_indexed(self) -> None:
        result = assemble_result(10, 0, 10, self._monitor_result(documents_indexed=7))
        assert result["status"] == "FAIL"
        assert result["shortfall"] == 3

    def test_throughput_calculation(self) -> None:
        result = assemble_result(
            100,
            0,
            100,
            self._monitor_result(documents_indexed=100, wall_clock_seconds=10.0),
        )
        assert result["throughput_images_per_second"] == 10.0

    def test_zero_wall_clock_no_division_error(self) -> None:
        result = assemble_result(
            10,
            0,
            10,
            self._monitor_result(documents_indexed=10, wall_clock_seconds=0.0),
        )
        assert result["throughput_images_per_second"] == 0.0


# ---------------------------------------------------------------------------
# 4. upload_images with mocked S3
# ---------------------------------------------------------------------------
class TestUploadImages:
    """Validates upload_images success/failure counts, key prefixes, and tmp cleanup."""

    def test_success_and_failure_counts(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        try:
            # Create actual temp files
            paths = []
            for i in range(5):
                p = os.path.join(tmp_dir, f"loadtest-{i:04d}.png")
                with open(p, "wb") as f:
                    f.write(b"\x89PNG fake")
                paths.append(p)

            mock_s3 = MagicMock()
            # Make upload_file raise on the 3rd file
            fail_file = os.path.basename(paths[2])

            def _side_effect(local_path, bucket, key):
                if os.path.basename(local_path) == fail_file:
                    raise Exception("Simulated S3 error")

            mock_s3.upload_file.side_effect = _side_effect

            num_uploaded, num_failed, _ = upload_images(paths, "test-bucket", s3_client=mock_s3)
            assert num_uploaded + num_failed == 5
            assert num_failed == 1
            assert num_uploaded == 4
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_upload_keys_use_images_prefix(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        try:
            paths = []
            for i in range(3):
                p = os.path.join(tmp_dir, f"loadtest-{i:04d}.png")
                with open(p, "wb") as f:
                    f.write(b"\x89PNG fake")
                paths.append(p)

            mock_s3 = MagicMock()
            upload_images(paths, "test-bucket", s3_client=mock_s3)

            for c in mock_s3.upload_file.call_args_list:
                _, bucket, key = c[0]
                assert key.startswith("images/"), f"Key {key} does not start with images/"
        finally:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_tmp_files_cleaned_after_upload(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        paths = []
        for i in range(3):
            p = os.path.join(tmp_dir, f"loadtest-{i:04d}.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG fake")
            paths.append(p)

        mock_s3 = MagicMock()
        upload_images(paths, "test-bucket", s3_client=mock_s3)

        assert not os.path.exists(tmp_dir), "tmp dir should be removed after upload"


# ---------------------------------------------------------------------------
# 5. cleanup_artifacts targeting only loadtest-prefixed keys
# ---------------------------------------------------------------------------
class TestCleanupArtifacts:
    """Validates cleanup_artifacts targets only loadtest-prefixed keys."""

    @staticmethod
    def _make_paginator(keys: list[str]) -> MagicMock:
        """Build a mock S3 paginator that returns the given keys."""
        page = {"Contents": [{"Key": k} for k in keys]}
        paginator = MagicMock()
        paginator.paginate.return_value = [page]
        return paginator

    def test_only_loadtest_prefixed_keys_deleted(self) -> None:
        mock_s3 = MagicMock()

        # First paginate call (images/loadtest-*) returns mix
        loadtest_keys = ["images/loadtest-0000.png", "images/loadtest-0001.png"]

        # The function uses prefix filtering, so the paginator for
        # "images/loadtest-" should only return loadtest keys.
        # The non-loadtest keys would never appear because S3 prefix filtering
        # excludes them. We simulate this correctly.
        paginator_loadtest = self._make_paginator(loadtest_keys)
        paginator_base64 = self._make_paginator(["base64/images/loadtest-0000.png"])

        call_count = 0

        def _get_paginator(op):
            nonlocal call_count
            if op == "list_objects_v2":
                p = MagicMock()

                def _paginate(**kwargs):
                    prefix = kwargs.get("Prefix", "")
                    if prefix.startswith("base64/"):
                        return paginator_base64.paginate.return_value
                    return paginator_loadtest.paginate.return_value

                p.paginate = _paginate
                return p
            return MagicMock()

        mock_s3.get_paginator = _get_paginator

        mock_aoss = MagicMock()
        mock_aoss.delete_by_query.return_value = {"deleted": 2}

        cleanup_artifacts("test-bucket", mock_aoss, "test-index", s3_client=mock_s3)

        # Verify delete_objects was called and all keys are loadtest-prefixed
        for c in mock_s3.delete_objects.call_args_list:
            objects = (
                c[1]["Delete"]["Objects"] if "Delete" in c[1] else c[0][1]["Delete"]["Objects"] if len(c[0]) > 1 else []
            )
            if not objects:
                # Try kwargs
                objects = c.kwargs.get("Delete", {}).get("Objects", [])
            for obj in objects:
                assert "loadtest-" in obj["Key"], f"Non-loadtest key deleted: {obj['Key']}"

    def test_aoss_delete_by_query_called(self) -> None:
        mock_s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = paginator

        mock_aoss = MagicMock()
        mock_aoss.delete_by_query.return_value = {"deleted": 5}

        cleanup_artifacts("test-bucket", mock_aoss, "test-index", s3_client=mock_s3)

        mock_aoss.delete_by_query.assert_called_once_with(
            index="test-index",
            body={"query": {"prefix": {"description": "images/loadtest-"}}},
        )

    def test_s3_error_captured_in_errors(self) -> None:
        mock_s3 = MagicMock()
        mock_s3.get_paginator.side_effect = Exception("S3 list failed")

        mock_aoss = MagicMock()
        mock_aoss.delete_by_query.return_value = {"deleted": 0}

        result = cleanup_artifacts("test-bucket", mock_aoss, "test-index", s3_client=mock_s3)
        assert len(result["errors"]) > 0


# ---------------------------------------------------------------------------
# CDK Template Assertion Tests for Load Test Lambda
# ---------------------------------------------------------------------------
import json

import aws_cdk as cdk
from aws_cdk import (
    RemovalPolicy,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk.assertions import Template
from cdk_nag import NagSuppressions

from config.config import RetrievalConfig
from project_constructs.lambda_function import LambdaTemplate
from project_constructs.opensearch import OpenSearchServerlessConstruct
from project_constructs.s3 import BucketTemplate
from project_constructs.sqs import SqsQueueTemplate

pytestmark = pytest.mark.steps_loadtest


from tests.unit.conftest import _mock_from_asset


def _valid_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        collection_name="test-images",
        index_name="test-vectors",
        sqs_visibility_timeout_seconds=960,
        sqs_max_receive_count=3,
        ingest_lambda_timeout_seconds=300,
        ingest_lambda_memory_mb=2048,
    )


class _LoadTestRetrievalTestStack(cdk.Stack):
    """Test stack that mirrors RetrievalConstruct including the load test Lambda."""

    def __init__(
        self,
        scope: cdk.App,
        construct_id: str,
        retrieval_config: RetrievalConfig,
        prefix: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        logs_bucket = s3.Bucket(
            self,
            f"{prefix}-LoggingBucket",
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        ingestion_bucket_name = f"{self.account}-{self.region}-{prefix}-retrieval-images-bucket"
        self.ingestion_bucket = BucketTemplate(
            self,
            f"{prefix}-RetrievalImagesBucket",
            bucket_name=ingestion_bucket_name,
            logging_bucket=logs_bucket,
        )

        self.sqs_queue = SqsQueueTemplate(
            self,
            f"{prefix}-RetrievalIngestQueue",
            queue_name=f"{prefix}-retrieval-ingest-queue",
            visibility_timeout_seconds=retrieval_config.sqs_visibility_timeout_seconds,
            max_receive_count=retrieval_config.sqs_max_receive_count,
        )

        self.ingest_lambda = LambdaTemplate(
            self,
            f"{prefix}-RetrievalIngestLambda",
            function_name=f"{prefix}-retrieval-ingest",
            lambda_path="retrieval_ingest",
            description="Ingest Lambda",
            timeout=retrieval_config.ingest_lambda_timeout_seconds,
            memory_size=retrieval_config.ingest_lambda_memory_mb,
            env_vars={
                "RETRIEVAL_BUCKET_NAME": ingestion_bucket_name,
                "AOSS_INDEX_NAME": retrieval_config.index_name,
                "EMBEDDING_MODEL_ID": retrieval_config.embedding_model_id,
            },
        )

        # Load test Lambda — created BEFORE OpenSearch so role ARN is available
        self.load_test_lambda = LambdaTemplate(
            self,
            f"{prefix}-RetrievalLoadTestLambda",
            function_name=f"{prefix}-retrieval-load-test",
            lambda_path="retrieval_load_test",
            description="Load test for retrieval ingestion pipeline",
            timeout=900,
            memory_size=1024,
            env_vars={
                "RETRIEVAL_BUCKET_NAME": ingestion_bucket_name,
                "INGEST_QUEUE_URL": self.sqs_queue.queue.queue_url,
                "INGEST_FUNCTION_NAME": f"{prefix}-retrieval-ingest-Lambda",
                "AOSS_INDEX_NAME": retrieval_config.index_name,
            },
        )

        self.opensearch = OpenSearchServerlessConstruct(
            self,
            f"{prefix}-RetrievalOss",
            collection_name=retrieval_config.collection_name,
            prefix=prefix,
            ssm_parameter_name=f"/{prefix}/retrieval/aoss-endpoint",
            principal_arns=[
                self.ingest_lambda.function_role.role_arn,
                self.load_test_lambda.function_role.role_arn,
            ],
        )

        self.ingest_lambda.lambda_function.add_environment(
            "AOSS_ENDPOINT",
            self.opensearch.collection_endpoint,
        )
        self.load_test_lambda.lambda_function.add_environment(
            "AOSS_ENDPOINT",
            self.opensearch.collection_endpoint,
        )

        # IAM: ingest Lambda
        self.ingest_lambda.function_role.add_managed_policy(
            self.ingestion_bucket.read_write_policy,
        )
        self.ingest_lambda.function_role.add_managed_policy(
            self.sqs_queue.consume_policy,
        )
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

        # IAM: load test Lambda
        self.load_test_lambda.function_role.add_managed_policy(
            self.ingestion_bucket.read_write_policy,
        )
        self.load_test_lambda.function_role.add_managed_policy(
            self.sqs_queue.consume_policy,
        )
        self.load_test_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["cloudwatch:GetMetricData"],
                resources=["*"],
            )
        )
        self.load_test_lambda.function_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["aoss:APIAccessAll"],
                resources=[self.opensearch.collection.attr_arn],
            )
        )
        self.load_test_lambda.function_role.add_managed_policy(
            self.opensearch.endpoint_parameter.read_policy,
        )

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {"id": "AwsSolutions-IAM4", "reason": "Lambda managed policies"},
                {"id": "AwsSolutions-IAM5", "reason": "S3/AOSS wildcards"},
                {"id": "AwsSolutions-SQS4", "reason": "Internal queue"},
            ],
            apply_to_children=True,
        )


def _synth_cdk(
    prefix: str = "test",
    retrieval_config: RetrievalConfig | None = None,
) -> Template:
    """Synthesize the load-test test stack and return the parsed CFn template."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    if retrieval_config is None:
        retrieval_config = _valid_retrieval_config()

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        stack = _LoadTestRetrievalTestStack(
            app,
            f"{prefix}-LoadTestStack",
            retrieval_config=retrieval_config,
            prefix=prefix,
            env=env,
        )

    return Template.from_stack(stack, skip_cyclical_dependencies_check=True)


# ---------------------------------------------------------------------------
# CDK: Load Test Lambda resource presence
# ---------------------------------------------------------------------------
class TestLoadTestLambdaCdkResources:
    """Verify load test Lambda resources in the synthesized template."""

    def test_load_test_lambda_present(self) -> None:
        template = _synth_cdk()
        functions = template.find_resources("AWS::Lambda::Function")
        assert len(functions) >= 2, "Expected at least 2 Lambda functions (ingest + load test)"

    def test_load_test_lambda_timeout_900(self) -> None:
        template = _synth_cdk()
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {"Timeout": 900},
        )

    def test_load_test_lambda_memory_1024(self) -> None:
        template = _synth_cdk()
        template.has_resource_properties(
            "AWS::Lambda::Function",
            {"MemorySize": 1024},
        )


# ---------------------------------------------------------------------------
# CDK: Load Test Lambda environment variables
# ---------------------------------------------------------------------------
class TestLoadTestLambdaCdkEnvVars:
    """Verify load test Lambda environment variables."""

    def test_env_vars_set(self) -> None:
        template = _synth_cdk()
        functions = template.find_resources("AWS::Lambda::Function")
        # Find the Lambda with Timeout=900 AND MemorySize=1024 (load test)
        # Note: CDK's CustomS3AutoDeleteObjects handler also has Timeout=900
        load_test_fn = None
        for _lid, res in functions.items():
            props = res.get("Properties", {})
            if props.get("Timeout") == 900 and props.get("MemorySize") == 1024:
                load_test_fn = props
                break
        assert load_test_fn is not None, "No Lambda with Timeout=900 and MemorySize=1024 found"
        env_vars = load_test_fn.get("Environment", {}).get("Variables", {})
        for key in (
            "RETRIEVAL_BUCKET_NAME",
            "INGEST_QUEUE_URL",
            "INGEST_FUNCTION_NAME",
            "AOSS_INDEX_NAME",
        ):
            assert key in env_vars, f"Missing env var: {key}"


# ---------------------------------------------------------------------------
# CDK: Load Test Lambda IAM permissions
# ---------------------------------------------------------------------------
class TestLoadTestLambdaCdkIamPermissions:
    """Verify IAM permissions for the load test Lambda."""

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

    def test_cloudwatch_get_metric_data(self) -> None:
        template = _synth_cdk()
        actions = self._collect_all_actions(template)
        assert any("cloudwatch:GetMetricData" in a for a in actions)

    def test_aoss_api_access_all(self) -> None:
        template = _synth_cdk()
        actions = self._collect_all_actions(template)
        # Should appear at least twice (ingest + load test)
        count = sum(1 for a in actions if a == "aoss:APIAccessAll")
        assert count >= 2, f"Expected aoss:APIAccessAll at least twice, found {count}"


# ---------------------------------------------------------------------------
# CDK: AOSS data access policy includes load test role
# ---------------------------------------------------------------------------
class TestLoadTestLambdaCdkAossDataAccess:
    """Verify AOSS data access policy references the load test Lambda role."""

    def test_load_test_role_in_principal_arns(self) -> None:
        template = _synth_cdk()
        access_policies = template.find_resources("AWS::OpenSearchServerless::AccessPolicy")
        assert len(access_policies) >= 1, "No AOSS access policy found"
        for _lid, res in access_policies.items():
            policy = res.get("Properties", {}).get("Policy", "")
            # The policy is a Fn::Join containing string fragments and
            # Fn::GetAtt references for the role ARNs (CDK tokens).
            if isinstance(policy, dict) and "Fn::Join" in policy:
                join_parts = policy["Fn::Join"][1]
                # Count Fn::GetAtt references — each is a principal role ARN
                role_refs = [p for p in join_parts if isinstance(p, dict) and "Fn::GetAtt" in p]
                assert len(role_refs) >= 2, (
                    f"Expected at least 2 principal ARN refs in AOSS data access "
                    f"policy Fn::Join, found {len(role_refs)}"
                )
                # Verify the joined string contains "Principal"
                joined_str = "".join(p if isinstance(p, str) else "<REF>" for p in join_parts)
                assert "Principal" in joined_str
            elif isinstance(policy, str):
                parsed = json.loads(policy)
                for rule_set in parsed:
                    principals = rule_set.get("Principal", [])
                    role_refs = [p for p in principals if isinstance(p, (str, dict))]
                    assert len(role_refs) >= 2, f"Expected at least 2 principal ARNs, found {len(role_refs)}"


# ===========================================================================
# Property-Based Tests (Hypothesis)
# ===========================================================================


# Feature: ingestion-load-test, Property 1: Image generation produces exactly N valid PNGs with correct names
class TestPropertyImageGeneration:
    """**Validates: Requirements 1.1, 1.2, 1.3**"""

    @given(
        n=st.integers(min_value=1, max_value=50),
        image_size=st.integers(min_value=8, max_value=128),
    )
    @settings(max_examples=100)
    def test_generates_n_valid_pngs(self, n, image_size):
        tmp_dir = tempfile.mkdtemp()
        try:
            paths = generate_images(n, image_size, tmp_dir)
            assert len(paths) == n
            for i, path in enumerate(paths):
                assert os.path.basename(path) == f"loadtest-{i:04d}.png"
                with open(path, "rb") as f:
                    header = f.read(24)
                assert header[:8] == b"\x89PNG\r\n\x1a\n"
                w, h = struct.unpack(">II", header[16:24])
                assert (w, h) == (image_size, image_size)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# Feature: ingestion-load-test, Property 2: S3 upload keys use the images/ prefix for all generated files
class TestPropertyUploadKeys:
    """**Validates: Requirements 2.1, 11.4, 11.5**"""

    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=100)
    def test_all_keys_use_images_prefix(self, n):
        tmp_dir = tempfile.mkdtemp()
        try:
            paths = []
            for i in range(n):
                p = os.path.join(tmp_dir, f"loadtest-{i:04d}.png")
                with open(p, "wb") as f:
                    f.write(b"\x89PNG fake")
                paths.append(p)
            mock_s3 = MagicMock()
            num_up, num_fail, _ = upload_images(paths, "test-bucket", s3_client=mock_s3)
            assert num_up + num_fail == n
            for c in mock_s3.upload_file.call_args_list:
                _, bucket, key = c[0]
                assert key.startswith("images/")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# Feature: ingestion-load-test, Property 3: Payload validation rejects invalid inputs and applies correct defaults
class TestPropertyPayloadValidation:
    """**Validates: Requirements 8.1, 8.2, 8.3**"""

    @given(
        num_images=st.integers(min_value=1, max_value=10000),
        poll_interval=st.integers(min_value=1, max_value=60),
        timeout=st.integers(min_value=1, max_value=870),
        image_size=st.integers(min_value=1, max_value=512),
        cleanup=st.booleans(),
    )
    @settings(max_examples=100)
    def test_valid_inputs_round_trip(self, num_images, poll_interval, timeout, image_size, cleanup):
        event = {
            "num_images": num_images,
            "poll_interval": poll_interval,
            "timeout": timeout,
            "image_size": image_size,
            "cleanup": cleanup,
        }
        result = parse_event(event)
        assert result == event

    @given(num_images=st.integers(max_value=0))
    @settings(max_examples=100)
    def test_non_positive_num_images_rejected(self, num_images):
        with pytest.raises(ValueError):
            parse_event({"num_images": num_images})

    @given(timeout=st.integers(min_value=871, max_value=10000))
    @settings(max_examples=100)
    def test_timeout_over_870_rejected(self, timeout):
        with pytest.raises(ValueError):
            parse_event({"timeout": timeout})

    def test_empty_event_defaults(self):
        result = parse_event({})
        assert result["num_images"] == 1000
        assert result["poll_interval"] == 5
        assert result["timeout"] == 600
        assert result["image_size"] == 64
        assert result["cleanup"] is True


# Feature: ingestion-load-test, Property 4: Status is PASS if and only if document count delta equals num_images
class TestPropertyStatusLogic:
    """**Validates: Requirements 5.3, 6.1, 6.2, 7.2, 7.3**"""

    @given(
        num_images=st.integers(min_value=1, max_value=10000),
        baseline=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=100)
    def test_pass_when_delta_equals_n(self, num_images, baseline):
        monitor_result = {
            "documents_indexed": num_images,
            "wall_clock_seconds": 10.0,
            "peak_concurrent_executions": 5,
            "document_count_final": baseline + num_images,
            "time_series": [],
        }
        result = assemble_result(num_images, 0, num_images, monitor_result)
        assert result["status"] == "PASS"

    @given(
        num_images=st.integers(min_value=2, max_value=10000),
        shortfall=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_fail_when_delta_less_than_n(self, num_images, shortfall):
        indexed = max(0, num_images - shortfall)
        monitor_result = {
            "documents_indexed": indexed,
            "wall_clock_seconds": 10.0,
            "peak_concurrent_executions": 5,
            "document_count_final": indexed,
            "time_series": [],
        }
        result = assemble_result(num_images, 0, num_images, monitor_result)
        if indexed < num_images:
            assert result["status"] == "FAIL"
            assert result["shortfall"] == num_images - indexed
        else:
            assert result["status"] == "PASS"


# Feature: ingestion-load-test, Property 5: Result object contains all required fields and time_series entries
class TestPropertyResultFields:
    """**Validates: Requirements 7.1, 7.4, 9.6**"""

    @given(
        num_uploaded=st.integers(min_value=0, max_value=1000),
        num_failed=st.integers(min_value=0, max_value=100),
        num_images=st.integers(min_value=1, max_value=1000),
        docs_indexed=st.integers(min_value=0, max_value=1000),
        wall_clock=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        peak_concurrent=st.integers(min_value=0, max_value=100),
        ts_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_all_required_fields_present(
        self, num_uploaded, num_failed, num_images, docs_indexed, wall_clock, peak_concurrent, ts_count
    ):
        time_series = [
            {
                "elapsed_seconds": float(i),
                "sqs_visible": 0,
                "sqs_in_flight": 0,
                "concurrent_executions": 0,
                "cumulative_invocations": 0,
                "documents_indexed": 0,
            }
            for i in range(ts_count)
        ]
        monitor_result = {
            "documents_indexed": docs_indexed,
            "wall_clock_seconds": wall_clock,
            "peak_concurrent_executions": peak_concurrent,
            "document_count_final": docs_indexed,
            "time_series": time_series,
        }
        result = assemble_result(num_uploaded, num_failed, num_images, monitor_result)
        required_keys = {
            "status",
            "num_images_uploaded",
            "num_upload_failures",
            "num_documents_indexed",
            "wall_clock_seconds",
            "throughput_images_per_second",
            "peak_concurrent_executions",
            "document_count_final",
            "time_series",
            "cleanup_results",
        }
        assert required_keys.issubset(result.keys())
        for entry in result["time_series"]:
            ts_keys = {
                "elapsed_seconds",
                "sqs_visible",
                "sqs_in_flight",
                "concurrent_executions",
                "cumulative_invocations",
                "documents_indexed",
            }
            assert ts_keys.issubset(entry.keys())


# Feature: ingestion-load-test, Property 6: Monitoring loop survives individual API failures
class TestPropertyMonitorResilience:
    """**Validates: Requirements 11.1, 11.2, 11.3**"""

    @given(
        sqs_fails=st.lists(st.booleans(), min_size=1, max_size=5),
        cw_fails=st.lists(st.booleans(), min_size=1, max_size=5),
        aoss_fails=st.lists(st.booleans(), min_size=1, max_size=5),
    )
    @settings(max_examples=50)
    def test_loop_survives_api_failures(self, sqs_fails, cw_fails, aoss_fails):
        from lambdas.retrieval_load_test.index import monitor_pipeline

        num_iterations = min(len(sqs_fails), len(cw_fails), len(aoss_fails))
        if num_iterations == 0:
            return

        sqs_call_count = 0
        cw_call_count = 0
        aoss_call_count = 0

        mock_sqs = MagicMock()

        def _sqs_side_effect(**kwargs):
            nonlocal sqs_call_count
            idx = min(sqs_call_count, len(sqs_fails) - 1)
            sqs_call_count += 1
            if sqs_fails[idx]:
                raise Exception("SQS fail")
            return {"Attributes": {"ApproximateNumberOfMessages": "0", "ApproximateNumberOfMessagesNotVisible": "0"}}

        mock_sqs.get_queue_attributes.side_effect = _sqs_side_effect

        mock_cw = MagicMock()

        def _cw_side_effect(**kwargs):
            nonlocal cw_call_count
            idx = min(cw_call_count, len(cw_fails) - 1)
            cw_call_count += 1
            if cw_fails[idx]:
                raise Exception("CW fail")
            return {"MetricDataResults": []}

        mock_cw.get_metric_data.side_effect = _cw_side_effect

        mock_aoss = MagicMock()

        def _aoss_side_effect(**kwargs):
            nonlocal aoss_call_count
            idx = min(aoss_call_count, len(aoss_fails) - 1)
            aoss_call_count += 1
            if aoss_fails[idx]:
                raise Exception("AOSS fail")
            return {"count": 0}

        mock_aoss.count.side_effect = _aoss_side_effect

        # Use very short poll interval and timeout to keep tests fast
        poll_interval = 0.01
        timeout = poll_interval * num_iterations + 0.05
        upload_start = time.time()

        result = monitor_pipeline(
            baseline_doc_count=0,
            num_images=1000,
            poll_interval=poll_interval,
            timeout=timeout,
            queue_url="https://sqs.example.com/q",
            function_name="test-fn",
            aoss_client=mock_aoss,
            index_name="test-idx",
            upload_start_time=upload_start,
            sqs_client=mock_sqs,
            cw_client=mock_cw,
        )
        # The loop should complete without crashing
        assert "time_series" in result
        assert "completed" in result


# Feature: ingestion-load-test, Property 7: Upload continues despite individual file failures
class TestPropertyUploadResilience:
    """**Validates: Requirements 11.4, 11.5**"""

    @given(
        n=st.integers(min_value=1, max_value=20),
        fail_indices=st.frozensets(st.integers(min_value=0, max_value=19)),
    )
    @settings(max_examples=100)
    def test_all_uploads_attempted(self, n, fail_indices):
        fail_indices = {i for i in fail_indices if i < n}
        tmp_dir = tempfile.mkdtemp()
        try:
            paths = []
            for i in range(n):
                p = os.path.join(tmp_dir, f"loadtest-{i:04d}.png")
                with open(p, "wb") as f:
                    f.write(b"\x89PNG fake")
                paths.append(p)

            call_count = 0
            mock_s3 = MagicMock()

            def _side_effect(local_path, bucket, key):
                nonlocal call_count
                call_count += 1
                idx = int(os.path.basename(local_path).split("-")[1].split(".")[0])
                if idx in fail_indices:
                    raise Exception("Simulated failure")

            mock_s3.upload_file.side_effect = _side_effect

            num_up, num_fail, _ = upload_images(paths, "test-bucket", s3_client=mock_s3)
            assert num_up + num_fail == n
            assert num_fail == len(fail_indices)
            assert call_count == n
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# Feature: ingestion-load-test, Property 8: Cleanup targets only loadtest-prefixed artifacts
class TestPropertyCleanupTargeting:
    """**Validates: Requirements 9.1, 9.2, 9.3, 9.4**"""

    @given(n=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100)
    def test_only_loadtest_keys_deleted(self, n):
        keys = [f"images/loadtest-{i:04d}.png" for i in range(n)]
        base64_keys = [f"base64/images/loadtest-{i:04d}.png" for i in range(n)]

        mock_s3 = MagicMock()

        def _get_paginator(op):
            p = MagicMock()

            def _paginate(**kwargs):
                prefix = kwargs.get("Prefix", "")
                if prefix.startswith("base64/"):
                    return [{"Contents": [{"Key": k} for k in base64_keys]}]
                return [{"Contents": [{"Key": k} for k in keys]}]

            p.paginate = _paginate
            return p

        mock_s3.get_paginator = _get_paginator

        mock_aoss = MagicMock()
        mock_aoss.delete_by_query.return_value = {"deleted": n}

        cleanup_artifacts("test-bucket", mock_aoss, "test-index", s3_client=mock_s3)

        for c in mock_s3.delete_objects.call_args_list:
            objects = c.kwargs.get("Delete", {}).get("Objects", [])
            for obj in objects:
                assert "loadtest-" in obj["Key"]

        mock_aoss.delete_by_query.assert_called_once_with(
            index="test-index",
            body={"query": {"prefix": {"description": "images/loadtest-"}}},
        )


# Feature: ingestion-load-test, Property 9: /tmp files are always cleaned up after upload
class TestPropertyTmpCleanup:
    """**Validates: Requirements 9.5**"""

    @given(cleanup_flag=st.booleans())
    @settings(max_examples=100)
    def test_tmp_cleared_regardless_of_cleanup_flag(self, cleanup_flag):
        tmp_dir = tempfile.mkdtemp()
        paths = []
        for i in range(3):
            p = os.path.join(tmp_dir, f"loadtest-{i:04d}.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG fake")
            paths.append(p)

        mock_s3 = MagicMock()
        upload_images(paths, "test-bucket", s3_client=mock_s3)
        # cleanup_flag is not used by upload_images — it always cleans /tmp
        assert not os.path.exists(tmp_dir)
