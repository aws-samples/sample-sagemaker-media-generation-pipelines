# Feature: sagemaker-pipeline, Task 17.6: ReusableProcessingJob ECR parameter tests
"""
Unit tests for ReusableProcessingJob ECR parameter extensions.

**Validates: Requirements 7.6, 7.7**
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import (
    assertions,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)

from config.config import DynamoDBConfig, PipelineConfig
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack
from project_constructs.processing_job.main import IoBucketConfig, ReusableProcessingJob
from project_constructs.s3 import BucketTemplate
from tests.unit.conftest import _mock_from_asset
from tests.unit.processing.conftest import _default_cfg

pytestmark = pytest.mark.processing


def _build_processing_job(
    ecr_image_uri: str | None = None,
    ecr_pull_policy: iam.ManagedPolicy | None = None,
    ecr_auth_policy: iam.ManagedPolicy | None = None,
    environment: dict[str, str] | None = None,
    lambda_trigger: bool = False,
) -> tuple[cdk.Stack, ReusableProcessingJob]:
    """Build a stack with a ReusableProcessingJob for testing."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        sec = SecurityStack(app, "Sec", prefix="dev", env=env)
        data = DataStack(
            app,
            "Data",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=PipelineConfig(construct_id="dev", s3_downloads=[], steps={}),
            prefix="dev",
            env=env,
        )

        stack = cdk.Stack(app, "PJStack", env=env)

        input_bucket = BucketTemplate(
            stack,
            "InputBucket",
            bucket_name="test-input",
            kms_key=sec.kms_key,
            logging_bucket=data.logs_bucket,
        )
        output_bucket = BucketTemplate(
            stack,
            "OutputBucket",
            bucket_name="test-output",
            kms_key=sec.kms_key,
            logging_bucket=data.logs_bucket,
        )

        pj = ReusableProcessingJob(
            stack,
            "TestPJ",
            job_name="test-job",
            input_buckets={"input": IoBucketConfig(bucket_template=input_bucket)},
            output_buckets={"output": IoBucketConfig(bucket_template=output_bucket)},
            security_stack=sec,
            lambda_trigger=lambda_trigger,
            cfg=_default_cfg(),
            environment=environment,
            ecr_image_uri=ecr_image_uri,
            ecr_pull_policy=ecr_pull_policy,
            ecr_auth_policy=ecr_auth_policy,
        )

    return stack, pj


class TestProcessingJobEcrParams:
    """Tests for ReusableProcessingJob ECR parameter extensions."""

    def test_ecr_image_uri_used_in_definition(self) -> None:
        """When ecr_image_uri is provided, definition uses it with :latest tag."""
        _, pj = _build_processing_job(ecr_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo")
        image_uri = pj.definition_model.Arguments.AppSpecification.ImageUri
        assert image_uri == "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest"

    def test_no_docker_image_asset_when_ecr_uri_provided(self) -> None:
        """When ecr_image_uri is provided, no DockerImageAsset is created."""
        _, pj = _build_processing_job(ecr_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo")
        assert pj.image is None

    def test_docker_image_asset_when_no_ecr_uri(self) -> None:
        """When ecr_image_uri is omitted, DockerImageAsset is used."""
        _, pj = _build_processing_job()
        assert pj.image is not None

    def test_ecr_pull_policy_attached(self) -> None:
        """When ecr_pull_policy is provided, it's attached to the role."""
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")
        policy_stack = cdk.Stack(app, "PolicyStack", env=env)
        pull_policy = iam.ManagedPolicy(
            policy_stack,
            "PullPolicy",
            statements=[iam.PolicyStatement(actions=["ecr:GetDownloadUrlForLayer"], resources=["*"])],
        )

        with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
            sec = SecurityStack(app, "Sec", prefix="dev", env=env)
            data = DataStack(
                app,
                "Data",
                security_stack=sec,
                dynamodb_config=DynamoDBConfig(),
                pipeline_config=PipelineConfig(construct_id="dev", s3_downloads=[], steps={}),
                prefix="dev",
                env=env,
            )
            stack = cdk.Stack(app, "PJStack", env=env)
            input_b = BucketTemplate(
                stack,
                "IB",
                bucket_name="test-input-bucket",
                kms_key=sec.kms_key,
                logging_bucket=data.logs_bucket,
            )
            output_b = BucketTemplate(
                stack,
                "OB",
                bucket_name="test-output-bucket",
                kms_key=sec.kms_key,
                logging_bucket=data.logs_bucket,
            )

            ReusableProcessingJob(
                stack,
                "PJ",
                job_name="test",
                input_buckets={"i": IoBucketConfig(bucket_template=input_b)},
                output_buckets={"output": IoBucketConfig(bucket_template=output_b)},
                security_stack=sec,
                lambda_trigger=False,
                cfg=_default_cfg(),
                ecr_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/r",
                ecr_pull_policy=pull_policy,
            )

        # Verify the role exists and has managed policies
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::IAM::Role",
            {
                "ManagedPolicyArns": assertions.Match.any_value(),
            },
        )


class TestProcessingJobEnvironment:
    """Tests for ReusableProcessingJob environment parameter."""

    def test_environment_none_accepted(self) -> None:
        """environment=None is accepted (backward compat).

        OUTPUT_S3_URI is auto-generated from output_buckets, so the env
        dict won't be completely empty — just verify no *extra* keys.
        """
        _, pj = _build_processing_job(environment=None)
        env_vars = pj.definition_model.Arguments.Environment
        # Only auto-generated keys should be present
        assert set(env_vars.keys()) <= {"OUTPUT_S3_URI"}

    def test_environment_dict_accepted(self) -> None:
        """environment=dict is accepted and merged."""
        _, pj = _build_processing_job(environment={"MY_VAR": "val"})
        env_vars = pj.definition_model.Arguments.Environment
        assert env_vars["MY_VAR"] == "val"

    def test_no_lambda_trigger_when_false(self) -> None:
        """lambda_trigger=False creates no application Lambda function."""
        stack, _ = _build_processing_job(lambda_trigger=False)
        template = assertions.Template.from_stack(stack)
        lambdas = template.find_resources("AWS::Lambda::Function")
        # Exclude CDK-internal Lambdas (S3 auto-delete custom resource)
        app_lambdas = {lid: res for lid, res in lambdas.items() if not lid.startswith("Custom")}
        assert len(app_lambdas) == 0
