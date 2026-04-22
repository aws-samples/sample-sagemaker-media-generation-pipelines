"""
CDK synthesis tests for setup processing jobs.

Verifies that when a pipeline config includes a `setup` key and retrieval
is enabled, the DataStack/RetrievalConstruct creates standalone setup
processing jobs with correct IAM policies, environment variables, ECR
repos, and AOSS data access policy wiring.

**Validates: Requirements 11.4, 11.5**
"""

import json
from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_lambda as lambda_

from config.config import (
    ContainerConfig,
    DynamoDBConfig,
    PipelineConfig,
    RetrievalConfig,
    SetupConfig,
)
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack
from tests.unit.conftest import PRIMARY_STEPS, _mock_from_asset

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_cfg() -> ContainerConfig:
    return ContainerConfig(
        InstanceCount=1,
        InstanceType="ml.g5.xlarge",
        VolumeSizeInGB=125,
        ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
        ContainerArguments=["300"],
    )


def _valid_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        collection_name="test-images",
        index_name="test-vectors",
        sqs_visibility_timeout_seconds=960,
        sqs_max_receive_count=3,
        ingest_lambda_timeout_seconds=300,
        ingest_lambda_memory_mb=2048,
    )


def _setup_config() -> SetupConfig:
    return SetupConfig(
        InstanceCount=1,
        InstanceType="ml.m5.xlarge",
        VolumeSizeInGB=50,
        ContainerEntrypoint=["python3", "main.py"],
        ContainerArguments=["--run"],
        dataset_url="1aurent/unsplash-lite",
        dataset_script="unsplash.py",
        num_prompts=100,
        test_image_count=500,
    )


def _pipeline_config_with_setup() -> PipelineConfig:
    return PipelineConfig(
        construct_id="dev",
        s3_downloads=[],
        steps={s: _default_cfg() for s in PRIMARY_STEPS},
        retrieval="retrieval.yaml",
        setup={"dataset_ingest": _setup_config()},
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures for CDK synthesis (expensive, run once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_stack_with_setup():
    """Synthesize a DataStack with retrieval + setup config once for the module."""
    pipeline_config = _pipeline_config_with_setup()
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        sec = SecurityStack(app, "SecStack", prefix="dev", env=env)
        data = DataStack(
            app,
            "DataStack",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=pipeline_config,
            prefix="dev",
            retrieval_config=_valid_retrieval_config(),
            env=env,
        )

    return data


@pytest.fixture(scope="module")
def data_template(data_stack_with_setup):
    """Return the CloudFormation template from the DataStack."""
    return assertions.Template.from_stack(data_stack_with_setup)


# ---------------------------------------------------------------------------
# Test: Setup ReusableProcessingJob is created inside RetrievalConstruct
# ---------------------------------------------------------------------------


class TestSetupJobCreatedInRetrievalConstruct:
    """Verify setup processing job is created inside RetrievalConstruct."""

    def test_retrieval_construct_has_setup_jobs(self, data_stack_with_setup):
        retrieval = data_stack_with_setup.retrieval
        assert retrieval is not None
        assert "dataset_ingest" in retrieval.setup_jobs

    def test_setup_job_has_trigger_lambda(self, data_stack_with_setup):
        retrieval = data_stack_with_setup.retrieval
        assert "dataset_ingest" in retrieval.setup_trigger_lambdas

    def test_trigger_lambda_function_exists(self, data_template):
        """A Lambda function for the setup job trigger exists in the template."""
        data_template.has_resource_properties(
            "AWS::Lambda::Function",
            assertions.Match.object_like(
                {
                    "FunctionName": "dev-dataset_ingest-trigger-Lambda",
                }
            ),
        )


# ---------------------------------------------------------------------------
# Test: IAM policies include Bedrock, S3, AOSS, SSM
# ---------------------------------------------------------------------------


class TestSetupJobIamPolicies:
    """Verify IAM policies for the setup job role."""

    def test_bedrock_policy_exists(self, data_template):
        """Supplemental Bedrock Claude policy is created."""
        data_template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            assertions.Match.object_like(
                {
                    "ManagedPolicyName": "dev-dataset-ingest-setup-bedrock-policy",
                    "PolicyDocument": assertions.Match.object_like(
                        {
                            "Statement": assertions.Match.array_with(
                                [
                                    assertions.Match.object_like(
                                        {
                                            "Action": [
                                                "bedrock:InvokeModel",
                                                "bedrock:InvokeModelWithResponseStream",
                                            ],
                                            "Effect": "Allow",
                                        }
                                    ),
                                ]
                            ),
                        }
                    ),
                }
            ),
        )

    def test_aoss_policy_exists(self, data_template):
        """AOSS APIAccessAll policy is created for the setup job."""
        data_template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            assertions.Match.object_like(
                {
                    "ManagedPolicyName": "dev-dataset-ingest-setup-aoss-policy",
                    "PolicyDocument": assertions.Match.object_like(
                        {
                            "Statement": assertions.Match.array_with(
                                [
                                    assertions.Match.object_like(
                                        {
                                            "Action": "aoss:APIAccessAll",
                                            "Effect": "Allow",
                                        }
                                    ),
                                ]
                            ),
                        }
                    ),
                }
            ),
        )

    def test_ecr_pull_policy_exists(self, data_template):
        """ECR pull policy is created for the setup job."""
        data_template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            assertions.Match.object_like(
                {
                    "ManagedPolicyName": "dev-dataset-ingest-setup-ecr-pull-policy",
                    "PolicyDocument": assertions.Match.object_like(
                        {
                            "Statement": assertions.Match.array_with(
                                [
                                    assertions.Match.object_like(
                                        {
                                            "Action": [
                                                "ecr:GetDownloadUrlForLayer",
                                                "ecr:BatchGetImage",
                                                "ecr:BatchCheckLayerAvailability",
                                            ],
                                            "Effect": "Allow",
                                        }
                                    ),
                                ]
                            ),
                        }
                    ),
                }
            ),
        )


# ---------------------------------------------------------------------------
# Test: Environment variables
# ---------------------------------------------------------------------------


class TestSetupJobEnvironmentVariables:
    """Verify environment variables on the setup processing job definition."""

    def test_env_vars_include_retrieval_bucket(self, data_stack_with_setup):
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        env = job.definition_model.Arguments.Environment
        assert "RETRIEVAL_BUCKET_NAME" in env
        assert "retrieval-images-bucket" in env["RETRIEVAL_BUCKET_NAME"]

    def test_env_vars_include_aoss_endpoint_ssm(self, data_stack_with_setup):
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        env = job.definition_model.Arguments.Environment
        assert "AOSS_ENDPOINT_SSM" in env
        assert env["AOSS_ENDPOINT_SSM"] == "/dev/retrieval/aoss-endpoint"

    def test_env_vars_include_aoss_index_name(self, data_stack_with_setup):
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        env = job.definition_model.Arguments.Environment
        assert env["AOSS_INDEX_NAME"] == "test-vectors"

    def test_env_vars_include_dataset_url(self, data_stack_with_setup):
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        env = job.definition_model.Arguments.Environment
        assert env["DATASET_URL"] == "1aurent/unsplash-lite"

    def test_env_vars_include_dataset_script(self, data_stack_with_setup):
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        env = job.definition_model.Arguments.Environment
        assert env["DATASET_SCRIPT"] == "unsplash.py"

    def test_env_vars_include_num_prompts(self, data_stack_with_setup):
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        env = job.definition_model.Arguments.Environment
        assert env["NUM_PROMPTS"] == "100"

    def test_env_vars_include_test_image_count(self, data_stack_with_setup):
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        env = job.definition_model.Arguments.Environment
        assert env["TEST_IMAGE_COUNT"] == "500"


# ---------------------------------------------------------------------------
# Test: AOSS data access policy includes setup job role ARN
# ---------------------------------------------------------------------------


class TestAossDataAccessPolicyIncludesSetupRole:
    """Verify the AOSS data access policy includes the setup job role ARN."""

    def test_data_access_policy_has_setup_role(self, data_template):
        """The AOSS data access policy Principal list includes the setup job role ARN."""
        resources = data_template.find_resources("AWS::OpenSearchServerless::AccessPolicy")
        assert len(resources) > 0

        for _logical_id, resource in resources.items():
            policy_str = resource["Properties"]["Policy"]
            # Policy is a JSON string (possibly with CFn intrinsics via Fn::Join)
            # For CDK synthesis, it's typically an Fn::Join with embedded Ref/GetAtt
            # We just verify the policy references the setup job role
            policy_json = json.dumps(policy_str)
            # The setup job role name contains "dataset-ingest-SetupJob"
            assert "dataset-ingest" in policy_json.lower() or "SetupJob" in policy_json


# ---------------------------------------------------------------------------
# Test: ECR repo naming for setup job
# ---------------------------------------------------------------------------


class TestSetupJobEcrRepo:
    """Verify ECR repo naming follows the expected pattern."""

    def test_ecr_image_uri_pattern(self, data_stack_with_setup):
        """The setup job uses the correct ECR image URI pattern."""
        job = data_stack_with_setup.retrieval.setup_jobs["dataset_ingest"]
        image_uri = job.definition_model.Arguments.AppSpecification.ImageUri
        assert "dev/processing/dataset-ingest" in image_uri
        assert image_uri.endswith(":latest")


# ---------------------------------------------------------------------------
# Test: Setup job is NOT part of SageMaker Pipeline
# ---------------------------------------------------------------------------


class TestSetupJobIsStandalone:
    """Verify setup job is standalone, not part of SageMaker Pipeline."""

    def test_no_sagemaker_pipeline_in_data_stack(self, data_template):
        """DataStack does not contain a SageMaker Pipeline resource."""
        data_template.resource_count_is("AWS::SageMaker::Pipeline", 0)

    def test_setup_job_not_in_pipeline_graph(self):
        """Setup job name is not in the pipeline_graph."""
        config = _pipeline_config_with_setup()
        assert "dataset_ingest" not in config.pipeline_graph
