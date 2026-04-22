"""
Unit tests for the PipelineStack.

Tests verify config-driven step creation, trigger Lambda, IAM policies,
CfnOutputs, and CDK Nag compliance.

Validates: Requirements 1.4, 2.1, 4.1, 4.3, 4.5, 7.1, 7.7, 7.8, 8.10
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_lambda as lambda_

from config.config import ContainerConfig, DynamoDBConfig, PipelineConfig
from infrastructure.data import DataStack
from infrastructure.pipeline import PipelineStack
from infrastructure.security import SecurityStack
from tests.unit.conftest import PRIMARY_STEPS, STEP_0
from tests.unit.core.conftest import _mock_from_asset

pytestmark = pytest.mark.core


def _default_cfg() -> ContainerConfig:
    return ContainerConfig(
        InstanceCount=1,
        InstanceType="ml.g5.xlarge",
        VolumeSizeInGB=125,
        ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
        ContainerArguments=["300"],
        models_prefix=["test-models"],
    )


def _default_pipeline_config(step_names: list[str] | None = None) -> PipelineConfig:
    names = step_names or PRIMARY_STEPS
    return PipelineConfig(construct_id="dev", s3_downloads=[], steps={name: _default_cfg() for name in names})


def _create_pipeline_stack(step_names=None) -> tuple[PipelineStack, assertions.Template]:
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    pipeline_config = _default_pipeline_config(step_names)
    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        sec = SecurityStack(app, "SecStack", prefix="dev", env=env)
        data = DataStack(
            app,
            "DataStack",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=pipeline_config,
            prefix="dev",
            env=env,
        )
        stack = PipelineStack(
            app,
            "PipelineStack",
            security_stack=sec,
            data_stack=data,
            pipeline_config=pipeline_config,
            prefix="dev",
            env=env,
        )
    return stack, assertions.Template.from_stack(stack)


@pytest.fixture(scope="module")
def ps():
    """Synthesize default PipelineStack once for the module. Returns (stack, template)."""
    return _create_pipeline_stack()


class TestPipelineResource:
    def test_pipeline_resource_created(self, ps):
        ps[1].resource_count_is("AWS::SageMaker::Pipeline", 1)

    def test_pipeline_has_correct_name(self, ps):
        ps[1].has_resource_properties("AWS::SageMaker::Pipeline", {"PipelineName": "dev-sagemaker-pipeline"})


class TestConfigDrivenSteps:
    def test_default_creates_output_buckets(self, ps):
        ps[1].resource_count_is("AWS::S3::Bucket", len(PRIMARY_STEPS))

    def test_single_step_creates_one_output_bucket(self):
        _, t = _create_pipeline_stack(step_names=[STEP_0])
        t.resource_count_is("AWS::S3::Bucket", 1)

    def test_step_output_bucket_names_use_dashes(self, ps):
        for step in PRIMARY_STEPS:
            ps[1].has_resource_properties(
                "AWS::S3::Bucket",
                {"BucketName": assertions.Match.string_like_regexp(f".*{step.replace('_', '-')}-output-bucket")},
            )


class TestTriggerLambda:
    def test_trigger_lambda_created(self, ps):
        ps[1].has_resource_properties(
            "AWS::Lambda::Function", {"FunctionName": assertions.Match.string_like_regexp(".*pipeline-trigger.*")}
        )

    def test_trigger_lambda_has_pipeline_name_env_var(self, ps):
        ps[1].has_resource_properties(
            "AWS::Lambda::Function",
            {
                "FunctionName": assertions.Match.string_like_regexp(".*pipeline-trigger.*"),
                "Environment": {"Variables": assertions.Match.object_like({"PIPELINE_NAME": "dev-sagemaker-pipeline"})},
            },
        )

    def test_trigger_lambda_in_vpc(self, ps):
        ps[1].has_resource_properties(
            "AWS::Lambda::Function",
            {
                "FunctionName": assertions.Match.string_like_regexp(".*pipeline-trigger.*"),
                "VpcConfig": assertions.Match.object_like({"SubnetIds": assertions.Match.any_value()}),
            },
        )


class TestPipelineExecutionPolicy:
    def test_pipeline_execution_policy_created(self, ps):
        ps[1].has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "dev-pipeline-execution-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {"Action": "sagemaker:StartPipelineExecution", "Effect": "Allow"}
                                )
                            ]
                        )
                    }
                ),
            },
        )


class TestPipelineExecutionRole:
    def test_pipeline_execution_role_created(self, ps):
        ps[1].has_resource_properties(
            "AWS::IAM::Role",
            {
                "RoleName": "dev-pipeline-execution-role",
                "AssumeRolePolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [assertions.Match.object_like({"Principal": {"Service": "sagemaker.amazonaws.com"}})]
                        )
                    }
                ),
            },
        )

    def test_pass_role_policy_created(self, ps):
        ps[1].has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "dev-pipeline-pass-role-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [assertions.Match.object_like({"Action": "iam:PassRole", "Effect": "Allow"})]
                        )
                    }
                ),
            },
        )


class TestCfnOutputs:
    def test_pipeline_arn_output(self, ps):
        ps[1].has_output("devPipelineArn", {})

    def test_trigger_lambda_name_output(self, ps):
        ps[1].has_output("devTriggerLambdaName", {})


class TestExposedAttributes:
    def test_pipeline_construct_attribute(self, ps):
        assert ps[0].pipeline_construct is not None

    def test_trigger_lambda_attribute(self, ps):
        assert ps[0].trigger_lambda is not None

    def test_pipeline_execution_policy_attribute(self, ps):
        assert ps[0].pipeline_execution_policy is not None


class TestNoModelDownloadProcessingJob:
    def test_no_model_download_processing_job(self, ps):
        from project_constructs.processing_job.main import ReusableProcessingJob

        for child in ps[0].node.find_all():
            if isinstance(child, ReusableProcessingJob):
                assert "model-download" not in child.definition_model.Name
                assert "model_download" not in child.definition_model.Name


class TestNoModelDownloadTriggerResources:
    def test_no_model_download_trigger_lambda_output(self, ps):
        for output_id in ps[1].find_outputs("*"):
            assert "ModelDownloadTrigger" not in output_id

    def test_no_model_download_project_name_output(self, ps):
        for output_id in ps[1].find_outputs("*"):
            assert "ModelDownloadProjectName" not in output_id


class TestModelsBucketInput:
    def test_processing_job_definition_contains_models_input(self, ps):
        from project_constructs.processing_job.main import ReusableProcessingJob

        jobs_found = 0
        for child in ps[0].node.find_all():
            if isinstance(child, ReusableProcessingJob):
                if "model-download" in child.definition_model.Name:
                    continue
                input_names = [i.InputName for i in child.definition_model.Arguments.ProcessingInputs]
                assert "models" in input_names, f"Job {child.definition_model.Name} missing 'models' input channel"
                jobs_found += 1
        assert jobs_found >= 1

    def test_models_input_maps_to_correct_local_path(self, ps):
        from project_constructs.processing_job.main import ReusableProcessingJob

        for child in ps[0].node.find_all():
            if isinstance(child, ReusableProcessingJob):
                if "model-download" in child.definition_model.Name:
                    continue
                models_input = [
                    i for i in child.definition_model.Arguments.ProcessingInputs if i.InputName == "models"
                ][0]
                assert models_input.S3Input.LocalPath == "/opt/ml/processing/input/models/"

    def test_processing_jobs_have_dynamodb_env_var(self, ps):
        from project_constructs.processing_job.main import ReusableProcessingJob

        for child in ps[0].node.find_all():
            if isinstance(child, ReusableProcessingJob):
                if "model-download" in child.definition_model.Name:
                    continue
                env = child.definition_model.Arguments.Environment
                assert "DYNAMODB_TABLE_NAME" in env
                assert "STEP_NAME" in env
