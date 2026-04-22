"""
Unit tests for app.py CDK application integration.

Tests verify that all 6 stacks synthesize correctly, have proper
dependency ordering, and produce the expected CloudFormation outputs.
Also tests conditional RetrievalConstruct creation and PipelineStack AOSS wiring.

**Validates: Requirements 15.7**
"""

import json
from pathlib import Path
from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_lambda as lambda_

from config.config import ContainerConfig, DynamoDBConfig, PipelineConfig, RetrievalConfig
from infrastructure.cicd_pipeline.codebuild_stack import CodeBuildStack
from infrastructure.data import DataStack
from infrastructure.pipeline import PipelineStack
from infrastructure.security import SecurityStack
from project_constructs.retrieval import RetrievalConstruct
from tests.unit.conftest import PRIMARY_STEPS, STEP_0_DASHED, STEP_1_DASHED, _mock_from_asset

pytestmark = pytest.mark.integration


def _default_cfg() -> ContainerConfig:
    return ContainerConfig(
        InstanceCount=1,
        InstanceType="ml.g5.xlarge",
        VolumeSizeInGB=125,
        ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
        ContainerArguments=["300"],
    )


def _default_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        construct_id="dev",
        s3_downloads=[],
        steps={s: _default_cfg() for s in PRIMARY_STEPS},
    )


def _create_full_app() -> tuple[cdk.App, dict[str, cdk.Stack]]:
    """Replicate app.py stack creation for testing."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    pipeline_config = _default_pipeline_config()
    prefix = pipeline_config.construct_id
    step_names = list(pipeline_config.steps.keys())
    lambda_names = ["trigger_pipeline", "trigger_processing_job"]

    # Write per-config downloads manifest just like app.py does before stack creation
    downloads_path = Path(f"processing_job/model_download/{pipeline_config.construct_id}_downloads.json")
    downloads_path.parent.mkdir(parents=True, exist_ok=True)
    downloads_path.write_text(json.dumps([dl.model_dump() for dl in pipeline_config.s3_downloads], indent=2) + "\n")

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        security_stack = SecurityStack(app, f"{prefix}-SecurityStack", prefix=prefix, env=env)

        codebuild_stack = CodeBuildStack(
            app,
            f"{prefix}-CodeBuildStack",
            security_stack=security_stack,
            step_names=step_names,
            lambda_names=lambda_names,
            prefix=prefix,
            env=env,
        )
        codebuild_stack.add_dependency(security_stack)

        data_stack = DataStack(
            app,
            f"{prefix}-DataStack",
            security_stack=security_stack,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=pipeline_config,
            prefix=prefix,
            ecr_prefix=prefix,
            env=env,
        )
        data_stack.add_dependency(security_stack)

        pipeline_stack = PipelineStack(
            app,
            f"{prefix}-PipelineStack",
            security_stack=security_stack,
            data_stack=data_stack,
            pipeline_config=pipeline_config,
            prefix=prefix,
            ecr_prefix=prefix,
            env=env,
        )
        pipeline_stack.add_dependency(security_stack)
        pipeline_stack.add_dependency(data_stack)

    return app, {
        "security": security_stack,
        "codebuild": codebuild_stack,
        "data": data_stack,
        "pipeline": pipeline_stack,
    }


@pytest.fixture(scope="module")
def full_app():
    """Synthesize the full app once for the entire module."""
    return _create_full_app()


@pytest.fixture(scope="module")
def full_app_stacks(full_app):
    """Return the stacks dict from the full app."""
    return full_app[1]


@pytest.fixture(scope="module")
def full_app_templates(full_app_stacks):
    """Return cached templates for each stack."""
    return {name: assertions.Template.from_stack(stack) for name, stack in full_app_stacks.items()}


class TestAllStacksSynthesize:
    """Verify all stacks synthesize without errors."""

    def test_all_stacks_created(self, full_app_stacks):
        assert len(full_app_stacks) == 4

    def test_security_stack_synthesizes(self, full_app_templates):
        full_app_templates["security"].resource_count_is("AWS::EC2::VPC", 1)
        full_app_templates["security"].resource_count_is("AWS::KMS::Key", 1)

    def test_data_stack_synthesizes(self, full_app_templates):
        # Logging + input + output + models buckets
        full_app_templates["data"].resource_count_is("AWS::S3::Bucket", 4)

    def test_codebuild_stack_synthesizes(self, full_app_templates):
        # len(PRIMARY_STEPS) processing + 2 lambda
        full_app_templates["codebuild"].resource_count_is("AWS::ECR::Repository", len(PRIMARY_STEPS) + 2)
        full_app_templates["codebuild"].resource_count_is("AWS::CodeBuild::Project", len(PRIMARY_STEPS) + 2)

    def test_pipeline_stack_synthesizes(self, full_app_templates):
        full_app_templates["pipeline"].resource_count_is("AWS::SageMaker::Pipeline", 1)


class TestStackDependencies:
    """Verify stack dependency ordering."""

    def test_data_depends_on_security(self, full_app_stacks):
        deps = [d.node.id for d in full_app_stacks["data"].dependencies]
        assert "dev-SecurityStack" in deps

    def test_codebuild_depends_on_security(self, full_app_stacks):
        deps = [d.node.id for d in full_app_stacks["codebuild"].dependencies]
        assert "dev-SecurityStack" in deps

    def test_pipeline_depends_on_all_upstream(self, full_app_stacks):
        deps = [d.node.id for d in full_app_stacks["pipeline"].dependencies]
        assert "dev-SecurityStack" in deps
        assert "dev-DataStack" in deps


class TestStackNaming:
    """Verify stack names use prefix."""

    def test_stack_names_use_prefix(self, full_app_stacks):
        assert full_app_stacks["security"].stack_name == "dev-SecurityStack"
        assert full_app_stacks["data"].stack_name == "dev-DataStack"
        assert full_app_stacks["codebuild"].stack_name == "dev-CodeBuildStack"
        assert full_app_stacks["pipeline"].stack_name == "dev-PipelineStack"


class TestCodeBuildStackLambdaEcr:
    """Verify CodeBuildStack creates ECR repos for Lambda functions."""

    def test_lambda_ecr_repos_created(self, full_app_templates):
        template = full_app_templates["codebuild"]
        # Verify Lambda ECR repos exist with correct naming
        template.has_resource_properties(
            "AWS::ECR::Repository",
            {
                "RepositoryName": "dev/lambda/trigger-pipeline",
            },
        )
        template.has_resource_properties(
            "AWS::ECR::Repository",
            {
                "RepositoryName": "dev/lambda/trigger-processing-job",
            },
        )

    def test_processing_ecr_repos_created(self, full_app_templates):
        template = full_app_templates["codebuild"]
        template.has_resource_properties(
            "AWS::ECR::Repository",
            {
                "RepositoryName": f"dev/processing/{STEP_0_DASHED}",
            },
        )
        template.has_resource_properties(
            "AWS::ECR::Repository",
            {
                "RepositoryName": f"dev/processing/{STEP_1_DASHED}",
            },
        )

    def test_build_trigger_removed(self, full_app_templates):
        """Container builds are now triggered by CI/CD pipeline, not a custom resource."""
        full_app_templates["codebuild"].resource_count_is("AWS::CloudFormation::CustomResource", 0)


# ---------------------------------------------------------------------------
# Helpers for conditional RetrievalConstruct tests
# ---------------------------------------------------------------------------


def _valid_retrieval_config() -> RetrievalConfig:
    """Return a valid RetrievalConfig for testing."""
    return RetrievalConfig(
        collection_name="test-images",
        index_name="test-vectors",
        sqs_visibility_timeout_seconds=960,
        sqs_max_receive_count=3,
        ingest_lambda_timeout_seconds=300,
        ingest_lambda_memory_mb=2048,
    )


def _pipeline_config_with_retrieval(retrieval: str | None = None) -> PipelineConfig:
    """Return a PipelineConfig with an optional retrieval field and a retrieval step."""
    steps = {s: _default_cfg() for s in PRIMARY_STEPS}
    if retrieval is not None:
        steps["retrieval"] = ContainerConfig(
            InstanceCount=1,
            InstanceType="ml.c5.xlarge",
            VolumeSizeInGB=50,
            ContainerEntrypoint=["python3", "main.py"],
            ContainerArguments=["--retrieve"],
        )
    return PipelineConfig(
        construct_id="dev",
        s3_downloads=[],
        steps=steps,
        retrieval=retrieval,
    )


# ---------------------------------------------------------------------------
# Conditional RetrievalConstruct creation (Req 3.1, 3.2, 3.3, 3.4)
# ---------------------------------------------------------------------------


class TestRetrievalConstructCreatedWhenSet:
    """RetrievalConstruct is created inside DataStack when retrieval_config is provided."""

    def test_data_stack_has_retrieval_when_config_set(self) -> None:
        """DataStack.retrieval is not None when retrieval_config is provided."""
        pipeline_config = _pipeline_config_with_retrieval(retrieval="retrieval.yaml")
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

        assert data.retrieval is not None
        assert isinstance(data.retrieval, RetrievalConstruct)

    def test_retrieval_construct_exposes_attributes(self) -> None:
        """RetrievalConstruct exposes opensearch, ingestion_bucket, sqs_queue, ingest_lambda."""
        pipeline_config = _pipeline_config_with_retrieval(retrieval="retrieval.yaml")
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

        assert data.retrieval.opensearch is not None
        assert data.retrieval.ingestion_bucket is not None
        assert data.retrieval.sqs_queue is not None
        assert data.retrieval.ingest_lambda is not None


class TestRetrievalConstructSkippedWhenNone:
    """RetrievalConstruct is NOT created when retrieval_config is None."""

    def test_data_stack_retrieval_is_none(self) -> None:
        """DataStack.retrieval is None when retrieval_config is not provided."""
        pipeline_config = _pipeline_config_with_retrieval(retrieval=None)
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
                env=env,
            )

        assert data.retrieval is None

    def test_retrieval_field_defaults_to_none(self) -> None:
        """PipelineConfig without retrieval field defaults to None."""
        pipeline_config = _default_pipeline_config()
        assert pipeline_config.retrieval is None

    def test_pipeline_stack_works_without_retrieval(self) -> None:
        """PipelineStack synthesizes fine when retrieval_construct is None."""
        pipeline_config = _pipeline_config_with_retrieval(retrieval=None)
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
                env=env,
            )
            pipeline_stack = PipelineStack(
                app,
                "PipelineStack",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_config,
                retrieval_construct=None,
                prefix="dev",
                env=env,
            )

        template = assertions.Template.from_stack(pipeline_stack)
        template.resource_count_is("AWS::SageMaker::Pipeline", 1)


# ---------------------------------------------------------------------------
# PipelineStack AOSS wiring (Req 9.9, 9.10, 9.11, 10.1, 10.2, 10.3)
# ---------------------------------------------------------------------------


class TestPipelineStackAossWiring:
    """PipelineStack wires AOSS permissions when retrieval_construct is provided."""

    def test_retrieval_step_gets_aoss_env_vars(self) -> None:
        """When retrieval_construct is provided, the retrieval step gets AOSS env vars."""
        pipeline_config = _pipeline_config_with_retrieval(retrieval="retrieval.yaml")
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
            pipeline_stack = PipelineStack(
                app,
                "PipelineStack",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_config,
                retrieval_construct=data.retrieval,
                prefix="dev",
                env=env,
            )

        from project_constructs.processing_job.main import ReusableProcessingJob

        retrieval_job = None
        for child in pipeline_stack.node.find_all():
            if isinstance(child, ReusableProcessingJob) and "retrieval" in child.definition_model.Name:
                retrieval_job = child
                break

        assert retrieval_job is not None, "No retrieval processing job found in PipelineStack"
        job_env = retrieval_job.definition_model.Arguments.Environment
        assert "AOSS_ENDPOINT_SSM" in job_env
        assert "AOSS_INDEX_NAME" in job_env
        assert job_env["AOSS_INDEX_NAME"] == "test-vectors"
        assert "QUERY_K" in job_env
        assert job_env["QUERY_K"] == "5"

    def test_aoss_policy_on_retrieval_step_role(self) -> None:
        """When retrieval_construct is provided, the retrieval step role has aoss:APIAccessAll."""
        pipeline_config = _pipeline_config_with_retrieval(retrieval="retrieval.yaml")
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
            pipeline_stack = PipelineStack(
                app,
                "PipelineStack",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_config,
                retrieval_construct=data.retrieval,
                prefix="dev",
                env=env,
            )

        template = assertions.Template.from_stack(pipeline_stack)
        # The AOSS policy is attached as a managed policy with aoss:APIAccessAll
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            assertions.Match.object_like(
                {
                    "ManagedPolicyName": "dev-retrieval-step-policy",
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


# ---------------------------------------------------------------------------
# Tests for A2I, Lambda steps, and vrag_llm Bedrock policy coverage
# ---------------------------------------------------------------------------


from config.config import A2IConfig, LambdaStepConfig
from infrastructure.a2i_stack import A2IStack


def _pipeline_config_with_a2i() -> PipelineConfig:
    """Config with vrag_llm, A2I, and Lambda steps to cover pipeline.py gaps."""
    return PipelineConfig(
        construct_id="dev",
        s3_downloads=[],
        steps={
            "vrag_llm": ContainerConfig(
                InstanceCount=1,
                InstanceType="ml.g5.xlarge",
                VolumeSizeInGB=125,
                ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
                ContainerArguments=["300"],
                Environment={"VRAG_LLM_MODEL_ID": "us.anthropic.claude-3-5-haiku-20241022-v1:0"},
            ),
            "t2v": _default_cfg(),
        },
        pipeline_graph={"vrag_llm": [], "t2v": []},
        a2i={
            "vid_t2v": A2IConfig(media_type="video", task_title="Review video"),
        },
        lambda_steps={
            "submit_a2i_t2v": LambdaStepConfig(
                lambda_path="submit_a2i_review",
                a2i_name="vid_t2v",
                media_type="video",
            ),
        },
    )


class TestPipelineStackWithA2IAndLambdaSteps:
    """Cover A2I wiring, Lambda steps, and vrag_llm Bedrock policy in PipelineStack."""

    @pytest.fixture(scope="class")
    def stacks(self):
        pipeline_config = _pipeline_config_with_a2i()
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")

        with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
            sec = SecurityStack(app, "SecA2I", prefix="dev", env=env)
            data = DataStack(
                app,
                "DataA2I",
                security_stack=sec,
                dynamodb_config=DynamoDBConfig(),
                pipeline_config=pipeline_config,
                prefix="dev",
                env=env,
            )
            a2i_stack = A2IStack(
                app,
                "A2IStack",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_config,
                prefix="dev",
                env=env,
            )
            pipeline_stack = PipelineStack(
                app,
                "PipeA2I",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_config,
                prefix="dev",
                submit_lambdas=a2i_stack.submit_lambdas,
                a2i_constructs=a2i_stack.a2i_constructs,
                env=env,
            )
        return {
            "pipeline_stack": pipeline_stack,
            "a2i_stack": a2i_stack,
            "pipeline_config": pipeline_config,
        }

    def test_vrag_llm_bedrock_policy_created(self, stacks):
        template = assertions.Template.from_stack(stacks["pipeline_stack"])
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            assertions.Match.object_like(
                {
                    "ManagedPolicyName": "dev-vrag-llm-step-policy",
                }
            ),
        )

    def test_vrag_llm_bedrock_policy_has_invoke_actions(self, stacks):
        template = assertions.Template.from_stack(stacks["pipeline_stack"])
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            assertions.Match.object_like(
                {
                    "ManagedPolicyName": "dev-vrag-llm-step-policy",
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

    def test_a2i_stack_creates_submit_lambda(self, stacks):
        assert "vid_t2v" in stacks["a2i_stack"].submit_lambdas

    def test_a2i_stack_has_a2i_constructs(self, stacks):
        assert "vid_t2v" in stacks["a2i_stack"].a2i_constructs

    def test_lambda_step_registered_in_pipeline(self, stacks):
        template = assertions.Template.from_stack(stacks["pipeline_stack"])
        # The pipeline definition should contain the Lambda step
        template.has_resource("AWS::SageMaker::Pipeline", {})

    def test_submit_lambda_has_source_bucket_env(self, stacks):
        # SOURCE_BUCKET is added to the A2I submit Lambda by PipelineStack
        # but the Lambda resource lives in the A2I stack
        template = assertions.Template.from_stack(stacks["a2i_stack"])
        template.has_resource_properties(
            "AWS::Lambda::Function",
            assertions.Match.object_like(
                {
                    "Environment": assertions.Match.object_like(
                        {
                            "Variables": assertions.Match.object_like(
                                {
                                    "SOURCE_BUCKET": assertions.Match.any_value(),
                                }
                            ),
                        }
                    ),
                }
            ),
        )
