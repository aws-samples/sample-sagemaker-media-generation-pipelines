"""
Unit tests for the A2IStack.

Tests verify stack synthesis with mocked dependencies, resource creation
for active A2I configs, and cross-stack references.
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_lambda as lambda_

from config.config import (
    A2IConfig,
    ContainerConfig,
    DynamoDBConfig,
    LambdaStepConfig,
    PipelineConfig,
)
from infrastructure.a2i_stack import A2IStack
from infrastructure.data import DataStack
from infrastructure.security import SecurityStack
from tests.unit.conftest import PRIMARY_STEPS, _mock_from_asset

pytestmark = pytest.mark.core


def _valid_step() -> dict:
    return {
        "InstanceCount": 1,
        "InstanceType": "ml.g5.xlarge",
        "VolumeSizeInGB": 125,
        "ContainerEntrypoint": ["/bin/bash", "./run_job.sh"],
        "ContainerArguments": ["300"],
    }


def _pipeline_config_with_a2i() -> PipelineConfig:
    """PipelineConfig with one active A2I config referenced by a lambda_step."""
    return PipelineConfig(
        construct_id="dev",
        steps={s: ContainerConfig(**_valid_step()) for s in PRIMARY_STEPS},
        a2i={"review_t2v": A2IConfig(media_type="video")},
        lambda_steps={
            "submit_review": LambdaStepConfig(
                lambda_path="submit_a2i_review",
                a2i_name="review_t2v",
            )
        },
    )


def _create_a2i_stack() -> tuple[A2IStack, assertions.Template]:
    """Create SecurityStack + DataStack + A2IStack with all mocks applied."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")

    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        sec = SecurityStack(app, "SecStack", prefix="dev", env=env)
        pipeline_cfg = _pipeline_config_with_a2i()
        data = DataStack(
            app,
            "DataStack",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=pipeline_cfg,
            prefix="dev",
            env=env,
        )

        # Mock the boto3 call that discovers Cognito from existing workteams
        with patch("infrastructure.a2i_stack._discover_cognito_from_workteams", return_value=None):
            stack = A2IStack(
                app,
                "A2IStack",
                security_stack=sec,
                data_stack=data,
                pipeline_config=pipeline_cfg,
                prefix="dev",
                env=env,
            )

    template = assertions.Template.from_stack(stack)
    return stack, template


class TestA2IStackSynthesis:
    """A2IStack synthesizes without errors."""

    @pytest.fixture(scope="class")
    def stack_and_template(self):
        return _create_a2i_stack()

    def test_synthesizes_successfully(self, stack_and_template) -> None:
        stack, _ = stack_and_template
        assert stack is not None

    def test_creates_lambda_functions(self, stack_and_template) -> None:
        _, template = stack_and_template
        # Submit + Process Lambdas + AwsCustomResource internal Lambda(s)
        resources = template.find_resources("AWS::Lambda::Function")
        assert len(resources) >= 2

    def test_creates_eventbridge_rule(self, stack_and_template) -> None:
        _, template = stack_and_template
        template.resource_count_is("AWS::Events::Rule", 1)

    def test_creates_sns_topic(self, stack_and_template) -> None:
        _, template = stack_and_template
        template.resource_count_is("AWS::SNS::Topic", 1)


class TestA2IStackAttributes:
    """A2IStack exposes expected attributes."""

    def test_a2i_constructs_populated(self) -> None:
        stack, _ = _create_a2i_stack()
        assert "review_t2v" in stack.a2i_constructs

    def test_submit_lambdas_populated(self) -> None:
        stack, _ = _create_a2i_stack()
        assert "review_t2v" in stack.submit_lambdas


class TestA2IStackSkipsInactive:
    """A2I configs not referenced by lambda_steps are skipped."""

    def test_unreferenced_a2i_skipped(self) -> None:
        app = cdk.App()
        env = cdk.Environment(account="123456789012", region="us-east-1")

        cfg = PipelineConfig(
            construct_id="dev",
            steps={s: ContainerConfig(**_valid_step()) for s in PRIMARY_STEPS},
            a2i={"review_t2v": A2IConfig(media_type="video")},
            lambda_steps={},  # No lambda_step references the A2I config
        )

        with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
            sec = SecurityStack(app, "SecStack2", prefix="dev", env=env)
            data = DataStack(
                app,
                "DataStack2",
                security_stack=sec,
                dynamodb_config=DynamoDBConfig(),
                pipeline_config=cfg,
                prefix="dev",
                env=env,
            )
            with patch("infrastructure.a2i_stack._discover_cognito_from_workteams", return_value=None):
                stack = A2IStack(
                    app,
                    "A2IStack2",
                    security_stack=sec,
                    data_stack=data,
                    pipeline_config=cfg,
                    prefix="dev",
                    env=env,
                )

        assert len(stack.a2i_constructs) == 0
        assert len(stack.submit_lambdas) == 0
