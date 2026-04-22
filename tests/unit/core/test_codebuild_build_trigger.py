"""
Unit tests for CodeBuildStack Lambda-category ECR/CodeBuild resources.

The build trigger custom resource has been removed — container builds
are now triggered by the CI/CD pipeline's ContainerBuild stage.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infrastructure.cicd_pipeline.codebuild_stack import CodeBuildStack
from infrastructure.security import SecurityStack
from tests.unit.conftest import PRIMARY_STEPS

pytestmark = pytest.mark.core


def _create_stack(
    step_names: list[str] | None = None,
    lambda_names: list[str] | None = None,
    prefix: str = "dev",
) -> tuple[CodeBuildStack, assertions.Template]:
    """Helper to create SecurityStack + CodeBuildStack."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    sec = SecurityStack(app, "SecStack", env=env)
    stack = CodeBuildStack(
        app,
        "CBStack",
        security_stack=sec,
        step_names=step_names or PRIMARY_STEPS,
        lambda_names=lambda_names,
        prefix=prefix,
        env=env,
    )
    template = assertions.Template.from_stack(stack)
    return stack, template


class TestBuildTriggerRemoved:
    """Verify the build trigger custom resource no longer exists."""

    def test_no_custom_resource(self):
        _, template = _create_stack()
        template.resource_count_is("AWS::CloudFormation::CustomResource", 0)

    def test_no_trigger_lambda(self):
        _, template = _create_stack()
        # No Lambda functions should exist in CodeBuildStack anymore
        template.resource_count_is("AWS::Lambda::Function", 0)


class TestLambdaCategory:
    """Tests for Lambda-category ECR repos and CodeBuild projects."""

    def test_lambda_ecr_repos_created(self):
        _, template = _create_stack(lambda_names=["trigger_pipeline"])
        # len(PRIMARY_STEPS) processing + 1 lambda
        template.resource_count_is("AWS::ECR::Repository", len(PRIMARY_STEPS) + 1)

    def test_lambda_ecr_repo_names(self):
        _, template = _create_stack(lambda_names=["trigger_pipeline"])
        template.has_resource_properties(
            "AWS::ECR::Repository",
            {"RepositoryName": "dev/lambda/trigger-pipeline"},
        )

    def test_lambda_codebuild_projects_created(self):
        _, template = _create_stack(lambda_names=["trigger_pipeline"])
        # len(PRIMARY_STEPS) processing + 1 lambda
        template.resource_count_is("AWS::CodeBuild::Project", len(PRIMARY_STEPS) + 1)

    def test_lambda_exposed_attributes(self):
        stack, _ = _create_stack(lambda_names=["trigger_pipeline"])
        assert "trigger_pipeline" in stack.lambda_ecr_repositories
        assert "trigger_pipeline" in stack.lambda_codebuild_projects

    def test_no_lambda_names_defaults_empty(self):
        stack, _ = _create_stack()
        assert stack.lambda_ecr_repositories == {}
        assert stack.lambda_codebuild_projects == {}
