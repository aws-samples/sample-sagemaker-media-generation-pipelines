"""
Unit tests for the CodeBuildStack.

Tests verify per-step ECR repositories, CodeBuild projects,
IAM policy wiring, and exposed attributes.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from infrastructure.cicd_pipeline.codebuild_stack import CodeBuildStack
from infrastructure.security import SecurityStack
from tests.unit.conftest import PRIMARY_STEPS, STEP_0, STEP_0_DASHED

pytestmark = pytest.mark.core


def _create_codebuild_stack(step_names=None):
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    sec = SecurityStack(app, "SecStack", env=env)
    kwargs = {"security_stack": sec, "step_names": step_names or PRIMARY_STEPS, "env": env}
    cb = CodeBuildStack(app, "CBStack", **kwargs)
    return cb, assertions.Template.from_stack(cb)


@pytest.fixture(scope="module")
def cb():
    """Default CodeBuildStack. Returns (stack, template)."""
    return _create_codebuild_stack()


class TestEcrRepositories:
    def test_ecr_repos_created(self, cb):
        cb[1].resource_count_is("AWS::ECR::Repository", len(PRIMARY_STEPS))

    def test_ecr_repo_names(self, cb):
        for step in PRIMARY_STEPS:
            cb[1].has_resource_properties(
                "AWS::ECR::Repository", {"RepositoryName": f"dev/processing/{step.replace('_', '-')}"}
            )

    def test_ecr_repos_have_image_scanning(self, cb):
        cb[1].has_resource_properties("AWS::ECR::Repository", {"ImageScanningConfiguration": {"ScanOnPush": True}})

    def test_ecr_repos_removal_policy_destroy(self, cb):
        cb[1].has_resource("AWS::ECR::Repository", {"UpdateReplacePolicy": "Delete", "DeletionPolicy": "Delete"})


class TestCodeBuildProjects:
    def test_codebuild_projects_created(self, cb):
        cb[1].resource_count_is("AWS::CodeBuild::Project", len(PRIMARY_STEPS))

    def test_codebuild_projects_privileged_mode(self, cb):
        cb[1].has_resource_properties(
            "AWS::CodeBuild::Project", {"Environment": assertions.Match.object_like({"PrivilegedMode": True})}
        )

    def test_codebuild_projects_in_vpc(self, cb):
        cb[1].has_resource_properties(
            "AWS::CodeBuild::Project",
            {"VpcConfig": assertions.Match.object_like({"VpcId": assertions.Match.any_value()})},
        )


class TestExposedAttributes:
    def test_ecr_repositories_dict_keyed_by_step(self, cb):
        for step in PRIMARY_STEPS:
            assert step in cb[0].ecr_repositories

    def test_codebuild_projects_dict_keyed_by_step(self, cb):
        for step in PRIMARY_STEPS:
            assert step in cb[0].codebuild_projects

    def test_ecr_repo_has_pull_policy(self, cb):
        assert cb[0].ecr_repositories[STEP_0].ecr_pull_policy is not None

    def test_ecr_repo_has_push_policy(self, cb):
        assert cb[0].ecr_repositories[STEP_0].ecr_push_policy is not None

    def test_ecr_repo_has_auth_policy(self, cb):
        assert cb[0].ecr_repositories[STEP_0].ecr_auth_policy is not None


class TestIamPolicies:
    def test_ecr_push_policy_created_per_step(self, cb):
        cb[1].has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": f"dev-processing-{STEP_0_DASHED}-ecr-push-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {"Action": assertions.Match.array_with(["ecr:PutImage"]), "Effect": "Allow"}
                                )
                            ]
                        )
                    }
                ),
            },
        )

    def test_ecr_pull_policy_created_per_step(self, cb):
        cb[1].has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": f"dev-processing-{STEP_0_DASHED}-ecr-pull-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": assertions.Match.array_with(["ecr:GetDownloadUrlForLayer"]),
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        )
                    }
                ),
            },
        )

    def test_ecr_auth_policy_has_wildcard_resource(self, cb):
        cb[1].has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": f"dev-processing-{STEP_0_DASHED}-ecr-auth-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [assertions.Match.object_like({"Action": "ecr:GetAuthorizationToken", "Resource": "*"})]
                        )
                    }
                ),
            },
        )


class TestScaling:
    def test_default_steps_creates_repos(self, cb):
        cb[1].resource_count_is("AWS::ECR::Repository", len(PRIMARY_STEPS))
        cb[1].resource_count_is("AWS::CodeBuild::Project", len(PRIMARY_STEPS))

    def test_single_step(self):
        _, t = _create_codebuild_stack(step_names=[STEP_0])
        t.resource_count_is("AWS::ECR::Repository", 1)
        t.resource_count_is("AWS::CodeBuild::Project", 1)
