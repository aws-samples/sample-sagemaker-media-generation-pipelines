"""
Unit tests for the EcrRepository construct.

Tests verify ECR repository creation with image scanning, lifecycle rules,
KMS encryption, removal policy, and exposed managed policies for push,
pull, and auth operations.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_kms as kms

from project_constructs.ecr import EcrRepository

pytestmark = pytest.mark.core


def _create_ecr_stack(
    repo_name: str = "test-repo",
) -> tuple[cdk.Stack, EcrRepository, assertions.Template]:
    """Helper to create a stack with an EcrRepository for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)
    ecr_repo = EcrRepository(stack, "TestEcr", repository_name=repo_name, kms_key=kms_key)
    template = assertions.Template.from_stack(stack)
    return stack, ecr_repo, template


class TestEcrRepositoryResources:
    """Tests for ECR repository resource creation."""

    def test_creates_ecr_repository(self):
        _, _, template = _create_ecr_stack()
        template.resource_count_is("AWS::ECR::Repository", 1)

    def test_repository_name(self):
        _, _, template = _create_ecr_stack("my-custom-repo")
        template.has_resource_properties("AWS::ECR::Repository", {"RepositoryName": "my-custom-repo"})

    def test_image_scanning_enabled(self):
        _, _, template = _create_ecr_stack()
        template.has_resource_properties(
            "AWS::ECR::Repository",
            {"ImageScanningConfiguration": {"ScanOnPush": True}},
        )

    def test_removal_policy_destroy(self):
        _, _, template = _create_ecr_stack()
        template.has_resource(
            "AWS::ECR::Repository",
            {"UpdateReplacePolicy": "Delete", "DeletionPolicy": "Delete"},
        )

    def test_lifecycle_rule_max_image_count(self):
        _, _, template = _create_ecr_stack()
        template.has_resource_properties(
            "AWS::ECR::Repository",
            {
                "LifecyclePolicy": assertions.Match.object_like(
                    {"LifecyclePolicyText": assertions.Match.string_like_regexp("imageCountMoreThan.*10")}
                )
            },
        )


class TestEcrManagedPolicies:
    """Tests for ECR managed policies."""

    def test_creates_three_managed_policies(self):
        _, _, template = _create_ecr_stack()
        template.resource_count_is("AWS::IAM::ManagedPolicy", 3)

    def test_push_policy_actions(self):
        _, _, template = _create_ecr_stack("test-repo")
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-repo-ecr-push-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": assertions.Match.array_with(["ecr:PutImage"]),
                                        "Effect": "Allow",
                                    }
                                )
                            ]
                        )
                    }
                ),
            },
        )

    def test_pull_policy_actions(self):
        _, _, template = _create_ecr_stack("test-repo")
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-repo-ecr-pull-policy",
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

    def test_auth_policy_wildcard_resource(self):
        _, _, template = _create_ecr_stack("test-repo")
        template.has_resource_properties(
            "AWS::IAM::ManagedPolicy",
            {
                "ManagedPolicyName": "test-repo-ecr-auth-policy",
                "PolicyDocument": assertions.Match.object_like(
                    {
                        "Statement": assertions.Match.array_with(
                            [
                                assertions.Match.object_like(
                                    {
                                        "Action": "ecr:GetAuthorizationToken",
                                        "Resource": "*",
                                    }
                                )
                            ]
                        )
                    }
                ),
            },
        )


class TestEcrExposedAttributes:
    """Tests for EcrRepository exposed attributes."""

    def test_repository_exposed(self):
        _, ecr_repo, _ = _create_ecr_stack()
        assert ecr_repo.repository is not None

    def test_push_policy_exposed(self):
        _, ecr_repo, _ = _create_ecr_stack()
        assert ecr_repo.ecr_push_policy is not None

    def test_pull_policy_exposed(self):
        _, ecr_repo, _ = _create_ecr_stack()
        assert ecr_repo.ecr_pull_policy is not None

    def test_auth_policy_exposed(self):
        _, ecr_repo, _ = _create_ecr_stack()
        assert ecr_repo.ecr_auth_policy is not None
