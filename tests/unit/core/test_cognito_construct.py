"""
Unit tests for the CognitoWorkforceTemplate construct.

Tests verify that the Cognito construct creates a user pool with
password policy, MFA, client, domain, and reviewer group — or
imports an existing pool when IDs are provided.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from project_constructs.cognito import CognitoWorkforceTemplate

pytestmark = pytest.mark.core


def _create_cognito_stack(
    existing_user_pool_id: str = "",
    existing_client_id: str = "",
) -> tuple[cdk.Stack, CognitoWorkforceTemplate]:
    """Helper to create a stack with a CognitoWorkforceTemplate for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))

    cognito = CognitoWorkforceTemplate(
        stack,
        "TestCognito",
        pool_name="test-reviewers",
        group_name="test-group",
        existing_user_pool_id=existing_user_pool_id,
        existing_client_id=existing_client_id,
        domain_prefix="test-workforce",
    )
    return stack, cognito


class TestCognitoNewPool:
    """Tests for CognitoWorkforceTemplate creating a new user pool."""

    def test_creates_user_pool(self):
        stack, _ = _create_cognito_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPool", 1)

    def test_user_pool_has_correct_name(self):
        stack, _ = _create_cognito_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPool",
            {
                "UserPoolName": "test-reviewers",
            },
        )

    def test_user_pool_disables_self_signup(self):
        stack, _ = _create_cognito_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPool",
            {
                "AdminCreateUserConfig": assertions.Match.object_like(
                    {
                        "AllowAdminCreateUserOnly": True,
                    }
                ),
            },
        )

    def test_user_pool_has_password_policy(self):
        stack, _ = _create_cognito_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPool",
            {
                "Policies": assertions.Match.object_like(
                    {
                        "PasswordPolicy": assertions.Match.object_like(
                            {
                                "MinimumLength": 8,
                                "RequireUppercase": True,
                                "RequireLowercase": True,
                                "RequireNumbers": True,
                                "RequireSymbols": True,
                            }
                        ),
                    }
                ),
            },
        )

    def test_creates_user_pool_client(self):
        stack, _ = _create_cognito_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPoolClient", 1)

    def test_creates_user_pool_domain(self):
        stack, _ = _create_cognito_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPoolDomain", 1)

    def test_creates_reviewer_group(self):
        stack, _ = _create_cognito_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPoolGroup",
            {
                "GroupName": "test-group",
            },
        )


class TestCognitoExistingPool:
    """Tests for CognitoWorkforceTemplate importing an existing pool."""

    def test_does_not_create_new_pool(self):
        stack, _ = _create_cognito_stack(
            existing_user_pool_id="us-east-1_EXISTING",
            existing_client_id="existing-client-id",
        )
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::Cognito::UserPool", 0)

    def test_still_creates_reviewer_group(self):
        stack, _ = _create_cognito_stack(
            existing_user_pool_id="us-east-1_EXISTING",
            existing_client_id="existing-client-id",
        )
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::Cognito::UserPoolGroup",
            {
                "GroupName": "test-group",
            },
        )


class TestCognitoAttributes:
    """Tests for CognitoWorkforceTemplate exposed attributes."""

    def test_attributes_set_for_new_pool(self):
        _, cognito = _create_cognito_stack()
        assert cognito.user_pool is not None
        assert cognito.user_pool_id is not None
        assert cognito.client_id is not None
        assert cognito.user_pool_group is not None

    def test_attributes_set_for_existing_pool(self):
        _, cognito = _create_cognito_stack(
            existing_user_pool_id="us-east-1_EXISTING",
            existing_client_id="existing-client-id",
        )
        assert cognito.user_pool_id == "us-east-1_EXISTING"
        assert cognito.client_id == "existing-client-id"
