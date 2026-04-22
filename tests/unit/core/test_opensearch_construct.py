"""
Unit tests for the OpenSearchServerlessConstruct.

Tests verify that the AOSS construct creates a VECTORSEARCH collection
with encryption, network, and data access policies, plus an SSM
parameter for the collection endpoint.
"""

import aws_cdk as cdk
import pytest
from aws_cdk import assertions
from aws_cdk import aws_kms as kms

from project_constructs.opensearch import OpenSearchServerlessConstruct

pytestmark = pytest.mark.core


def _create_opensearch_stack() -> tuple[cdk.Stack, OpenSearchServerlessConstruct]:
    """Helper to create a stack with an OpenSearchServerlessConstruct for testing."""
    app = cdk.App()
    stack = cdk.Stack(app, "TestStack", env=cdk.Environment(account="123456789012", region="us-east-1"))
    kms_key = kms.Key(stack, "TestKmsKey", enable_key_rotation=True)

    opensearch = OpenSearchServerlessConstruct(
        stack,
        "TestOpenSearch",
        collection_name="test-collection",
        prefix="dev",
        ssm_parameter_name="/dev/rag/aoss-endpoint",
        principal_arns=["arn:aws:iam::123456789012:role/test-role"],
        kms_key=kms_key,
    )
    return stack, opensearch


class TestOpenSearchResources:
    """Tests for OpenSearchServerlessConstruct resource creation."""

    def test_creates_collection(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        template.resource_count_is("AWS::OpenSearchServerless::Collection", 1)

    def test_collection_is_vectorsearch_type(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::OpenSearchServerless::Collection",
            {
                "Name": "dev-test-collection",
                "Type": "VECTORSEARCH",
            },
        )

    def test_creates_ssm_parameter(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::SSM::Parameter",
            {
                "Name": "/dev/rag/aoss-endpoint",
            },
        )


class TestOpenSearchSecurityPolicies:
    """Tests for OpenSearchServerless encryption and network policies."""

    def test_creates_encryption_policy(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::OpenSearchServerless::SecurityPolicy",
            {
                "Name": "dev-test-collection-enc",
                "Type": "encryption",
            },
        )

    def test_creates_network_policy(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::OpenSearchServerless::SecurityPolicy",
            {
                "Name": "dev-test-collection-net",
                "Type": "network",
            },
        )

    def test_encryption_policy_uses_aws_owned_key(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        # The policy JSON is stored as a string in the template
        template.has_resource_properties(
            "AWS::OpenSearchServerless::SecurityPolicy",
            {
                "Name": "dev-test-collection-enc",
                "Type": "encryption",
                "Policy": assertions.Match.string_like_regexp("AWSOwnedKey.*true"),
            },
        )


class TestOpenSearchAccessPolicy:
    """Tests for OpenSearchServerless data access policy."""

    def test_creates_data_access_policy(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::OpenSearchServerless::AccessPolicy",
            {
                "Name": "dev-test-collection-access",
                "Type": "data",
            },
        )

    def test_data_access_policy_references_collection(self):
        stack, _ = _create_opensearch_stack()
        template = assertions.Template.from_stack(stack)
        template.has_resource_properties(
            "AWS::OpenSearchServerless::AccessPolicy",
            {
                "Name": "dev-test-collection-access",
                "Policy": assertions.Match.string_like_regexp("collection/dev-test-collection"),
            },
        )


class TestOpenSearchAttributes:
    """Tests for OpenSearchServerlessConstruct exposed attributes."""

    def test_collection_exposed(self):
        _, opensearch = _create_opensearch_stack()
        assert opensearch.collection is not None

    def test_endpoint_parameter_exposed(self):
        _, opensearch = _create_opensearch_stack()
        assert opensearch.endpoint_parameter is not None

    def test_collection_endpoint_exposed(self):
        _, opensearch = _create_opensearch_stack()
        assert opensearch.collection_endpoint is not None
