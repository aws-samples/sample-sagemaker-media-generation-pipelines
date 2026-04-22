"""
Reusable OpenSearch Serverless (AOSS) construct for vector search.

This module provides an OpenSearchServerlessConstruct that creates an AOSS
VECTORSEARCH collection with encryption, network, and data access policies,
plus an SSM parameter for the collection endpoint.
"""

import json

from aws_cdk import (
    aws_kms as kms,
)
from aws_cdk import (
    aws_opensearchserverless as opensearchserverless,
)
from cdk_nag import NagSuppressions
from constructs import Construct
from loguru import logger

from .ssm_parameter import SsmParameter


class OpenSearchServerlessConstruct(Construct):
    """
    Reusable OpenSearch Serverless VECTORSEARCH collection with access policies.

    Attributes:
        collection: The CfnCollection resource (VECTORSEARCH type).
        collection_endpoint: The collection endpoint hostname (CFn token).
        endpoint_parameter: SSM parameter storing the collection endpoint.
    """

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        collection_name: str,
        prefix: str,
        ssm_parameter_name: str,
        principal_arns: list[str],
        kms_key: kms.Key | None = None,
    ) -> None:
        """
        Initialize the OpenSearchServerlessConstruct.

        Args:
            scope: The parent construct.
            construct_id: Unique identifier for this construct.
            collection_name: Name for the AOSS collection (alphanumeric, hyphens, 3-32 chars).
            prefix: Stack prefix for resource naming.
            kms_key: KMS key (stored but not used for AOSS encryption; AOSS uses AWS-owned key).
            ssm_parameter_name: Full SSM parameter name for storing the endpoint (e.g. /dev/rag/aoss-endpoint).
            principal_arns: List of IAM role ARNs that should have data access to the collection.
        """
        super().__init__(scope, construct_id)

        # Prefix the collection name to avoid account-wide naming collisions
        prefixed_collection = f"{prefix}-{collection_name}"

        # AOSS policy names have a 32-char limit; use short suffixes
        security_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            f"{construct_id}-SecurityPolicy",
            name=f"{prefixed_collection}-enc",
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{prefixed_collection}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )

        # Network security policy: allow public access (Lambdas reach via NAT)
        network_security_policy = opensearchserverless.CfnSecurityPolicy(
            self,
            f"{construct_id}-NetworkSecurityPolicy",
            name=f"{prefixed_collection}-net",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{prefixed_collection}"],
                            },
                            {
                                "ResourceType": "dashboard",
                                "Resource": [f"collection/{prefixed_collection}"],
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )

        # Data access policy: grant collection and index CRUD to principal ARNs
        opensearchserverless.CfnAccessPolicy(
            self,
            f"{construct_id}-DataAccessPolicy",
            name=f"{prefixed_collection}-access",
            type="data",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{prefixed_collection}"],
                                "Permission": [
                                    "aoss:CreateCollectionItems",
                                    "aoss:DeleteCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                    "aoss:DescribeCollectionItems",
                                ],
                            },
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{prefixed_collection}/*"],
                                "Permission": [
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex",
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                ],
                            },
                        ],
                        "Principal": principal_arns,
                    }
                ]
            ),
        )

        # VECTORSEARCH collection (depends on security policies)
        self.collection = opensearchserverless.CfnCollection(
            self,
            f"{construct_id}-Collection",
            name=prefixed_collection,
            type="VECTORSEARCH",
        )
        self.collection.add_dependency(security_policy)
        self.collection.add_dependency(network_security_policy)

        # Store collection endpoint from the CFn attribute (CDK token)
        self.collection_endpoint = self.collection.attr_collection_endpoint

        # SSM parameter for endpoint
        self.endpoint_parameter = SsmParameter(
            self,
            f"{construct_id}-EndpointParameter",
            parameter_name=ssm_parameter_name,
            string_value=self.collection_endpoint,
            description=f"AOSS collection endpoint for {collection_name}",
        )

        logger.info("Created OpenSearch Serverless collection: {}", prefixed_collection)

        NagSuppressions.add_resource_suppressions(
            self,
            suppressions=[
                {
                    "id": "AwsSolutions-OS1",
                    "reason": "AOSS VECTORSEARCH does not support dedicated master nodes; OpenSearch Serverless manages this automatically",
                },
                {
                    "id": "AwsSolutions-OS3",
                    "reason": "Access control is enforced via AOSS data access policy with explicit principal ARNs",
                },
                {
                    "id": "AwsSolutions-OS5",
                    "reason": "Access control is enforced via AOSS data access policy; IP-based restrictions not applicable",
                },
                {
                    "id": "AwsSolutions-OS7",
                    "reason": "AOSS VECTORSEARCH collections do not support zone awareness configuration",
                },
                {
                    "id": "AwsSolutions-OS9",
                    "reason": "AOSS audit logging is not configurable via CfnCollection L1 construct",
                },
            ],
            apply_to_children=True,
        )
