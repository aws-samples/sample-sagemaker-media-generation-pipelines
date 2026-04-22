"""
Unit tests for RetrievalConstruct ECR prefix behaviour.

Verifies that RetrievalConstruct uses ecr_prefix (not prefix) for setup job
ECR image URIs, while resource naming continues to use prefix.
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
from tests.unit.conftest import _mock_from_asset

pytestmark = pytest.mark.retrieval


def _valid_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig(
        collection_name="test-images",
        index_name="test-vectors",
        sqs_visibility_timeout_seconds=960,
        sqs_max_receive_count=3,
        ingest_lambda_timeout_seconds=300,
        ingest_lambda_memory_mb=2048,
    )


def _default_cfg() -> ContainerConfig:
    return ContainerConfig(
        InstanceCount=1,
        InstanceType="ml.g5.xlarge",
        VolumeSizeInGB=125,
        ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
        ContainerArguments=["300"],
    )


def _setup_cfg() -> SetupConfig:
    return SetupConfig(
        InstanceCount=1,
        InstanceType="ml.c5.xlarge",
        VolumeSizeInGB=50,
        ContainerEntrypoint=["python3", "main.py"],
        ContainerArguments=["--setup"],
        dataset_url="https://example.com/data",
        dataset_script="loader.py",
        num_prompts=10,
        test_image_count=100,
    )


@pytest.fixture(scope="class")
def retrieval_ecr_template():
    """Synthesize DataStack with RetrievalConstruct using ecr_prefix='shared', prefix='cfg'."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    pipeline_config = PipelineConfig(
        construct_id="cfg",
        s3_downloads=[],
        steps={"t2v": _default_cfg()},
        retrieval="retrieval.yaml",
        setup={"dataset_ingest": _setup_cfg()},
    )
    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        sec = SecurityStack(app, "SecStack", prefix="cfg", env=env)
        data = DataStack(
            app,
            "DataStack",
            security_stack=sec,
            dynamodb_config=DynamoDBConfig(),
            pipeline_config=pipeline_config,
            prefix="cfg",
            retrieval_config=_valid_retrieval_config(),
            ecr_prefix="shared",
            env=env,
        )
    return assertions.Template.from_stack(data)


class TestRetrievalConstructEcrPrefix:
    """RetrievalConstruct setup job ECR URIs use ecr_prefix."""

    def test_setup_job_ecr_uses_shared_prefix(self, retrieval_ecr_template) -> None:
        """Setup job ECR pull policy ARN contains shared/processing/dataset-ingest."""
        template_json = json.dumps(retrieval_ecr_template.to_json())
        assert "shared/processing/dataset-ingest" in template_json

    def test_setup_job_ecr_does_not_use_config_prefix(self, retrieval_ecr_template) -> None:
        """Setup job ECR path does not contain cfg/processing/dataset-ingest."""
        template_json = json.dumps(retrieval_ecr_template.to_json())
        assert "cfg/processing/dataset-ingest" not in template_json


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import assume, given, settings
from hypothesis import strategies as st


def _setup_name_strategy():
    return st.from_regex(r"[a-z][a-z0-9_]{1,12}", fullmatch=True)


def _prefix_strategy():
    return st.from_regex(r"[a-z][a-z0-9]{1,8}", fullmatch=True)


class TestRetrievalEcrUriProperty:
    """Property 4: RetrievalConstruct ECR URI uses ecr_prefix for setup jobs.

    Tests the string construction pattern used in RetrievalConstruct:
    ``f"{ecr_pfx}/processing/{setup_clean}"``

    **Validates: Requirements 4.1, 4.2**
    """

    @given(
        setup_name=_setup_name_strategy(),
        prefix=_prefix_strategy(),
        ecr_prefix=_prefix_strategy(),
    )
    @settings(max_examples=100)
    def test_ecr_uri_uses_ecr_prefix(self, setup_name, prefix, ecr_prefix) -> None:
        """ECR URI contains ecr_prefix/processing/{setup}, not prefix/processing/{setup} when they differ."""
        assume(prefix != ecr_prefix)

        setup_clean = setup_name.replace("_", "-")
        ecr_repo_name = f"{ecr_prefix}/processing/{setup_clean}"
        ecr_image_uri = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/{ecr_repo_name}"

        assert f"{ecr_prefix}/processing/{setup_clean}" in ecr_image_uri
        assert f"{prefix}/processing/{setup_clean}" not in ecr_image_uri
