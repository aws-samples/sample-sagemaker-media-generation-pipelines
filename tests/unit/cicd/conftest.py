"""Shared fixtures and helpers for cicd/ tests.

Provides CiCdPipelineStack factory with mocked s3_assets.Asset and
a fixture to mock _read_config_prefix() to avoid reading real YAML files.

Importable helpers (not fixtures):
    _mock_s3_asset              — re-exported from root conftest
    _create_cicd_pipeline_stack — synthesise SecurityStack + CiCdPipelineStack
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import assertions

from config.config import CicdConfig
from infrastructure.cicd_pipeline.stack import CiCdPipelineStack
from infrastructure.security import SecurityStack
from tests.unit.conftest import _mock_s3_asset  # noqa: F401 — re-exported

# Default single-config for fast tests
_SINGLE_CONFIG = CicdConfig(pipeline_configs=["config_vrag.yaml"])


def _create_cicd_pipeline_stack(
    cicd_config: CicdConfig | None = None,
) -> tuple[CiCdPipelineStack, assertions.Template]:
    """Helper to create a SecurityStack + CiCdPipelineStack for testing."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    sec = SecurityStack(app, "SecStack", env=env)
    config = cicd_config or _SINGLE_CONFIG
    with patch("infrastructure.cicd_pipeline.s3_assets.Asset", side_effect=_mock_s3_asset):
        stack = CiCdPipelineStack(
            app,
            "CiCdStack",
            security_stack=sec,
            cicd_config=config,
            prefix="dev",
            env=env,
        )
    template = assertions.Template.from_stack(stack)
    return stack, template


@pytest.fixture
def mock_read_config_prefix():
    """Mock _read_config_prefix to avoid reading real YAML files."""
    with patch.object(CiCdPipelineStack, "_read_config_prefix", return_value="dev"):
        yield
