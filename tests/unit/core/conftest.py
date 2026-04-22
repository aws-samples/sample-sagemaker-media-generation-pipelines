"""Shared fixtures and helpers for core/ tests.

Provides SecurityStack factory, default config helpers, and template
cache fixtures for DataStack/PipelineStack synthesis.

Importable helpers (not fixtures):
    _mock_from_asset  — re-exported from root conftest
    _default_cfg      — minimal ContainerConfig
    _default_pipeline_config — minimal PipelineConfig (accepts optional step_names)
"""

from unittest.mock import patch

import aws_cdk as cdk
import pytest
from aws_cdk import aws_lambda as lambda_

from config.config import ContainerConfig, PipelineConfig
from infrastructure.security import SecurityStack
from tests.unit.conftest import PRIMARY_STEPS, _mock_from_asset  # noqa: F401 — re-exported


def _default_cfg() -> ContainerConfig:
    """Return a minimal valid ContainerConfig for testing."""
    return ContainerConfig(
        InstanceCount=1,
        InstanceType="ml.g5.xlarge",
        VolumeSizeInGB=125,
        ContainerEntrypoint=["/bin/bash", "./run_job.sh"],
        ContainerArguments=["300"],
    )


def _default_pipeline_config(step_names: list[str] | None = None) -> PipelineConfig:
    """Return a minimal valid PipelineConfig using PRIMARY_STEPS."""
    names = step_names or PRIMARY_STEPS
    return PipelineConfig(
        construct_id="dev",
        s3_downloads=[],
        steps={s: _default_cfg() for s in names},
    )


@pytest.fixture(scope="class")
def security_stack():
    """Create a SecurityStack with mocked Lambda assets (class-scoped)."""
    app = cdk.App()
    env = cdk.Environment(account="123456789012", region="us-east-1")
    with patch.object(lambda_.Code, "from_asset", side_effect=_mock_from_asset):
        return SecurityStack(app, "SecStack", prefix="dev", env=env)
