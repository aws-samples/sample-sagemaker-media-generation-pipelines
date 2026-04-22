"""Shared fixtures and helpers for processing/ tests.

Provides _valid_step() helper for creating valid ContainerConfig dicts
and _default_cfg() for creating ContainerConfig instances used across
processing job tests.
"""

from config.config import ContainerConfig


def _valid_step() -> dict:
    """Return a minimal valid ContainerConfig dict for testing."""
    return {
        "InstanceCount": 1,
        "InstanceType": "ml.g5.xlarge",
        "VolumeSizeInGB": 125,
        "ContainerEntrypoint": ["/bin/bash", "./run_job.sh"],
        "ContainerArguments": ["300"],
    }


def _default_cfg() -> ContainerConfig:
    """Return a minimal valid ContainerConfig for testing."""
    return ContainerConfig(**_valid_step())
