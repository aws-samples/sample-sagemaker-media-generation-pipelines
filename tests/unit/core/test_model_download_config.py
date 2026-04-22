"""
Unit tests for ModelDownloadConfig model validation.
"""

import pytest
from pydantic import ValidationError

from config.config import ModelDownloadConfig

pytestmark = pytest.mark.core


class TestModelDownloadConfigValid:
    """Valid ModelDownloadConfig with defaults and overrides."""

    def test_defaults(self) -> None:
        cfg = ModelDownloadConfig()
        assert cfg.InstanceCount == 1
        assert cfg.InstanceType == "ml.m5.xlarge"
        assert cfg.VolumeSizeInGB == 125
        assert cfg.ContainerEntrypoint == ["python3", "main.py"]
        assert cfg.ContainerArguments == ["--download"]
        assert cfg.MaxRuntimeInSeconds == 86400

    def test_override_instance_type(self) -> None:
        cfg = ModelDownloadConfig(InstanceType="ml.c5.xlarge")
        assert cfg.InstanceType == "ml.c5.xlarge"

    def test_override_max_runtime(self) -> None:
        cfg = ModelDownloadConfig(MaxRuntimeInSeconds=3600)
        assert cfg.MaxRuntimeInSeconds == 3600

    def test_max_runtime_boundary_min(self) -> None:
        cfg = ModelDownloadConfig(MaxRuntimeInSeconds=600)
        assert cfg.MaxRuntimeInSeconds == 600

    def test_max_runtime_boundary_max(self) -> None:
        cfg = ModelDownloadConfig(MaxRuntimeInSeconds=604800)
        assert cfg.MaxRuntimeInSeconds == 604800


class TestModelDownloadConfigInvalid:
    """Invalid inputs raise ValidationError."""

    def test_max_runtime_below_min(self) -> None:
        with pytest.raises(ValidationError):
            ModelDownloadConfig(MaxRuntimeInSeconds=599)

    def test_max_runtime_above_max(self) -> None:
        with pytest.raises(ValidationError):
            ModelDownloadConfig(MaxRuntimeInSeconds=604801)

    def test_invalid_instance_type(self) -> None:
        with pytest.raises(ValidationError):
            ModelDownloadConfig(InstanceType="ml.p4d.24xlarge")
