"""
Unit tests for config loader functions: get_pipeline_config, get_cicd_config, get_retrieval_config.

All file reads are mocked to avoid filesystem I/O.
"""

from unittest.mock import mock_open, patch

import pytest
import yaml

from config.config import get_cicd_config, get_pipeline_config, get_retrieval_config

pytestmark = pytest.mark.core


def _pipeline_yaml() -> str:
    return yaml.dump(
        {
            "construct_id": "test",
            "steps": {
                "step1": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.g5.xlarge",
                    "VolumeSizeInGB": 125,
                    "ContainerEntrypoint": ["/bin/bash", "./run_job.sh"],
                    "ContainerArguments": ["300"],
                }
            },
        }
    )


def _cicd_yaml() -> str:
    return yaml.dump(
        {
            "enabled": True,
            "pipeline_configs": ["config_vrag.yaml"],
            "test_commands": {
                "config_vrag.yaml": "uv run pytest tests/unit/ -x -m core",
            },
        }
    )


def _retrieval_yaml() -> str:
    return yaml.dump(
        {
            "collection_name": "test-col",
            "index_name": "test-idx",
            "sqs_visibility_timeout_seconds": 960,
            "sqs_max_receive_count": 3,
            "ingest_lambda_timeout_seconds": 300,
            "ingest_lambda_memory_mb": 2048,
        }
    )


class TestGetPipelineConfig:
    """get_pipeline_config loads and validates pipeline YAML."""

    def test_loads_valid_config(self) -> None:
        with patch("builtins.open", mock_open(read_data=_pipeline_yaml())):
            cfg = get_pipeline_config("test.yaml")
        assert cfg.construct_id == "test"
        assert "step1" in cfg.steps

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_pipeline_config("nonexistent.yaml")


class TestGetCicdConfig:
    """get_cicd_config loads and validates CI/CD YAML."""

    def test_loads_valid_config(self) -> None:
        with patch("builtins.open", mock_open(read_data=_cicd_yaml())):
            cfg = get_cicd_config("test.yaml")
        assert cfg.enabled is True
        assert "config_vrag.yaml" in cfg.test_commands

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_cicd_config("nonexistent.yaml")


class TestGetRetrievalConfig:
    """get_retrieval_config loads and validates retrieval YAML."""

    def test_loads_valid_config(self) -> None:
        with patch("builtins.open", mock_open(read_data=_retrieval_yaml())):
            cfg = get_retrieval_config("test.yaml")
        assert cfg.collection_name == "test-col"
        assert cfg.index_name == "test-idx"

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_retrieval_config("nonexistent.yaml")
