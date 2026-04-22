"""
Unit tests for DynamoDBConfig model validation.
"""

import pytest
from pydantic import ValidationError

from config.config import DynamoDBConfig

pytestmark = pytest.mark.core


class TestDynamoDBConfigValid:
    """Valid DynamoDBConfig with defaults and overrides."""

    def test_defaults(self) -> None:
        cfg = DynamoDBConfig()
        assert cfg.partition_key == "id"
        assert cfg.sort_key == "step"

    def test_custom_keys(self) -> None:
        cfg = DynamoDBConfig(partition_key="pk", sort_key="sk")
        assert cfg.partition_key == "pk"
        assert cfg.sort_key == "sk"

    def test_round_trip(self) -> None:
        original = DynamoDBConfig(partition_key="my_pk", sort_key="my_sk")
        restored = DynamoDBConfig(**original.model_dump())
        assert restored == original


class TestDynamoDBConfigInvalid:
    """Invalid inputs raise ValidationError."""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DynamoDBConfig(unknown_field="bad")
