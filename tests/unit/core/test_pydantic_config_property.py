# Feature: unit-test-reorganization, Property 9.4: Pydantic config model property tests
"""
Property-based tests for Pydantic config models: A2IConfig, LambdaStepConfig,
ModelDownloadConfig, DynamoDBConfig, S3Download.

Verifies round-trip serialization and constraint enforcement across
randomized inputs.

**Validates: Requirements 9.4**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from config.config import (
    A2IConfig,
    DynamoDBConfig,
    LambdaStepConfig,
    ModelDownloadConfig,
    S3Download,
)

pytestmark = pytest.mark.core


# --- Strategies ---

_media_type_st = st.sampled_from(["image", "video", "audio"])
_nonempty_str = st.text(min_size=1, max_size=50)
_instance_type_st = st.sampled_from(
    [
        "ml.c5.xlarge",
        "ml.g4dn.2xlarge",
        "ml.g5.xlarge",
        "ml.g5.8xlarge",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
    ]
)


class TestA2IConfigProperty:
    """Property tests for A2IConfig round-trip and constraints."""

    @given(
        media_type=_media_type_st,
        task_count=st.integers(min_value=1, max_value=9),
        task_timeout=st.integers(min_value=60, max_value=28800),
        max_concurrent=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_round_trip(
        self,
        media_type: str,
        task_count: int,
        task_timeout: int,
        max_concurrent: int,
    ) -> None:
        """**Validates: Requirements 9.4**"""
        cfg = A2IConfig(
            media_type=media_type,
            task_count=task_count,
            task_timeout_seconds=task_timeout,
            max_concurrent_tasks=max_concurrent,
        )
        restored = A2IConfig(**cfg.model_dump())
        assert restored == cfg

    @given(task_count=st.integers().filter(lambda x: x < 1 or x > 9))
    @settings(max_examples=100, deadline=None)
    def test_invalid_task_count_rejected(self, task_count: int) -> None:
        """**Validates: Requirements 9.4**"""
        with pytest.raises(ValidationError):
            A2IConfig(task_count=task_count)


class TestDynamoDBConfigProperty:
    """Property tests for DynamoDBConfig round-trip."""

    @given(
        partition_key=_nonempty_str,
        sort_key=_nonempty_str,
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_round_trip(self, partition_key: str, sort_key: str) -> None:
        """**Validates: Requirements 9.4**"""
        cfg = DynamoDBConfig(partition_key=partition_key, sort_key=sort_key)
        restored = DynamoDBConfig(**cfg.model_dump())
        assert restored == cfg

    @given(extra_field=_nonempty_str.filter(lambda s: s not in {"partition_key", "sort_key"}))
    @settings(max_examples=100, deadline=None)
    def test_extra_field_rejected(self, extra_field: str) -> None:
        """**Validates: Requirements 9.4**"""
        with pytest.raises(ValidationError):
            DynamoDBConfig(**{extra_field: "value"})


class TestLambdaStepConfigProperty:
    """Property tests for LambdaStepConfig round-trip."""

    @given(
        lambda_path=_nonempty_str,
        media_type=_media_type_st,
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_round_trip(self, lambda_path: str, media_type: str) -> None:
        """**Validates: Requirements 9.4**"""
        cfg = LambdaStepConfig(lambda_path=lambda_path, media_type=media_type)
        restored = LambdaStepConfig(**cfg.model_dump())
        assert restored == cfg


class TestModelDownloadConfigProperty:
    """Property tests for ModelDownloadConfig constraints."""

    @given(
        instance_type=_instance_type_st,
        max_runtime=st.integers(min_value=600, max_value=604800),
        volume_size=st.integers(min_value=50, max_value=125),
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_round_trip(
        self,
        instance_type: str,
        max_runtime: int,
        volume_size: int,
    ) -> None:
        """**Validates: Requirements 9.4**"""
        cfg = ModelDownloadConfig(
            InstanceType=instance_type,
            MaxRuntimeInSeconds=max_runtime,
            VolumeSizeInGB=volume_size,
        )
        restored = ModelDownloadConfig(**cfg.model_dump())
        assert restored == cfg

    @given(max_runtime=st.integers().filter(lambda x: x < 600 or x > 604800))
    @settings(max_examples=100, deadline=None)
    def test_invalid_max_runtime_rejected(self, max_runtime: int) -> None:
        """**Validates: Requirements 9.4**"""
        with pytest.raises(ValidationError):
            ModelDownloadConfig(MaxRuntimeInSeconds=max_runtime)


class TestS3DownloadProperty:
    """Property tests for S3Download round-trip."""

    @given(
        url=_nonempty_str,
        path=_nonempty_str,
        extract=st.booleans(),
    )
    @settings(max_examples=100, deadline=None)
    def test_valid_round_trip(self, url: str, path: str, extract: bool) -> None:
        """**Validates: Requirements 9.4**"""
        cfg = S3Download(url=url, path=path, extract=extract)
        restored = S3Download(**cfg.model_dump())
        assert restored == cfg
