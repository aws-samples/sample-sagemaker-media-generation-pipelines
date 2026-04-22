"""
Unit tests for A2IConfig model validation.
"""

import pytest
from pydantic import ValidationError

from config.config import A2IConfig

pytestmark = pytest.mark.core


class TestA2IConfigValid:
    """Valid A2IConfig round-trips."""

    def test_defaults(self) -> None:
        cfg = A2IConfig()
        assert cfg.media_type == "video"
        assert cfg.task_count == 1
        assert cfg.task_timeout_seconds == 3600
        assert cfg.max_concurrent_tasks == 10

    def test_all_fields(self) -> None:
        cfg = A2IConfig(
            workteam_name="my-team",
            task_template_s3_uri="s3://bucket/template.html",
            media_type="image",
            task_title="Review images",
            task_description="Pick the best image",
            task_count=3,
            task_timeout_seconds=1800,
            max_concurrent_tasks=50,
        )
        assert cfg.workteam_name == "my-team"
        assert cfg.media_type == "image"
        assert cfg.task_count == 3

    def test_audio_media_type(self) -> None:
        cfg = A2IConfig(media_type="audio")
        assert cfg.media_type == "audio"

    def test_task_count_boundary_min(self) -> None:
        cfg = A2IConfig(task_count=1)
        assert cfg.task_count == 1

    def test_task_count_boundary_max(self) -> None:
        cfg = A2IConfig(task_count=9)
        assert cfg.task_count == 9

    def test_task_timeout_boundary_min(self) -> None:
        cfg = A2IConfig(task_timeout_seconds=60)
        assert cfg.task_timeout_seconds == 60

    def test_task_timeout_boundary_max(self) -> None:
        cfg = A2IConfig(task_timeout_seconds=28800)
        assert cfg.task_timeout_seconds == 28800

    def test_max_concurrent_tasks_boundary(self) -> None:
        cfg = A2IConfig(max_concurrent_tasks=1000)
        assert cfg.max_concurrent_tasks == 1000


class TestA2IConfigInvalid:
    """Invalid inputs raise ValidationError."""

    def test_invalid_media_type(self) -> None:
        with pytest.raises(ValidationError):
            A2IConfig(media_type="text")

    def test_task_count_below_min(self) -> None:
        with pytest.raises(ValidationError):
            A2IConfig(task_count=0)

    def test_task_count_above_max(self) -> None:
        with pytest.raises(ValidationError):
            A2IConfig(task_count=10)

    def test_task_timeout_below_min(self) -> None:
        with pytest.raises(ValidationError):
            A2IConfig(task_timeout_seconds=59)

    def test_task_timeout_above_max(self) -> None:
        with pytest.raises(ValidationError):
            A2IConfig(task_timeout_seconds=28801)

    def test_max_concurrent_tasks_below_min(self) -> None:
        with pytest.raises(ValidationError):
            A2IConfig(max_concurrent_tasks=0)

    def test_max_concurrent_tasks_above_max(self) -> None:
        with pytest.raises(ValidationError):
            A2IConfig(max_concurrent_tasks=1001)

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            A2IConfig(unknown_field="bad")
