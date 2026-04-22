"""
Unit tests for S3Download model validation.
"""

import pytest
from pydantic import ValidationError

from config.config import S3Download

pytestmark = pytest.mark.core


class TestS3DownloadValid:
    """Valid S3Download round-trips."""

    def test_minimal(self) -> None:
        cfg = S3Download(url="https://example.com/model.bin", path="models/model.bin")
        assert cfg.url == "https://example.com/model.bin"
        assert cfg.path == "models/model.bin"
        assert cfg.extract is False

    def test_extract_flag_true(self) -> None:
        cfg = S3Download(
            url="https://example.com/archive.zip",
            path="models/archive",
            extract=True,
        )
        assert cfg.extract is True

    def test_round_trip(self) -> None:
        original = S3Download(url="https://example.com/a.bin", path="a/b", extract=True)
        restored = S3Download(**original.model_dump())
        assert restored == original


class TestS3DownloadInvalid:
    """Invalid inputs raise ValidationError."""

    def test_missing_url(self) -> None:
        with pytest.raises(ValidationError):
            S3Download(path="models/model.bin")

    def test_missing_path(self) -> None:
        with pytest.raises(ValidationError):
            S3Download(url="https://example.com/model.bin")

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            S3Download(url="https://example.com/a.bin", path="a/b", unknown="bad")
