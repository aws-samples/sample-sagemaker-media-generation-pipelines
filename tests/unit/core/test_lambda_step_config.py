"""
Unit tests for LambdaStepConfig model validation.
"""

import pytest
from pydantic import ValidationError

from config.config import LambdaStepConfig

pytestmark = pytest.mark.core


class TestLambdaStepConfigValid:
    """Valid LambdaStepConfig round-trips."""

    def test_minimal(self) -> None:
        cfg = LambdaStepConfig(lambda_path="submit_a2i_review")
        assert cfg.lambda_path == "submit_a2i_review"
        assert cfg.a2i_name == ""
        assert cfg.media_type == "video"

    def test_all_fields(self) -> None:
        cfg = LambdaStepConfig(
            lambda_path="submit_a2i_review",
            a2i_name="review_t2v",
            media_type="image",
        )
        assert cfg.a2i_name == "review_t2v"
        assert cfg.media_type == "image"

    def test_audio_media_type(self) -> None:
        cfg = LambdaStepConfig(lambda_path="my_lambda", media_type="audio")
        assert cfg.media_type == "audio"


class TestLambdaStepConfigInvalid:
    """Invalid inputs raise ValidationError."""

    def test_missing_lambda_path(self) -> None:
        with pytest.raises(ValidationError):
            LambdaStepConfig()

    def test_invalid_media_type(self) -> None:
        with pytest.raises(ValidationError):
            LambdaStepConfig(lambda_path="my_lambda", media_type="text")

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            LambdaStepConfig(lambda_path="my_lambda", unknown="bad")
