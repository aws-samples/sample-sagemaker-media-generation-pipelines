# Feature: sagemaker-pipeline, Task 17.5: PipelineConfig unit tests
"""
Unit tests for PipelineConfig model validation.

**Validates: Requirements 5.1, 5.5, 5.7**
"""

import pytest
from pydantic import ValidationError

from config.config import ContainerConfig, PipelineConfig, S3Download
from tests.unit.conftest import STEP_0

pytestmark = pytest.mark.core


def _valid_step() -> dict:
    return {
        "InstanceCount": 1,
        "InstanceType": "ml.g5.xlarge",
        "VolumeSizeInGB": 125,
        "ContainerEntrypoint": ["/bin/bash", "./run_job.sh"],
        "ContainerArguments": ["300"],
    }


class TestPipelineConfigValid:
    """Valid config round-trips through PipelineConfig."""

    def test_minimal_valid_config(self) -> None:
        cfg = PipelineConfig(steps={STEP_0: ContainerConfig(**_valid_step())})
        assert cfg.construct_id == "dev"
        assert cfg.s3_downloads == []
        assert STEP_0 in cfg.steps

    def test_full_valid_config(self) -> None:
        cfg = PipelineConfig(
            construct_id="prod",
            s3_downloads=[S3Download(url="https://example.com/model.bin", path="models/model.bin")],
            steps={
                "step_a": ContainerConfig(**_valid_step()),
                "step_b": ContainerConfig(**_valid_step()),
            },
        )
        assert cfg.construct_id == "prod"
        assert len(cfg.s3_downloads) == 1
        assert len(cfg.steps) == 2

    def test_round_trip_dict(self) -> None:
        original = PipelineConfig(
            construct_id="test",
            s3_downloads=[S3Download(url="https://example.com/a.bin", path="models/a.bin")],
            steps={"s1": ContainerConfig(**_valid_step())},
        )
        dumped = original.model_dump()
        restored = PipelineConfig(**dumped)
        assert restored == original


class TestPipelineConfigExtraForbid:
    """extra='forbid' rejects unknown keys (Req 5.7)."""

    def test_extra_top_level_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PipelineConfig(
                steps={"s1": ContainerConfig(**_valid_step())},
                unknown_field="bad",
            )

    def test_extra_step_key_rejected(self) -> None:
        step = _valid_step()
        step["ExtraField"] = "bad"
        with pytest.raises(ValidationError):
            PipelineConfig(steps={"s1": ContainerConfig(**step)})


class TestPipelineConfigConstructIdPattern:
    """construct_id pattern validation (Req 5.1)."""

    def test_valid_construct_ids(self) -> None:
        for cid in ["dev", "prod", "my-env", "a1b2"]:
            cfg = PipelineConfig(construct_id=cid, steps={"s": ContainerConfig(**_valid_step())})
            assert cfg.construct_id == cid

    def test_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id="Dev", steps={"s": ContainerConfig(**_valid_step())})

    def test_starts_with_digit_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id="1dev", steps={"s": ContainerConfig(**_valid_step())})

    def test_underscore_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id="my_env", steps={"s": ContainerConfig(**_valid_step())})


class TestPipelineConfigSteps:
    """steps dict maps step names to valid ContainerConfig (Req 5.5)."""

    def test_missing_steps_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id="dev")

    def test_empty_steps_allowed(self) -> None:
        cfg = PipelineConfig(steps={})
        assert cfg.steps == {}

    def test_steps_with_invalid_config_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig(steps={"s1": {"bad": "config"}})
