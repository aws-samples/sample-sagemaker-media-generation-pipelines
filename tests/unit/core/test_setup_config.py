"""
Unit tests for SetupConfig and PipelineConfig.setup validation.

Tests default values, missing required fields, extra fields rejected,
cross-field validation, and PipelineConfig integration with setup entries.

_Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 11.7_
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from config.config import ContainerConfig, PipelineConfig, SetupConfig

pytestmark = pytest.mark.core


def _valid_step() -> dict:
    """Return a minimal valid ContainerConfig dict."""
    return {
        "InstanceCount": 1,
        "InstanceType": "ml.g5.xlarge",
        "VolumeSizeInGB": 125,
        "ContainerEntrypoint": ["/bin/bash", "./run_job.sh"],
        "ContainerArguments": ["300"],
    }


def _valid_setup() -> dict:
    """Return a minimal valid SetupConfig dict."""
    return {
        **_valid_step(),
        "dataset_url": "1aurent/unsplash-lite",
        "dataset_script": "unsplash.py",
        "num_prompts": 100,
        "test_image_count": 500,
    }


class TestSetupConfigDefaults:
    """SetupConfig default values (Req 1.2, 7.1)."""

    def test_test_image_count_defaults_to_25000(self) -> None:
        data = _valid_setup()
        del data["test_image_count"]
        cfg = SetupConfig(**data)
        assert cfg.test_image_count == 25000

    def test_environment_defaults_to_empty_dict(self) -> None:
        cfg = SetupConfig(**_valid_setup())
        assert cfg.Environment == {}

    def test_models_prefix_defaults_to_empty_list(self) -> None:
        cfg = SetupConfig(**_valid_setup())
        assert cfg.models_prefix == []

    def test_ecr_image_defaults_to_empty_string(self) -> None:
        cfg = SetupConfig(**_valid_setup())
        assert cfg.ecr_image == ""

    def test_inherits_container_config_fields(self) -> None:
        cfg = SetupConfig(**_valid_setup())
        assert cfg.InstanceCount == 1
        assert cfg.InstanceType == "ml.g5.xlarge"
        assert cfg.VolumeSizeInGB == 125


class TestSetupConfigValid:
    """Valid SetupConfig acceptance (Req 1.1, 1.2)."""

    def test_minimal_valid_config(self) -> None:
        cfg = SetupConfig(**_valid_setup())
        assert cfg.dataset_url == "1aurent/unsplash-lite"
        assert cfg.dataset_script == "unsplash.py"
        assert cfg.num_prompts == 100
        assert cfg.test_image_count == 500

    def test_num_prompts_equals_test_image_count(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 500
        data["test_image_count"] = 500
        cfg = SetupConfig(**data)
        assert cfg.num_prompts == cfg.test_image_count

    def test_boundary_num_prompts_min(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 1
        cfg = SetupConfig(**data)
        assert cfg.num_prompts == 1

    def test_boundary_num_prompts_max(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 25000
        data["test_image_count"] = 25000
        cfg = SetupConfig(**data)
        assert cfg.num_prompts == 25000

    def test_boundary_test_image_count_min(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 1
        data["test_image_count"] = 1
        cfg = SetupConfig(**data)
        assert cfg.test_image_count == 1

    def test_boundary_test_image_count_max(self) -> None:
        data = _valid_setup()
        data["test_image_count"] = 25000
        cfg = SetupConfig(**data)
        assert cfg.test_image_count == 25000

    def test_round_trip_dict(self) -> None:
        original = SetupConfig(**_valid_setup())
        dumped = original.model_dump()
        restored = SetupConfig(**dumped)
        assert restored == original


class TestSetupConfigMissingRequired:
    """Missing required fields rejected (Req 1.2, 11.7)."""

    def test_missing_dataset_url(self) -> None:
        data = _valid_setup()
        del data["dataset_url"]
        with pytest.raises(ValidationError, match="dataset_url"):
            SetupConfig(**data)

    def test_missing_dataset_script(self) -> None:
        data = _valid_setup()
        del data["dataset_script"]
        with pytest.raises(ValidationError, match="dataset_script"):
            SetupConfig(**data)

    def test_missing_num_prompts(self) -> None:
        data = _valid_setup()
        del data["num_prompts"]
        with pytest.raises(ValidationError, match="num_prompts"):
            SetupConfig(**data)

    def test_missing_container_arguments(self) -> None:
        data = _valid_setup()
        del data["ContainerArguments"]
        with pytest.raises(ValidationError, match="ContainerArguments"):
            SetupConfig(**data)


class TestSetupConfigExtraForbid:
    """Extra fields rejected via extra='forbid' (Req 1.4)."""

    def test_extra_field_rejected(self) -> None:
        data = _valid_setup()
        data["unknown_field"] = "bad"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SetupConfig(**data)

    def test_extra_field_typo_rejected(self) -> None:
        data = _valid_setup()
        data["datset_url"] = "typo"
        with pytest.raises(ValidationError):
            SetupConfig(**data)


class TestSetupConfigCrossFieldValidation:
    """Cross-field validation: num_prompts <= test_image_count (Req 1.5)."""

    def test_num_prompts_exceeds_test_image_count(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 501
        data["test_image_count"] = 500
        with pytest.raises(ValidationError, match="num_prompts cannot exceed test_image_count"):
            SetupConfig(**data)

    def test_num_prompts_far_exceeds_test_image_count(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 50000
        data["test_image_count"] = 1
        with pytest.raises(ValidationError, match="num_prompts cannot exceed test_image_count"):
            SetupConfig(**data)


class TestSetupConfigOutOfRange:
    """Out-of-range field values rejected (Req 1.5)."""

    def test_num_prompts_zero(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 0
        with pytest.raises(ValidationError):
            SetupConfig(**data)

    def test_num_prompts_negative(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = -1
        with pytest.raises(ValidationError):
            SetupConfig(**data)

    def test_num_prompts_above_max(self) -> None:
        data = _valid_setup()
        data["num_prompts"] = 50001
        with pytest.raises(ValidationError):
            SetupConfig(**data)

    def test_test_image_count_zero(self) -> None:
        data = _valid_setup()
        data["test_image_count"] = 0
        with pytest.raises(ValidationError):
            SetupConfig(**data)

    def test_test_image_count_above_max(self) -> None:
        data = _valid_setup()
        data["test_image_count"] = 25001
        with pytest.raises(ValidationError):
            SetupConfig(**data)

    def test_volume_size_below_min(self) -> None:
        data = _valid_setup()
        data["VolumeSizeInGB"] = 49
        with pytest.raises(ValidationError):
            SetupConfig(**data)

    def test_invalid_instance_type(self) -> None:
        data = _valid_setup()
        data["InstanceType"] = "ml.p4d.24xlarge"
        with pytest.raises(ValidationError):
            SetupConfig(**data)


class TestPipelineConfigSetupField:
    """PipelineConfig.setup field integration (Req 1.1, 1.3)."""

    def test_setup_defaults_to_empty_dict(self) -> None:
        cfg = PipelineConfig(steps={"s": ContainerConfig(**_valid_step())})
        assert cfg.setup == {}

    def test_setup_accepts_valid_entry(self) -> None:
        cfg = PipelineConfig(
            steps={"s": ContainerConfig(**_valid_step())},
            setup={"dataset_ingest": SetupConfig(**_valid_setup())},
        )
        assert "dataset_ingest" in cfg.setup
        assert cfg.setup["dataset_ingest"].dataset_url == "1aurent/unsplash-lite"

    def test_setup_accepts_multiple_entries(self) -> None:
        setup2 = _valid_setup()
        setup2["dataset_url"] = "other/dataset"
        setup2["dataset_script"] = "other.py"
        cfg = PipelineConfig(
            steps={"s": ContainerConfig(**_valid_step())},
            setup={
                "job_a": SetupConfig(**_valid_setup()),
                "job_b": SetupConfig(**setup2),
            },
        )
        assert len(cfg.setup) == 2

    def test_setup_invalid_entry_rejected(self) -> None:
        bad_setup = _valid_setup()
        bad_setup["num_prompts"] = 501
        bad_setup["test_image_count"] = 500
        with pytest.raises(ValidationError, match="num_prompts cannot exceed test_image_count"):
            PipelineConfig(
                steps={"s": ContainerConfig(**_valid_step())},
                setup={"bad": SetupConfig(**bad_setup)},
            )

    def test_setup_round_trip(self) -> None:
        original = PipelineConfig(
            steps={"s": ContainerConfig(**_valid_step())},
            setup={"dl": SetupConfig(**_valid_setup())},
        )
        dumped = original.model_dump()
        restored = PipelineConfig(**dumped)
        assert restored.setup["dl"].dataset_url == original.setup["dl"].dataset_url


# Feature: open-images-ingestion, Property 6: VolumeSizeInGB constraint enforcement
class TestVolumeSizeConstraintEnforcement:
    """
    Property 6: VolumeSizeInGB constraint enforcement.

    **Validates: Requirements 12.2, 12.3**

    For any integer > 125, ContainerConfig construction raises ValidationError.
    For 50–125 inclusive, construction succeeds.
    """

    @given(volume=st.integers(min_value=126, max_value=10000))
    @settings(max_examples=100)
    def test_volume_above_125_rejected(self, volume: int) -> None:
        """Any VolumeSizeInGB > 125 SHALL raise ValidationError."""
        with pytest.raises(ValidationError):
            ContainerConfig(
                ContainerArguments=["--run"],
                InstanceCount=1,
                InstanceType="ml.g5.xlarge",
                VolumeSizeInGB=volume,
            )

    @given(volume=st.integers(min_value=50, max_value=125))
    @settings(max_examples=100)
    def test_volume_50_to_125_accepted(self, volume: int) -> None:
        """Any VolumeSizeInGB in [50, 125] SHALL be accepted."""
        cfg = ContainerConfig(
            ContainerArguments=["--run"],
            InstanceCount=1,
            InstanceType="ml.g5.xlarge",
            VolumeSizeInGB=volume,
        )
        assert cfg.VolumeSizeInGB == volume
