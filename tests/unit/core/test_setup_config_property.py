# Feature: unsplash-setup-ingestion, Property 1: Valid SetupConfig acceptance
# Feature: unsplash-setup-ingestion, Property 2: Invalid SetupConfig rejection
"""
Property-based tests for SetupConfig validation.

Property 1: Valid SetupConfig acceptance
- For any SetupConfig with dataset_url as a non-empty string, num_prompts in [1, 50000],
  test_image_count in [1, 25000], and num_prompts <= test_image_count, plus valid inherited
  ContainerConfig fields, the config SHALL be accepted without validation error.

Property 2: Invalid SetupConfig rejection
- For any SetupConfig where either an unknown field is present, or num_prompts > test_image_count,
  or any field is out of its valid range, the config SHALL raise a ValidationError.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from config.config import SetupConfig

pytestmark = pytest.mark.core


# --- Strategies ---

# Valid instance types (from ContainerConfig Literal)
instance_type_st = st.sampled_from(
    [
        "ml.c5.xlarge",
        "ml.g4dn.2xlarge",
        "ml.g5.xlarge",
        "ml.g5.8xlarge",
        "ml.m5.xlarge",
        "ml.m5.2xlarge",
    ]
)

# Non-empty dataset URL
dataset_url_st = st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P")))

# Dataset script filename
dataset_script_st = st.from_regex(r"^[a-z][a-z0-9_]{0,15}\.py$", fullmatch=True)


# num_prompts and test_image_count with cross-field constraint
@st.composite
def valid_prompt_image_counts(draw):
    """Generate (num_prompts, test_image_count) where num_prompts <= test_image_count."""
    test_image_count = draw(st.integers(min_value=1, max_value=25000))
    num_prompts = draw(st.integers(min_value=1, max_value=test_image_count))
    return num_prompts, test_image_count


# Full valid SetupConfig dict strategy
@st.composite
def valid_setup_config_st(draw):
    """Generate a valid SetupConfig dict with all required fields."""
    num_prompts, test_image_count = draw(valid_prompt_image_counts())
    return {
        "InstanceCount": draw(st.integers(min_value=1, max_value=10)),
        "InstanceType": draw(instance_type_st),
        "VolumeSizeInGB": draw(st.integers(min_value=50, max_value=125)),
        "ContainerEntrypoint": draw(st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=3)),
        "ContainerArguments": draw(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=3)),
        "dataset_url": draw(dataset_url_st),
        "dataset_script": draw(dataset_script_st),
        "num_prompts": num_prompts,
        "test_image_count": test_image_count,
    }


class TestValidSetupConfigAcceptance:
    """
    Property 1: Valid SetupConfig acceptance.

    **Validates: Requirements 1.1, 1.2, 7.1**
    """

    @given(config_dict=valid_setup_config_st())
    @settings(max_examples=100)
    def test_valid_config_accepted(self, config_dict: dict) -> None:
        """
        **Validates: Requirements 1.1, 1.2, 7.1**

        For any valid field combination with num_prompts <= test_image_count,
        SetupConfig SHALL be accepted without ValidationError.
        """
        model = SetupConfig(**config_dict)
        assert model.dataset_url == config_dict["dataset_url"]
        assert model.dataset_script == config_dict["dataset_script"]
        assert model.num_prompts == config_dict["num_prompts"]
        assert model.test_image_count == config_dict["test_image_count"]
        assert model.InstanceType == config_dict["InstanceType"]
        assert model.VolumeSizeInGB == config_dict["VolumeSizeInGB"]

    @given(config_dict=valid_setup_config_st())
    @settings(max_examples=100)
    def test_valid_config_round_trips(self, config_dict: dict) -> None:
        """
        **Validates: Requirements 1.1, 1.2, 7.1**

        For any valid config dict, SetupConfig(**d) succeeds and
        model_dump() preserves the setup-specific fields.
        """
        model = SetupConfig(**config_dict)
        dumped = model.model_dump()
        assert dumped["dataset_url"] == config_dict["dataset_url"]
        assert dumped["dataset_script"] == config_dict["dataset_script"]
        assert dumped["num_prompts"] == config_dict["num_prompts"]
        assert dumped["test_image_count"] == config_dict["test_image_count"]


class TestInvalidSetupConfigRejection:
    """
    Property 2: Invalid SetupConfig rejection.

    **Validates: Requirements 1.4, 1.5**
    """

    @given(config_dict=valid_setup_config_st())
    @settings(max_examples=100)
    def test_num_prompts_exceeds_test_image_count_rejected(self, config_dict: dict) -> None:
        """
        **Validates: Requirements 1.5**

        When num_prompts > test_image_count, SetupConfig SHALL raise ValidationError.
        """
        # Force num_prompts to exceed test_image_count
        config_dict["num_prompts"] = config_dict["test_image_count"] + 1
        with pytest.raises(ValidationError, match="num_prompts cannot exceed test_image_count"):
            SetupConfig(**config_dict)

    @given(
        config_dict=valid_setup_config_st(),
        extra_key=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L",))).filter(
            lambda k: (
                k
                not in {
                    "InstanceCount",
                    "InstanceType",
                    "VolumeSizeInGB",
                    "ContainerEntrypoint",
                    "ContainerArguments",
                    "Environment",
                    "models_prefix",
                    "ecr_image",
                    "input_channel",
                    "include_data_input",
                    "num_assets_per_prompt",
                    "dataset_url",
                    "dataset_script",
                    "num_prompts",
                    "test_image_count",
                }
            )
        ),
        extra_value=st.text(min_size=1, max_size=10),
    )
    @settings(max_examples=100)
    def test_extra_fields_rejected(self, config_dict: dict, extra_key: str, extra_value: str) -> None:
        """
        **Validates: Requirements 1.4**

        When an unknown field is present, SetupConfig SHALL raise ValidationError.
        """
        invalid = {**config_dict, extra_key: extra_value}
        with pytest.raises(ValidationError):
            SetupConfig(**invalid)

    @given(config_dict=valid_setup_config_st())
    @settings(max_examples=100)
    def test_num_prompts_out_of_range_rejected(self, config_dict: dict) -> None:
        """
        **Validates: Requirements 1.5**

        When num_prompts is out of range [1, 50000], SetupConfig SHALL raise ValidationError.
        """
        config_dict["num_prompts"] = 0
        with pytest.raises(ValidationError):
            SetupConfig(**config_dict)

    @given(config_dict=valid_setup_config_st())
    @settings(max_examples=100)
    def test_test_image_count_out_of_range_rejected(self, config_dict: dict) -> None:
        """
        **Validates: Requirements 1.5**

        When test_image_count exceeds 25000, SetupConfig SHALL raise ValidationError.
        """
        config_dict["test_image_count"] = 25001
        with pytest.raises(ValidationError):
            SetupConfig(**config_dict)

    @given(config_dict=valid_setup_config_st())
    @settings(max_examples=100)
    def test_volume_size_out_of_range_rejected(self, config_dict: dict) -> None:
        """
        **Validates: Requirements 1.5**

        When VolumeSizeInGB is out of range [50, 125], SetupConfig SHALL raise ValidationError.
        """
        config_dict["VolumeSizeInGB"] = 49
        with pytest.raises(ValidationError):
            SetupConfig(**config_dict)
