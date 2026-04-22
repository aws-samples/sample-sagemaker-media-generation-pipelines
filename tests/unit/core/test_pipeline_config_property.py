# Feature: sagemaker-pipeline, Property 7: PipelineConfig validation round trip
"""
Property-based tests for PipelineConfig validation.

Validates: Requirements 5.1, 5.2, 5.5, 5.7

Property 7: PipelineConfig validation round trip
- For any valid dictionary with a construct_id (matching ^[a-z][a-z0-9-]*$),
  an s3_downloads list of URL strings, and a steps dict mapping step names
  to valid ContainerConfig dicts, PipelineConfig(**d) SHALL succeed and
  model.model_dump() SHALL produce an equivalent dictionary.
- For any dictionary containing extra keys not in the schema,
  PipelineConfig(**d) SHALL raise a ValidationError.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from config.config import PipelineConfig, S3Download

pytestmark = pytest.mark.core


# --- Strategies ---

# Valid construct_id: starts with lowercase letter, followed by lowercase letters, digits, or hyphens
construct_id_st = st.from_regex(r"^[a-z][a-z0-9\-]{0,9}$", fullmatch=True).filter(lambda s: not s.endswith("-"))

# Valid instance types
instance_type_st = st.sampled_from(["ml.g4dn.2xlarge", "ml.g5.xlarge"])

# Valid ContainerConfig as a dict
container_config_st = st.fixed_dictionaries(
    {
        "InstanceCount": st.integers(min_value=1, max_value=10),
        "InstanceType": instance_type_st,
        "VolumeSizeInGB": st.integers(min_value=50, max_value=125),
        "ContainerEntrypoint": st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=3),
        "ContainerArguments": st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=3),
        "Environment": st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
            values=st.text(min_size=0, max_size=20),
            max_size=3,
        ),
    }
)

# Step name: simple alphanumeric with underscores
step_name_st = st.from_regex(r"^[a-z][a-z0-9_]{0,9}$", fullmatch=True)

# Valid steps dict: 1-5 steps
steps_st = st.dictionaries(
    keys=step_name_st,
    values=container_config_st,
    min_size=1,
    max_size=5,
)

# Valid s3_downloads list (list of S3Download dicts)
_s3_url_st = st.from_regex(r"^https://[a-z0-9\.\-]+/[a-z0-9/\-]+\.[a-z]{1,5}$", fullmatch=True)
_s3_path_st = st.from_regex(r"^[a-z][a-z0-9/\-]{0,30}\.[a-z]{1,5}$", fullmatch=True)

s3_downloads_st = st.lists(
    st.fixed_dictionaries(
        {
            "url": _s3_url_st,
            "path": _s3_path_st,
        }
    ),
    min_size=0,
    max_size=3,
)

# Full valid PipelineConfig dict
valid_pipeline_config_st = st.fixed_dictionaries(
    {
        "construct_id": construct_id_st,
        "s3_downloads": s3_downloads_st,
        "steps": steps_st,
    }
)


class TestPipelineConfigValidRoundTrip:
    """Valid configs round-trip through PipelineConfig."""

    @given(config_dict=valid_pipeline_config_st)
    @settings(max_examples=100)
    def test_valid_config_round_trips(self, config_dict: dict) -> None:
        """
        **Validates: Requirements 5.1, 5.2**

        For any valid config dict, PipelineConfig(**d) succeeds and
        model_dump() produces an equivalent dictionary.
        """
        model = PipelineConfig(**config_dict)
        dumped = model.model_dump()

        assert dumped["construct_id"] == config_dict["construct_id"]
        # Compare s3_downloads by re-parsing through the model to include defaults (e.g. extract)
        expected_downloads = [S3Download(**d).model_dump() for d in config_dict["s3_downloads"]]
        assert dumped["s3_downloads"] == expected_downloads
        assert set(dumped["steps"].keys()) == set(config_dict["steps"].keys())

        for step_name, step_config in config_dict["steps"].items():
            dumped_step = dumped["steps"][step_name]
            assert dumped_step["InstanceCount"] == step_config["InstanceCount"]
            assert dumped_step["InstanceType"] == step_config["InstanceType"]
            assert dumped_step["VolumeSizeInGB"] == step_config["VolumeSizeInGB"]
            assert dumped_step["ContainerEntrypoint"] == step_config["ContainerEntrypoint"]
            assert dumped_step["ContainerArguments"] == step_config["ContainerArguments"]
            assert dumped_step["Environment"] == step_config["Environment"]


class TestPipelineConfigRejectsExtraFields:
    """Extra fields cause ValidationError due to extra='forbid'."""

    @given(
        config_dict=valid_pipeline_config_st,
        extra_key=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L",))).filter(
            lambda k: (
                k
                not in (
                    "construct_id",
                    "s3_downloads",
                    "steps",
                    "dynamodb",
                    "pipeline_graph",
                    "a2i",
                    "lambda_steps",
                )
            )
        ),
        extra_value=st.text(min_size=1, max_size=10),
    )
    @settings(max_examples=100)
    def test_extra_top_level_fields_rejected(self, config_dict: dict, extra_key: str, extra_value: str) -> None:
        """
        **Validates: Requirements 5.7**

        For any dictionary containing extra keys not in the schema,
        PipelineConfig(**d) SHALL raise a ValidationError.
        """
        invalid = {**config_dict, extra_key: extra_value}
        with pytest.raises(ValidationError):
            PipelineConfig(**invalid)


class TestPipelineConfigRejectsBadTypes:
    """Invalid types for fields cause ValidationError due to strict=True."""

    @given(steps=steps_st)
    @settings(max_examples=100)
    def test_bad_construct_id_type_rejected(self, steps: dict) -> None:
        """
        **Validates: Requirements 5.1**

        construct_id must be a string matching the pattern. Non-string types are rejected.
        """
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id=123, steps=steps)

    @given(construct_id=construct_id_st, steps=steps_st)
    @settings(max_examples=100)
    def test_bad_s3_downloads_type_rejected(self, construct_id: str, steps: dict) -> None:
        """
        **Validates: Requirements 5.5**

        s3_downloads must be a list of strings. A non-list type is rejected.
        """
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id=construct_id, s3_downloads="not-a-list", steps=steps)


class TestPipelineConfigRejectsMissingRequired:
    """Missing required fields cause ValidationError."""

    @given(construct_id=construct_id_st, s3_downloads=s3_downloads_st)
    @settings(max_examples=100)
    def test_missing_steps_rejected(self, construct_id: str, s3_downloads: list) -> None:
        """
        **Validates: Requirements 5.5**

        steps is a required field with no default. Omitting it raises ValidationError.
        """
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id=construct_id, s3_downloads=s3_downloads)


class TestPipelineConfigConstructIdPattern:
    """construct_id must match ^[a-z][a-z0-9-]*$ pattern."""

    @given(steps=steps_st)
    @settings(max_examples=100)
    def test_construct_id_starting_with_digit_rejected(self, steps: dict) -> None:
        """
        **Validates: Requirements 5.1**

        construct_id starting with a digit violates the pattern.
        """
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id="1invalid", steps=steps)

    @given(steps=steps_st)
    @settings(max_examples=100)
    def test_construct_id_with_uppercase_rejected(self, steps: dict) -> None:
        """
        **Validates: Requirements 5.1**

        construct_id with uppercase letters violates the pattern.
        """
        with pytest.raises(ValidationError):
            PipelineConfig(construct_id="Invalid", steps=steps)
