# Feature: cicd-pipeline, Property 1: CicdConfig valid round-trip
"""
Property-based tests for CicdConfig validation.

Validates: Requirements 5.1, 7.2, 21.1, 22.1

Property 1: CicdConfig valid round-trip
- For any valid combination of CicdConfig field values, constructing a
  CicdConfig and then calling .model_dump() followed by CicdConfig(**dumped)
  should produce an equivalent config object.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from config.config import CicdConfig

# --- Strategies ---

# Valid compute types
compute_type_st = st.sampled_from(["SMALL", "MEDIUM", "LARGE", "X2_LARGE"])

# Valid CicdConfig field strategies
enabled_st = st.booleans()
notification_email_st = st.one_of(st.none(), st.text(min_size=1, max_size=100))
source_excludes_st = st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=10)
timeout_minutes_st = st.integers(min_value=1, max_value=480)
test_commands_st = st.dictionaries(
    keys=st.text(min_size=1, max_size=50),
    values=st.text(min_size=1, max_size=200),
    min_size=1,
    max_size=5,
)
rollback_st = st.booleans()
test_a2i_st = st.booleans()
pipeline_configs_st = st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10)


class TestCicdConfigValidRoundTrip:
    """Valid CicdConfig instances round-trip through model_dump and reconstruction."""

    @given(
        enabled=enabled_st,
        notification_email=notification_email_st,
        source_excludes=source_excludes_st,
        compute_type=compute_type_st,
        timeout_minutes=timeout_minutes_st,
        test_commands=test_commands_st,
        rollback=rollback_st,
        test_a2i=test_a2i_st,
    )
    @settings(max_examples=100)
    def test_valid_config_round_trips(
        self,
        enabled: bool,
        notification_email: str | None,
        source_excludes: list[str],
        compute_type: str,
        timeout_minutes: int,
        test_commands: dict[str, str],
        rollback: bool,
        test_a2i: bool,
    ) -> None:
        """
        **Validates: Requirements 5.1, 7.2, 21.1, 22.1**

        For any valid combination of CicdConfig field values, constructing a
        CicdConfig and then calling .model_dump() followed by
        CicdConfig(**dumped) should produce an equivalent config object.
        """
        # pipeline_configs must be a subset of test_commands keys
        pipeline_configs = list(test_commands.keys())
        config = CicdConfig(
            enabled=enabled,
            notification_email=notification_email,
            source_excludes=source_excludes,
            compute_type=compute_type,
            timeout_minutes=timeout_minutes,
            test_commands=test_commands,
            rollback=rollback,
            test_a2i=test_a2i,
            pipeline_configs=pipeline_configs,
        )

        dumped = config.model_dump()
        reconstructed = CicdConfig(**dumped)

        assert config == reconstructed


# Feature: cicd-pipeline, Property 2: CicdConfig rejects unknown fields
"""
Property 2: CicdConfig rejects unknown fields

For any valid CicdConfig base values and any additional field name not in
the CicdConfig schema, constructing a CicdConfig with the extra field
should raise a pydantic.ValidationError with a message about extra inputs
not being permitted.

Validates: Requirements 5.3, 14.8
"""

import pytest
from pydantic import ValidationError

# All field names in the CicdConfig schema
CICD_CONFIG_FIELDS: set[str] = {
    "enabled",
    "notification_email",
    "source_excludes",
    "compute_type",
    "timeout_minutes",
    "test_commands",
    "rollback",
    "test_a2i",
    "shared_prefix",
    "pipeline_configs",
    "input_data",
}


def _valid_cicd_config_dict() -> dict:
    """Return a dict of valid default values for CicdConfig."""
    return {
        "enabled": True,
        "notification_email": None,
        "source_excludes": [
            ".venv/",
            "cdk.out/",
            ".git/",
            "__pycache__/",
            ".hypothesis/",
            "node_modules/",
            "security/",
        ],
        "compute_type": "SMALL",
        "timeout_minutes": 60,
        "test_commands": {
            "config_vrag.yaml": "uv run pytest tests/unit/ -x --no-header -q",
        },
        "rollback": True,
        "test_a2i": False,
        "pipeline_configs": ["config_vrag.yaml"],
    }


class TestCicdConfigRejectsUnknownFields:
    """CicdConfig with extra='forbid' rejects any field not in the schema."""

    @given(extra_field_name=st.text(min_size=1, max_size=50).filter(lambda s: s not in CICD_CONFIG_FIELDS))
    @settings(max_examples=100)
    def test_extra_field_rejected(self, extra_field_name: str) -> None:
        """
        **Validates: Requirements 5.3, 14.8**

        For any field name not in the CicdConfig schema, constructing a
        CicdConfig with that extra field should raise a ValidationError.
        """
        base = _valid_cicd_config_dict()
        base[extra_field_name] = "unexpected"
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CicdConfig(**base)


# Feature: cicd-pipeline, Property 3: CicdConfig pipeline_configs must be non-empty
"""
Property 3: CicdConfig pipeline_configs must be non-empty

For any otherwise-valid CicdConfig values, setting pipeline_configs to an
empty list should raise a pydantic.ValidationError, ensuring the pipeline
always has at least one config to deploy.

Validates: Requirements 21.8
"""


class TestCicdConfigPipelineConfigsNonEmpty:
    """CicdConfig rejects an empty pipeline_configs list."""

    @given(
        enabled=enabled_st,
        notification_email=notification_email_st,
        source_excludes=source_excludes_st,
        compute_type=compute_type_st,
        timeout_minutes=timeout_minutes_st,
        test_commands=test_commands_st,
        rollback=rollback_st,
        test_a2i=test_a2i_st,
    )
    @settings(max_examples=100)
    def test_empty_pipeline_configs_rejected(
        self,
        enabled: bool,
        notification_email: str | None,
        source_excludes: list[str],
        compute_type: str,
        timeout_minutes: int,
        test_commands: dict[str, str],
        rollback: bool,
        test_a2i: bool,
    ) -> None:
        """
        **Validates: Requirements 21.8**

        For any otherwise-valid CicdConfig values, setting pipeline_configs
        to an empty list should raise a ValidationError.
        """
        with pytest.raises(ValidationError):
            CicdConfig(
                enabled=enabled,
                notification_email=notification_email,
                source_excludes=source_excludes,
                compute_type=compute_type,
                timeout_minutes=timeout_minutes,
                test_commands=test_commands,
                rollback=rollback,
                test_a2i=test_a2i,
                pipeline_configs=[],
            )


# Feature: cicd-pipeline, Property 4: Config hash determinism and sensitivity
"""
Property 4: Config hash determinism and sensitivity

For any valid list of s3_downloads entries (each with url, path, and optional
extract fields), computing the SHA-256 hash of the YAML-serialized s3_downloads
section twice should produce the same hash. Additionally, for any two distinct
s3_downloads lists that differ in content or ordering, the computed hashes
should differ.

Validates: Requirements 10.9
"""

import hashlib

import yaml

pytestmark = pytest.mark.cicd


def compute_config_hash(s3_downloads: list[dict]) -> str:
    """Compute SHA-256 hex digest of YAML-serialized s3_downloads list."""
    return hashlib.sha256(yaml.dump(s3_downloads, default_flow_style=False).encode()).hexdigest()


# Strategy for a single s3_downloads entry
s3_download_entry_st = st.fixed_dictionaries(
    {
        "url": st.text(min_size=1, max_size=200),
        "path": st.text(min_size=1, max_size=200),
        "extract": st.booleans(),
    }
)

# Strategy for a list of s3_downloads entries
s3_downloads_st = st.lists(s3_download_entry_st, min_size=0, max_size=10)


class TestConfigHashDeterminismAndSensitivity:
    """Config hash is deterministic for same input and sensitive to changes."""

    @given(s3_downloads=s3_downloads_st)
    @settings(max_examples=100)
    def test_config_hash_deterministic(self, s3_downloads: list[dict]) -> None:
        """
        **Validates: Requirements 10.9**

        For any valid list of s3_downloads entries, computing the SHA-256 hash
        of the YAML-serialized list twice should produce the same hash.
        """
        hash1 = compute_config_hash(s3_downloads)
        hash2 = compute_config_hash(s3_downloads)
        assert hash1 == hash2

    @given(
        downloads_a=s3_downloads_st,
        downloads_b=s3_downloads_st,
    )
    @settings(max_examples=100)
    def test_config_hash_sensitive(self, downloads_a: list[dict], downloads_b: list[dict]) -> None:
        """
        **Validates: Requirements 10.9**

        For any two distinct s3_downloads lists that differ in content or
        ordering, the computed hashes should differ.
        """
        from hypothesis import assume

        assume(downloads_a != downloads_b)

        hash_a = compute_config_hash(downloads_a)
        hash_b = compute_config_hash(downloads_b)
        assert hash_a != hash_b


# Feature: unit-test-reorganization, Property 3: test_commands accepts arbitrary string values
# Feature: unit-test-reorganization, Property 4: Pipeline configs must have test commands

# Strategy for valid pipeline config filenames
_config_filename_st = st.from_regex(r"[a-z][a-z0-9_]{0,30}\.yaml", fullmatch=True)


class TestTestCommandsAcceptsArbitraryValues:
    """
    Property 3: For any dict[str, str] where keys are valid pipeline config
    filenames and values are arbitrary non-empty strings,
    CicdConfig(test_commands=d, pipeline_configs=list(d.keys())) validates.

    **Validates: Requirements 3.2, 3.9**
    """

    @given(
        test_commands=st.dictionaries(
            keys=_config_filename_st,
            values=st.text(min_size=1, max_size=200),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_arbitrary_test_commands_accepted(self, test_commands: dict[str, str]) -> None:
        """
        **Validates: Requirements 3.2, 3.9**
        """
        cfg = CicdConfig(
            test_commands=test_commands,
            pipeline_configs=list(test_commands.keys()),
        )
        assert cfg.test_commands == test_commands


class TestPipelineConfigsMustHaveTestCommands:
    """
    Property 4: For any CicdConfig where pipeline_configs contains a filename
    not in test_commands, validation raises ValidationError.

    **Validates: Requirements 3.3**
    """

    @given(
        existing_key=_config_filename_st,
        missing_key=_config_filename_st,
    )
    @settings(max_examples=100, deadline=None)
    def test_missing_test_command_raises(self, existing_key: str, missing_key: str) -> None:
        """
        **Validates: Requirements 3.3**
        """
        from hypothesis import assume

        assume(existing_key != missing_key)

        with pytest.raises(ValidationError, match="missing from test_commands"):
            CicdConfig(
                test_commands={existing_key: "uv run pytest"},
                pipeline_configs=[existing_key, missing_key],
            )
