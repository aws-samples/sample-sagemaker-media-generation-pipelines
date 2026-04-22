# Feature: model-validation-integration, Property 1: DAG-flow shard round trip
# Feature: model-validation-integration, Property 2: Sidecar key-set preservation
# Feature: model-validation-integration, Property 3: Invalid entry filtering
# Feature: model-validation-integration, Property 4: VBench exclude_none serialization
"""
Property-based tests for the four model-validation-integration correctness properties.

**Validates: Requirements 1.2, 1.4, 1.5, 5.1, 6.2, 6.3, 7.2, 7.3, 8.2, 8.3, 9.2, 9.3,
             10.3, 11.1, 11.3, 12.2, 12.3, 14.1, 14.2, 14.3**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from processing_job.common.models import (
    AudioEntry,
    AudioSidecarEntry,
    CaptioningSidecarEntry,
    ImageSidecarEntry,
    VBenchMetrics,
    VideoSidecarEntry,
    VisualEntry,
)

pytestmark = pytest.mark.model_validation


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

visual_entry_st = st.builds(
    VisualEntry,
    id=st.text(min_size=1, max_size=50),
    prompt=st.text(max_size=200),
    image=st.text(min_size=1, max_size=100),
)

audio_entry_st = st.builds(
    AudioEntry,
    id=st.text(min_size=1, max_size=50),
    tags=st.text(max_size=200),
    lyrics=st.text(max_size=500),
    bpm=st.integers(min_value=1, max_value=10000),
    duration=st.integers(min_value=1, max_value=10000),
    timesignature=st.text(min_size=1, max_size=5),
    language=st.text(min_size=1, max_size=10),
    keyscale=st.text(min_size=1, max_size=20),
)

video_sidecar_st = st.builds(
    VideoSidecarEntry,
    input_id=st.text(min_size=1, max_size=50),
    model=st.text(min_size=1, max_size=50),
    mode=st.text(min_size=1, max_size=20),
    prompt=st.text(max_size=200),
    source_filename=st.text(min_size=1, max_size=100),
    seed=st.integers(min_value=0, max_value=10000),
    generation_index=st.integers(min_value=-1000, max_value=1000),
)

image_sidecar_st = st.builds(
    ImageSidecarEntry,
    input_id=st.text(min_size=1, max_size=50),
    model=st.text(min_size=1, max_size=50),
    mode=st.text(min_size=1, max_size=20),
    prompt=st.text(max_size=200),
    seed=st.integers(min_value=0, max_value=10000),
    generation_index=st.integers(min_value=-1000, max_value=1000),
)

audio_sidecar_st = st.builds(
    AudioSidecarEntry,
    input_id=st.text(min_size=1, max_size=50),
    model=st.text(min_size=1, max_size=50),
    mode=st.text(min_size=1, max_size=20),
    prompt=st.text(max_size=200),
    tags=st.text(max_size=200),
    lyrics=st.text(max_size=500),
    seed=st.integers(min_value=0, max_value=10000),
    bpm=st.integers(min_value=1, max_value=10000),
    duration=st.integers(min_value=1, max_value=10000),
    keyscale=st.text(min_size=1, max_size=20),
    generation_index=st.integers(min_value=-1000, max_value=1000),
)

captioning_sidecar_st = st.builds(
    CaptioningSidecarEntry,
    input_id=st.text(min_size=1, max_size=50),
    model=st.text(min_size=1, max_size=50),
    mode=st.text(min_size=1, max_size=20),
    prompt=st.text(max_size=500),
    source_filename=st.text(min_size=1, max_size=100),
    generation_index=st.integers(min_value=-1000, max_value=1000),
)

_optional_float = st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False))

vbench_st = st.builds(
    VBenchMetrics,
    subject_consistency=_optional_float,
    background_consistency=_optional_float,
    motion_smoothness=_optional_float,
    dynamic_degree=_optional_float,
    aesthetic_quality=_optional_float,
    imaging_quality=_optional_float,
    temporal_flickering=_optional_float,
    temporal_style=_optional_float,
    overall_consistency=_optional_float,
    human_action=_optional_float,
)


# ---------------------------------------------------------------------------
# Property 1: DAG-flow shard round trip
# ---------------------------------------------------------------------------


class TestDAGFlowRoundTrip:
    """For any valid VisualEntry or AudioEntry, serializing via model_dump_json()
    and deserializing via model_validate_json() produces an equal instance.

    **Validates: Requirements 1.2, 1.4, 1.5, 5.1, 14.1, 14.2, 14.3**
    """

    # Feature: model-validation-integration, Property 1: DAG-flow shard round trip

    @given(instance=visual_entry_st)
    @settings(max_examples=100)
    def test_visual_entry_round_trip(self, instance: VisualEntry) -> None:
        json_str = instance.model_dump_json()
        restored = VisualEntry.model_validate_json(json_str)
        assert restored == instance

    @given(instance=audio_entry_st)
    @settings(max_examples=100)
    def test_audio_entry_round_trip(self, instance: AudioEntry) -> None:
        json_str = instance.model_dump_json()
        restored = AudioEntry.model_validate_json(json_str)
        assert restored == instance


# ---------------------------------------------------------------------------
# Property 2: Sidecar key-set preservation
# ---------------------------------------------------------------------------

# Expected key sets per sidecar model
_EXPECTED_KEYS: dict[type, set[str]] = {
    VideoSidecarEntry: {"input_id", "model", "mode", "prompt", "source_filename", "seed", "generation_index"},
    ImageSidecarEntry: {"input_id", "model", "mode", "prompt", "seed", "generation_index"},
    AudioSidecarEntry: {
        "input_id",
        "model",
        "mode",
        "prompt",
        "tags",
        "lyrics",
        "seed",
        "bpm",
        "duration",
        "keyscale",
        "generation_index",
    },
    CaptioningSidecarEntry: {
        "input_id",
        "model",
        "mode",
        "prompt",
        "source_filename",
        "generation_index",
    },
}

_SIDECAR_STRATEGIES: dict[type, st.SearchStrategy] = {
    VideoSidecarEntry: video_sidecar_st,
    ImageSidecarEntry: image_sidecar_st,
    AudioSidecarEntry: audio_sidecar_st,
    CaptioningSidecarEntry: captioning_sidecar_st,
}


class TestSidecarKeySetPreservation:
    """For any valid sidecar model instance, model_dump().keys() equals exactly
    the expected key set — no extra, no missing keys.

    **Validates: Requirements 6.2, 6.3, 7.2, 7.3, 8.2, 8.3, 9.2, 9.3, 11.1, 11.3**
    """

    # Feature: model-validation-integration, Property 2: Sidecar key-set preservation

    @pytest.mark.parametrize(
        "model_cls",
        list(_EXPECTED_KEYS.keys()),
        ids=lambda c: c.__name__,
    )
    @given(data=st.data())
    @settings(max_examples=100)
    def test_sidecar_key_set(self, model_cls: type, data: st.DataObject) -> None:
        instance = data.draw(_SIDECAR_STRATEGIES[model_cls])
        dumped = instance.model_dump()
        assert set(dumped.keys()) == _EXPECTED_KEYS[model_cls]


# ---------------------------------------------------------------------------
# Property 3: Invalid entry filtering
# ---------------------------------------------------------------------------

# Strategy helpers for generating valid and invalid entry dicts

_valid_visual_dict_st = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1, max_size=50),
        "prompt": st.text(max_size=200),
        "image": st.text(min_size=1, max_size=100),
    }
)

_invalid_missing_field_st = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1, max_size=50),
        # missing "prompt" and "image"
    }
)

_invalid_wrong_type_st = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=0, max_value=1000),  # should be str
        "prompt": st.text(max_size=200),
        "image": st.text(min_size=1, max_size=100),
    }
)

_invalid_extra_field_st = st.fixed_dictionaries(
    {
        "id": st.text(min_size=1, max_size=50),
        "prompt": st.text(max_size=200),
        "image": st.text(min_size=1, max_size=100),
        "unexpected_field": st.text(min_size=1, max_size=20),
    }
)

_invalid_dict_st = st.one_of(
    _invalid_missing_field_st,
    _invalid_wrong_type_st,
    _invalid_extra_field_st,
)

# A tagged entry: (dict, is_valid)
_tagged_entry_st = st.one_of(
    _valid_visual_dict_st.map(lambda d: (d, True)),
    _invalid_dict_st.map(lambda d: (d, False)),
)


def _filter_valid_entries(entries: list[dict], model_cls: type) -> list:
    """Validate each dict, skip ValidationError entries, return valid instances."""
    result = []
    for raw in entries:
        try:
            result.append(model_cls.model_validate(raw))
        except ValidationError:
            continue
    return result


class TestInvalidEntryFiltering:
    """For any list of dicts with a mix of valid and invalid entries, validating
    each dict and skipping ValidationError entries produces a result list
    containing exactly the valid entries, in order.

    **Validates: Requirements 1.3, 2.2, 3.2, 4.2, 10.2, 12.2, 12.3**
    """

    # Feature: model-validation-integration, Property 3: Invalid entry filtering

    @given(tagged_entries=st.lists(_tagged_entry_st, min_size=1, max_size=20))
    @settings(max_examples=100)
    def test_filter_keeps_only_valid_entries(self, tagged_entries: list[tuple[dict, bool]]) -> None:
        all_dicts = [d for d, _ in tagged_entries]
        expected_valid = [d for d, is_valid in tagged_entries if is_valid]

        result = _filter_valid_entries(all_dicts, VisualEntry)

        # Same count
        assert len(result) == len(expected_valid)
        # Each result matches the corresponding valid dict
        for instance, raw in zip(result, expected_valid):
            assert instance.id == raw["id"]
            assert instance.prompt == raw["prompt"]
            assert instance.image == raw["image"]


# ---------------------------------------------------------------------------
# Property 4: VBench exclude_none serialization
# ---------------------------------------------------------------------------


class TestVBenchExcludeNone:
    """For any VBenchMetrics instance with a mix of None and float fields,
    model_dump(exclude_none=True) produces a dict containing only non-None
    fields, and every value is a float.

    **Validates: Requirements 10.3**
    """

    # Feature: model-validation-integration, Property 4: VBench exclude_none serialization

    @given(instance=vbench_st)
    @settings(max_examples=100)
    def test_exclude_none_contains_only_floats(self, instance: VBenchMetrics) -> None:
        dumped = instance.model_dump(exclude_none=True)

        # Every value must be a float
        for key, value in dumped.items():
            assert isinstance(value, float), f"{key}={value!r} is not a float"

    @given(instance=vbench_st)
    @settings(max_examples=100)
    def test_exclude_none_matches_non_none_fields(self, instance: VBenchMetrics) -> None:
        dumped = instance.model_dump(exclude_none=True)
        full = instance.model_dump()

        # dumped keys should be exactly the non-None keys from the full dump
        expected_keys = {k for k, v in full.items() if v is not None}
        assert set(dumped.keys()) == expected_keys
