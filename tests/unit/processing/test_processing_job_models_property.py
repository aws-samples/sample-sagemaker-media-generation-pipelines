# Feature: processing-job-models, Property 1: Serialization round trip
"""
Property-based tests for processing job Pydantic models.

**Validates: Requirements 11.1, 11.2, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from processing_job.common.models import (
    AudioEntry,
    AudioSidecarEntry,
    BaseEntry,
    BaseSidecarEntry,
    CaptioningSidecarEntry,
    ImageSidecarEntry,
    VBenchMetrics,
    VideoSidecarEntry,
    VisualEntry,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies for each model
# ---------------------------------------------------------------------------

base_entry_st = st.builds(BaseEntry, id=st.text(min_size=1, max_size=50))

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
    bpm=st.integers(min_value=1, max_value=300),
    duration=st.integers(min_value=1, max_value=600),
    timesignature=st.text(min_size=1, max_size=5),
    language=st.text(min_size=1, max_size=10),
    keyscale=st.text(min_size=1, max_size=20),
)

base_sidecar_st = st.builds(
    BaseSidecarEntry,
    input_id=st.text(min_size=1, max_size=50),
    model=st.text(min_size=1, max_size=50),
    mode=st.text(min_size=1, max_size=20),
    generation_index=st.integers(min_value=-1000, max_value=1000),
)

video_sidecar_st = st.builds(
    VideoSidecarEntry,
    input_id=st.text(min_size=1, max_size=50),
    model=st.text(min_size=1, max_size=50),
    mode=st.text(min_size=1, max_size=20),
    prompt=st.text(max_size=200),
    source_filename=st.text(min_size=1, max_size=100),
    seed=st.integers(min_value=0, max_value=2**31),
    generation_index=st.integers(min_value=-1000, max_value=1000),
)

image_sidecar_st = st.builds(
    ImageSidecarEntry,
    input_id=st.text(min_size=1, max_size=50),
    model=st.text(min_size=1, max_size=50),
    mode=st.text(min_size=1, max_size=20),
    prompt=st.text(max_size=200),
    seed=st.integers(min_value=0, max_value=2**31),
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
    seed=st.integers(min_value=0, max_value=2**31),
    bpm=st.integers(min_value=1, max_value=300),
    duration=st.integers(min_value=1, max_value=600),
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
# Property 1: Serialization round trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    """For every model, model_validate_json(instance.model_dump_json()) == instance
    and model_validate(instance.model_dump()) == instance.

    **Validates: Requirements 11.1, 11.2, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1**
    """

    @given(instance=base_entry_st)
    @settings(max_examples=100)
    def test_base_entry_round_trip(self, instance: BaseEntry) -> None:
        assert BaseEntry.model_validate_json(instance.model_dump_json()) == instance
        assert BaseEntry.model_validate(instance.model_dump()) == instance

    @given(instance=visual_entry_st)
    @settings(max_examples=100)
    def test_visual_entry_round_trip(self, instance: VisualEntry) -> None:
        assert VisualEntry.model_validate_json(instance.model_dump_json()) == instance
        assert VisualEntry.model_validate(instance.model_dump()) == instance

    @given(instance=audio_entry_st)
    @settings(max_examples=100)
    def test_audio_entry_round_trip(self, instance: AudioEntry) -> None:
        assert AudioEntry.model_validate_json(instance.model_dump_json()) == instance
        assert AudioEntry.model_validate(instance.model_dump()) == instance

    @given(instance=base_sidecar_st)
    @settings(max_examples=100)
    def test_base_sidecar_entry_round_trip(self, instance: BaseSidecarEntry) -> None:
        assert BaseSidecarEntry.model_validate_json(instance.model_dump_json()) == instance
        assert BaseSidecarEntry.model_validate(instance.model_dump()) == instance

    @given(instance=video_sidecar_st)
    @settings(max_examples=100)
    def test_video_sidecar_entry_round_trip(self, instance: VideoSidecarEntry) -> None:
        assert VideoSidecarEntry.model_validate_json(instance.model_dump_json()) == instance
        assert VideoSidecarEntry.model_validate(instance.model_dump()) == instance

    @given(instance=image_sidecar_st)
    @settings(max_examples=100)
    def test_image_sidecar_entry_round_trip(self, instance: ImageSidecarEntry) -> None:
        assert ImageSidecarEntry.model_validate_json(instance.model_dump_json()) == instance
        assert ImageSidecarEntry.model_validate(instance.model_dump()) == instance

    @given(instance=audio_sidecar_st)
    @settings(max_examples=100)
    def test_audio_sidecar_entry_round_trip(self, instance: AudioSidecarEntry) -> None:
        assert AudioSidecarEntry.model_validate_json(instance.model_dump_json()) == instance
        assert AudioSidecarEntry.model_validate(instance.model_dump()) == instance

    @given(instance=captioning_sidecar_st)
    @settings(max_examples=100)
    def test_captioning_sidecar_entry_round_trip(self, instance: CaptioningSidecarEntry) -> None:
        assert CaptioningSidecarEntry.model_validate_json(instance.model_dump_json()) == instance
        assert CaptioningSidecarEntry.model_validate(instance.model_dump()) == instance

    @given(instance=vbench_st)
    @settings(max_examples=100)
    def test_vbench_metrics_round_trip(self, instance: VBenchMetrics) -> None:
        assert VBenchMetrics.model_validate_json(instance.model_dump_json()) == instance
        assert VBenchMetrics.model_validate(instance.model_dump()) == instance


# Feature: processing-job-models, Property 2: Strict mode rejects wrong types
# ---------------------------------------------------------------------------
# Property 2: Strict mode rejects wrong types
# ---------------------------------------------------------------------------


class TestStrictTypeRejection:
    """For models with int fields, providing a str value raises ValidationError.

    **Validates: Requirements 1.3, 3.3, 3.4, 10.2**
    """

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_audio_entry_bpm_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            AudioEntry(id="x", tags="t", lyrics="l", bpm=bad_value, duration=120)

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_audio_entry_duration_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            AudioEntry(id="x", tags="t", lyrics="l", bpm=120, duration=bad_value)

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_base_sidecar_generation_index_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            BaseSidecarEntry(input_id="i", model="m", mode="x", generation_index=bad_value)

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_video_sidecar_seed_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            VideoSidecarEntry(
                input_id="i",
                model="m",
                mode="x",
                prompt="p",
                source_filename="img",
                seed=bad_value,
                generation_index=0,
            )

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_image_sidecar_seed_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            ImageSidecarEntry(
                input_id="i",
                model="m",
                mode="x",
                prompt="p",
                seed=bad_value,
                generation_index=0,
            )

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_audio_sidecar_seed_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            AudioSidecarEntry(
                input_id="i",
                model="m",
                mode="x",
                tags="t",
                lyrics="l",
                seed=bad_value,
                bpm=120,
                duration=120,
                keyscale="C major",
                generation_index=0,
            )

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_audio_sidecar_bpm_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            AudioSidecarEntry(
                input_id="i",
                model="m",
                mode="x",
                tags="t",
                lyrics="l",
                seed=42,
                bpm=bad_value,
                duration=120,
                keyscale="C major",
                generation_index=0,
            )

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_audio_sidecar_duration_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            AudioSidecarEntry(
                input_id="i",
                model="m",
                mode="x",
                tags="t",
                lyrics="l",
                seed=42,
                bpm=120,
                duration=bad_value,
                keyscale="C major",
                generation_index=0,
            )

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_captioning_sidecar_image_size_bytes_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            CaptioningSidecarEntry(
                input_id="i",
                model="m",
                mode="x",
                prompt="c",
                source_filename="f.png",
                generation_index=bad_value,
            )

    @given(bad_value=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_captioning_sidecar_generation_index_rejects_str(self, bad_value: str) -> None:
        with pytest.raises(Exception, match="validation error"):
            CaptioningSidecarEntry(
                input_id="i",
                model="m",
                mode="x",
                prompt="c",
                source_filename="f.png",
                generation_index=bad_value,
            )


# Feature: processing-job-models, Property 3: Extra fields rejected
# ---------------------------------------------------------------------------
# Property 3: Extra fields rejected
# ---------------------------------------------------------------------------

from pydantic import ValidationError

pytestmark = pytest.mark.processing


# Map each model class to a dict of valid kwargs for construction
_VALID_KWARGS: dict[type, dict] = {
    BaseEntry: {"id": "x"},
    VisualEntry: {"id": "x", "prompt": "p", "image": "img"},
    AudioEntry: {"id": "x", "tags": "t", "lyrics": "l"},
    BaseSidecarEntry: {"input_id": "i", "model": "m", "mode": "x", "generation_index": 0},
    VideoSidecarEntry: {
        "input_id": "i",
        "model": "m",
        "mode": "x",
        "prompt": "p",
        "source_filename": "img",
        "seed": 42,
        "generation_index": 0,
    },
    ImageSidecarEntry: {
        "input_id": "i",
        "model": "m",
        "mode": "x",
        "prompt": "p",
        "seed": 42,
        "generation_index": 0,
    },
    AudioSidecarEntry: {
        "input_id": "i",
        "model": "m",
        "mode": "x",
        "prompt": "",
        "tags": "t",
        "lyrics": "l",
        "seed": 42,
        "bpm": 120,
        "duration": 120,
        "keyscale": "C major",
        "generation_index": 0,
    },
    CaptioningSidecarEntry: {
        "input_id": "i",
        "model": "m",
        "mode": "x",
        "prompt": "c",
        "source_filename": "f.png",
        "generation_index": 0,
    },
    VBenchMetrics: {},
}

_ALL_MODELS = list(_VALID_KWARGS.keys())

# Strategy: generate a random field name that is NOT one of the model's defined fields
_extra_field_name_st = st.text(min_size=1, max_size=30, alphabet=st.characters(categories=("L",)))


class TestExtraFieldRejection:
    """For every model class, constructing with an undefined field name raises ValidationError.

    **Validates: Requirements 1.4, 10.3**
    """

    @pytest.mark.parametrize("model_cls", _ALL_MODELS, ids=lambda c: c.__name__)
    @given(data=st.data())
    @settings(max_examples=100)
    def test_extra_field_rejected(self, model_cls: type, data: st.DataObject) -> None:
        defined_fields = set(model_cls.model_fields.keys())
        extra_name = data.draw(_extra_field_name_st.filter(lambda n: n not in defined_fields))
        kwargs = {**_VALID_KWARGS[model_cls], extra_name: "unexpected"}
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            model_cls(**kwargs)
