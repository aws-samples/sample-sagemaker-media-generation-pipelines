# Feature: processing-job-models, Task 3.1: Structural and example-based tests
"""
Unit tests for processing job Pydantic models.

**Validates: Requirements 1.1–1.4, 2.1–2.6, 3.1–3.5, 4.1–4.4, 5.1–5.4,
6.1–6.3, 7.1–7.3, 8.1–8.3, 9.3–9.5, 10.1–10.3**
"""

import pytest
from pydantic import ValidationError

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

pytestmark = pytest.mark.processing


ALL_MODELS = [
    BaseEntry,
    VisualEntry,
    AudioEntry,
    BaseSidecarEntry,
    VideoSidecarEntry,
    ImageSidecarEntry,
    AudioSidecarEntry,
    CaptioningSidecarEntry,
    VBenchMetrics,
]


class TestModelConfigStrict:
    """Every model has strict=True and extra='forbid' (Req 10.1)."""

    @pytest.mark.parametrize("cls", ALL_MODELS, ids=lambda c: c.__name__)
    def test_strict_true(self, cls: type) -> None:
        assert cls.model_config.get("strict") is True

    @pytest.mark.parametrize("cls", ALL_MODELS, ids=lambda c: c.__name__)
    def test_extra_forbid(self, cls: type) -> None:
        assert cls.model_config.get("extra") == "forbid"


class TestInheritance:
    """Verify inheritance hierarchy (Req 9.4, 9.5, 4.4)."""

    def test_visual_entry_subclasses_base_entry(self) -> None:
        assert issubclass(VisualEntry, BaseEntry)

    def test_audio_entry_subclasses_base_entry(self) -> None:
        assert issubclass(AudioEntry, BaseEntry)

    def test_video_sidecar_subclasses_base_sidecar(self) -> None:
        assert issubclass(VideoSidecarEntry, BaseSidecarEntry)

    def test_image_sidecar_subclasses_base_sidecar(self) -> None:
        assert issubclass(ImageSidecarEntry, BaseSidecarEntry)

    def test_audio_sidecar_subclasses_base_sidecar(self) -> None:
        assert issubclass(AudioSidecarEntry, BaseSidecarEntry)

    def test_captioning_sidecar_subclasses_base_sidecar(self) -> None:
        assert issubclass(CaptioningSidecarEntry, BaseSidecarEntry)

    def test_vbench_does_not_subclass_base_entry(self) -> None:
        assert not issubclass(VBenchMetrics, BaseEntry)


class TestVisualEntryFields:
    """VisualEntry field set is exactly {id, prompt, image} (Req 2.2)."""

    def test_field_set(self) -> None:
        assert set(VisualEntry.model_fields.keys()) == {"id", "prompt", "image"}

    def test_valid_instance(self) -> None:
        entry = VisualEntry(id="abc", prompt="a cat", image="s3://bucket/img.png")
        assert entry.id == "abc"
        assert entry.prompt == "a cat"
        assert entry.image == "s3://bucket/img.png"


class TestAudioEntryDefaults:
    """AudioEntry defaults match codebase values (Req 3.2)."""

    def test_defaults(self) -> None:
        entry = AudioEntry(id="x", tags="pop", lyrics="la la")
        assert entry.bpm == 120
        assert entry.duration == 120
        assert entry.timesignature == "4"
        assert entry.language == "en"
        assert entry.keyscale == "C major"

    def test_field_set(self) -> None:
        assert set(AudioEntry.model_fields.keys()) == {
            "id",
            "tags",
            "lyrics",
            "bpm",
            "duration",
            "timesignature",
            "language",
            "keyscale",
        }


class TestVBenchMetricsDefaults:
    """VBenchMetrics() with no args succeeds and all fields are None (Req 4.3)."""

    def test_no_args(self) -> None:
        m = VBenchMetrics()
        for field_name in VBenchMetrics.model_fields:
            assert getattr(m, field_name) is None

    def test_field_count(self) -> None:
        assert len(VBenchMetrics.model_fields) == 10


class TestSidecarFieldSets:
    """Each sidecar model's field names match the corresponding dict keys from the codebase."""

    def test_video_sidecar_fields(self) -> None:
        # From t2v/main.py video_metadata dict
        expected = {"input_id", "model", "mode", "prompt", "source_filename", "seed", "generation_index"}
        assert set(VideoSidecarEntry.model_fields.keys()) == expected

    def test_image_sidecar_fields(self) -> None:
        # From t2i/main.py image_metadata dict
        expected = {"input_id", "model", "mode", "prompt", "seed", "generation_index"}
        assert set(ImageSidecarEntry.model_fields.keys()) == expected

    def test_audio_sidecar_fields(self) -> None:
        # From t2a/main.py audio_metadata dict
        expected = {
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
        }
        assert set(AudioSidecarEntry.model_fields.keys()) == expected

    def test_captioning_sidecar_fields(self) -> None:
        # From captioning/main.py caption_metadata dict
        expected = {
            "input_id",
            "model",
            "mode",
            "prompt",
            "source_filename",
            "generation_index",
        }
        assert set(CaptioningSidecarEntry.model_fields.keys()) == expected


class TestEdgeCases:
    """Edge cases: empty strings, negative ints, wrong types, extra fields."""

    def test_empty_string_prompt_accepted(self) -> None:
        entry = VisualEntry(id="x", prompt="", image="img.png")
        assert entry.prompt == ""

    def test_negative_generation_index_accepted(self) -> None:
        entry = VideoSidecarEntry(
            input_id="x",
            model="ltx23",
            mode="t2v",
            prompt="p",
            source_filename="i",
            seed=42,
            generation_index=-1,
        )
        assert entry.generation_index == -1

    def test_wrong_type_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            BaseEntry(id=123)  # type: ignore[arg-type]

    def test_string_for_int_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            AudioEntry(id="x", tags="t", lyrics="l", bpm="fast")  # type: ignore[arg-type]

    def test_extra_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VisualEntry(id="x", prompt="p", image="i", extra="bad")  # type: ignore[call-arg]

    def test_vbench_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            VBenchMetrics(unknown_metric=0.5)  # type: ignore[call-arg]
