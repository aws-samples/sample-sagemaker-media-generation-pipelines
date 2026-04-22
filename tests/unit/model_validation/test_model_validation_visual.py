"""Unit tests for visual container (t2v/i2v/flf2v) model validation logic.

Tests VisualEntry validation and VideoSidecarEntry shape.
"""

import pytest
from pydantic import ValidationError

from processing_job.common.models import VideoSidecarEntry, VisualEntry

pytestmark = pytest.mark.model_validation


EXPECTED_VIDEO_SIDECAR_KEYS = {
    "input_id",
    "model",
    "mode",
    "prompt",
    "source_filename",
    "seed",
    "generation_index",
}


class TestVisualEntryValidation:
    """VisualEntry validates input shard dicts from load_inputs()."""

    def test_valid_dict_validates(self) -> None:
        raw = {"id": "img001", "prompt": "a flying car", "image": "car.png"}
        entry = VisualEntry.model_validate(raw)
        assert entry.id == "img001"
        assert entry.prompt == "a flying car"
        assert entry.image == "car.png"

    def test_missing_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            VisualEntry.model_validate({"id": "x", "image": "i.png"})

    def test_missing_image_defaults_to_empty(self) -> None:
        entry = VisualEntry.model_validate({"id": "x", "prompt": "p"})
        assert entry.image == ""

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VisualEntry.model_validate({"id": "x", "prompt": "p", "image": "i", "foo": "bar"})

    def test_int_id_rejected_strict(self) -> None:
        with pytest.raises(ValidationError):
            VisualEntry.model_validate({"id": 42, "prompt": "p", "image": "i"})

    def test_invalid_entries_filtered(self) -> None:
        """Simulate container loop: valid entries pass, invalid are skipped."""
        raw_inputs = [
            {"id": "ok1", "prompt": "good", "image": "a.png"},
            {"id": 123, "prompt": "bad type", "image": "b.png"},
            {"id": "ok2", "prompt": "also good", "image": "c.png"},
        ]
        validated = []
        for raw in raw_inputs:
            try:
                validated.append(VisualEntry.model_validate(raw))
            except ValidationError:
                continue
        assert len(validated) == 2
        assert validated[0].id == "ok1"
        assert validated[1].id == "ok2"


class TestVideoSidecarEntryShape:
    """VideoSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_model_dump_keys(self) -> None:
        entry = VideoSidecarEntry(
            input_id="v1",
            model="ltx23",
            mode="t2v",
            prompt="a cat",
            source_filename="cat.png",
            seed=42,
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_VIDEO_SIDECAR_KEYS

    def test_model_dump_values(self) -> None:
        entry = VideoSidecarEntry(
            input_id="v1",
            model="wan22",
            mode="i2v",
            prompt="sunset",
            source_filename="sun.png",
            seed=99,
            generation_index=1,
        )
        d = entry.model_dump()
        assert d["input_id"] == "v1"
        assert d["model"] == "wan22"
        assert d["mode"] == "i2v"
        assert d["seed"] == 99
        assert d["generation_index"] == 1

    def test_flf2v_sidecar_same_shape(self) -> None:
        """flf2v uses the same VideoSidecarEntry model."""
        entry = VideoSidecarEntry(
            input_id="f1",
            model="wan22_flf",
            mode="flf2v",
            prompt="loop",
            source_filename="frame.png",
            seed=42,
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_VIDEO_SIDECAR_KEYS
