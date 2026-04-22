"""Unit tests for t2i container model validation logic.

Tests VisualEntry validation for t2i and ImageSidecarEntry shape.
"""

import pytest
from pydantic import ValidationError

from processing_job.common.models import ImageSidecarEntry, VisualEntry

pytestmark = pytest.mark.model_validation


EXPECTED_IMAGE_SIDECAR_KEYS = {
    "input_id",
    "model",
    "mode",
    "prompt",
    "seed",
    "generation_index",
}


class TestT2iInputValidation:
    """t2i validates input shards via VisualEntry.model_validate()."""

    def test_valid_shard_validates(self) -> None:
        raw = {"id": "img01", "prompt": "a mountain landscape", "image": "mountain.png"}
        entry = VisualEntry.model_validate(raw)
        assert entry.id == "img01"
        assert entry.prompt == "a mountain landscape"

    def test_missing_image_defaults_to_empty(self) -> None:
        entry = VisualEntry.model_validate({"id": "x", "prompt": "p"})
        assert entry.image == ""

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            VisualEntry.model_validate({"id": "x"})  # missing prompt

    def test_invalid_entries_skipped(self) -> None:
        raw_inputs = [
            {"id": "ok", "prompt": "good", "image": "a.png"},
            {"bad": "data"},
            {"id": "ok2", "prompt": "fine", "image": "b.png"},
        ]
        validated = []
        for raw in raw_inputs:
            try:
                validated.append(VisualEntry.model_validate(raw))
            except ValidationError:
                continue
        assert len(validated) == 2


class TestImageSidecarEntryShape:
    """ImageSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_model_dump_keys(self) -> None:
        entry = ImageSidecarEntry(
            input_id="i1",
            model="z_image_turbo",
            mode="t2i",
            prompt="a cat",
            seed=0,
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_IMAGE_SIDECAR_KEYS

    def test_model_dump_values(self) -> None:
        entry = ImageSidecarEntry(
            input_id="i2",
            model="z_image_turbo",
            mode="t2i",
            prompt="sunset",
            seed=10,
            generation_index=1,
        )
        d = entry.model_dump()
        assert d["input_id"] == "i2"
        assert d["model"] == "z_image_turbo"
        assert d["seed"] == 10
        assert d["generation_index"] == 1

    def test_no_image_field_in_image_sidecar(self) -> None:
        """ImageSidecarEntry does NOT have an image field (unlike VideoSidecarEntry)."""
        assert "image" not in ImageSidecarEntry.model_fields
