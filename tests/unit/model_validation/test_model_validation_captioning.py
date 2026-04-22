"""Unit tests for captioning container model validation logic.

Tests VisualEntry serialization for output shards and CaptioningSidecarEntry shape.
"""

import json

import pytest
from pydantic import ValidationError

from processing_job.common.models import CaptioningSidecarEntry, VisualEntry

pytestmark = pytest.mark.model_validation


EXPECTED_CAPTIONING_SIDECAR_KEYS = {
    "input_id",
    "model",
    "mode",
    "prompt",
    "source_filename",
    "generation_index",
}


class TestCaptioningVisualEntrySerialization:
    """Captioning constructs VisualEntry for downstream shards and serializes via model_dump_json()."""

    def test_visual_entry_constructed_correctly(self) -> None:
        entry = VisualEntry(id="sunset", prompt="A golden sunset over the ocean", image="sunset.png")
        assert entry.id == "sunset"
        assert entry.prompt == "A golden sunset over the ocean"
        assert entry.image == "sunset.png"

    def test_visual_entry_serializes_to_valid_json(self) -> None:
        entry = VisualEntry(id="cat", prompt="A fluffy cat", image="cat.jpg")
        json_str = entry.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "cat"
        assert parsed["prompt"] == "A fluffy cat"
        assert parsed["image"] == "cat.jpg"

    def test_visual_entry_round_trip(self) -> None:
        """Shard written by captioning is readable by downstream visual container."""
        entry = VisualEntry(id="img01", prompt="a mountain", image="mountain.png")
        json_str = entry.model_dump_json()
        restored = VisualEntry.model_validate_json(json_str)
        assert restored == entry

    def test_visual_entry_only_has_expected_keys(self) -> None:
        """Captioning output shard should only contain id, prompt, image — no extras."""
        entry = VisualEntry(id="x", prompt="p", image="i.png")
        parsed = json.loads(entry.model_dump_json())
        assert set(parsed.keys()) == {"id", "prompt", "image"}

    def test_empty_caption_accepted(self) -> None:
        """If caption generation fails, an empty string is still valid."""
        entry = VisualEntry(id="fail", prompt="", image="fail.png")
        assert entry.prompt == ""


class TestCaptioningSidecarEntryShape:
    """CaptioningSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_model_dump_keys(self) -> None:
        entry = CaptioningSidecarEntry(
            input_id="img01",
            model="qwen3_5_9b",
            mode="captioning",
            prompt="A beautiful sunset",
            source_filename="sunset.png",
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_CAPTIONING_SIDECAR_KEYS

    def test_model_dump_values(self) -> None:
        entry = CaptioningSidecarEntry(
            input_id="img02",
            model="qwen3_5_9b",
            mode="captioning",
            prompt="A cat sleeping",
            source_filename="cat.jpg",
            generation_index=0,
        )
        d = entry.model_dump()
        assert d["input_id"] == "img02"
        assert d["model"] == "qwen3_5_9b"
        assert d["mode"] == "captioning"
        assert d["prompt"] == "A cat sleeping"
        assert d["source_filename"] == "cat.jpg"
        assert d["generation_index"] == 0

    def test_no_extra_keys(self) -> None:
        entry = CaptioningSidecarEntry(
            input_id="x",
            model="m",
            mode="captioning",
            prompt="c",
            source_filename="f.png",
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_CAPTIONING_SIDECAR_KEYS

    def test_wrong_type_raises(self) -> None:
        """generation_index must be int, not str (strict mode)."""
        with pytest.raises(ValidationError):
            CaptioningSidecarEntry(
                input_id="x",
                model="m",
                mode="captioning",
                prompt="c",
                source_filename="f.png",
                generation_index="bad",  # type: ignore[arg-type]
            )
