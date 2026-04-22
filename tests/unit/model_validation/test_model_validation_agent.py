"""Unit tests for agent container model validation logic.

Tests that valid shards are validated and serialized correctly,
and that invalid entries are skipped with a logged ValidationError.
"""

import json

import pytest
from pydantic import ValidationError

from processing_job.common.models import AudioEntry, VisualEntry

pytestmark = pytest.mark.model_validation


class TestAgentVisualValidation:
    """Agent validates visual entries from inputs.json."""

    def test_valid_visual_entry_validates(self) -> None:
        raw = {"id": "abc123", "prompt": "a sunset over the ocean", "image": "sunset.png"}
        entry = VisualEntry.model_validate(raw)
        assert entry.id == "abc123"
        assert entry.prompt == "a sunset over the ocean"
        assert entry.image == "sunset.png"

    def test_valid_visual_entry_serializes_to_json(self) -> None:
        raw = {"id": "abc123", "prompt": "a cat", "image": "cat.png"}
        entry = VisualEntry.model_validate(raw)
        json_str = entry.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed == raw

    def test_visual_round_trip(self) -> None:
        raw = {"id": "v1", "prompt": "hello world", "image": "img.jpg"}
        entry = VisualEntry.model_validate(raw)
        restored = VisualEntry.model_validate_json(entry.model_dump_json())
        assert restored == entry


class TestAgentAudioValidation:
    """Agent validates audio entries from inputs.json."""

    def test_valid_audio_entry_validates(self) -> None:
        raw = {"id": "a1", "tags": "pop rock", "lyrics": "la la la"}
        entry = AudioEntry.model_validate(raw)
        assert entry.id == "a1"
        assert entry.tags == "pop rock"
        assert entry.lyrics == "la la la"
        assert entry.bpm == 120  # default

    def test_audio_entry_with_all_fields(self) -> None:
        raw = {
            "id": "a2",
            "tags": "jazz",
            "lyrics": "do re mi",
            "bpm": 90,
            "duration": 60,
            "timesignature": "3",
            "language": "es",
            "keyscale": "D minor",
        }
        entry = AudioEntry.model_validate(raw)
        assert entry.bpm == 90
        assert entry.keyscale == "D minor"

    def test_audio_round_trip(self) -> None:
        raw = {"id": "a1", "tags": "pop", "lyrics": "hey"}
        entry = AudioEntry.model_validate(raw)
        restored = AudioEntry.model_validate_json(entry.model_dump_json())
        assert restored == entry


class TestAgentInvalidEntries:
    """Invalid entries raise ValidationError and would be skipped."""

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            VisualEntry.model_validate({"prompt": "p", "image": "i"})

    def test_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            VisualEntry.model_validate({"id": "x", "prompt": "p", "image": "i", "extra": "bad"})

    def test_wrong_type_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            VisualEntry.model_validate({"id": 123, "prompt": "p", "image": "i"})

    def test_audio_wrong_bpm_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            AudioEntry.model_validate({"id": "a", "tags": "t", "lyrics": "l", "bpm": "fast"})


class TestAgentMixedEntryDiscrimination:
    """Agent discriminates visual vs audio entries by 'tags' key presence."""

    def test_tags_present_means_audio(self) -> None:
        raw = {"id": "a1", "tags": "rock", "lyrics": "yeah"}
        assert "tags" in raw
        entry = AudioEntry.model_validate(raw)
        assert isinstance(entry, AudioEntry)

    def test_tags_absent_means_visual(self) -> None:
        raw = {"id": "v1", "prompt": "sunset", "image": "img.png"}
        assert "tags" not in raw
        entry = VisualEntry.model_validate(raw)
        assert isinstance(entry, VisualEntry)

    def test_mixed_list_filters_invalid(self) -> None:
        """Simulate agent loop: valid entries pass, invalid are skipped."""
        entries = [
            {"id": "v1", "prompt": "cat", "image": "cat.png"},
            {"id": 999, "prompt": "bad", "image": "x"},  # invalid: int id
            {"id": "a1", "tags": "pop", "lyrics": "la"},
            {"prompt": "no id"},  # invalid: missing id
        ]
        valid = []
        for entry in entries:
            try:
                if "tags" in entry:
                    valid.append(AudioEntry.model_validate(entry))
                else:
                    valid.append(VisualEntry.model_validate(entry))
            except ValidationError:
                continue
        assert len(valid) == 2
        assert isinstance(valid[0], VisualEntry)
        assert isinstance(valid[1], AudioEntry)
