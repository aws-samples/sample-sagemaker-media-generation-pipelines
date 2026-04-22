"""Unit tests for t2a container model validation logic.

Tests AudioEntry validation and AudioSidecarEntry shape.
"""

import pytest
from pydantic import ValidationError

from processing_job.common.models import AudioEntry, AudioSidecarEntry

pytestmark = pytest.mark.model_validation


EXPECTED_AUDIO_SIDECAR_KEYS = {
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


class TestT2aInputValidation:
    """t2a validates input shards via AudioEntry.model_validate()."""

    def test_valid_shard_validates(self) -> None:
        raw = {"id": "a1", "tags": "pop rock", "lyrics": "la la la"}
        entry = AudioEntry.model_validate(raw)
        assert entry.id == "a1"
        assert entry.tags == "pop rock"
        assert entry.lyrics == "la la la"

    def test_defaults_applied(self) -> None:
        raw = {"id": "a1", "tags": "jazz", "lyrics": "do re mi"}
        entry = AudioEntry.model_validate(raw)
        assert entry.bpm == 120
        assert entry.duration == 120
        assert entry.timesignature == "4"
        assert entry.language == "en"
        assert entry.keyscale == "C major"

    def test_custom_values_override_defaults(self) -> None:
        raw = {
            "id": "a2",
            "tags": "metal",
            "lyrics": "scream",
            "bpm": 180,
            "duration": 60,
            "timesignature": "6",
            "language": "de",
            "keyscale": "E minor",
        }
        entry = AudioEntry.model_validate(raw)
        assert entry.bpm == 180
        assert entry.duration == 60
        assert entry.keyscale == "E minor"

    def test_missing_tags_raises(self) -> None:
        with pytest.raises(ValidationError):
            AudioEntry.model_validate({"id": "a", "lyrics": "l"})

    def test_missing_lyrics_raises(self) -> None:
        with pytest.raises(ValidationError):
            AudioEntry.model_validate({"id": "a", "tags": "t"})

    def test_wrong_bpm_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            AudioEntry.model_validate({"id": "a", "tags": "t", "lyrics": "l", "bpm": "fast"})

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AudioEntry.model_validate({"id": "a", "tags": "t", "lyrics": "l", "genre": "pop"})

    def test_invalid_entries_skipped(self) -> None:
        """Simulate container loop: valid entries pass, invalid are skipped."""
        raw_inputs = [
            {"id": "ok1", "tags": "pop", "lyrics": "hey"},
            {"id": 123, "tags": "bad", "lyrics": "nope"},  # int id
            {"id": "ok2", "tags": "rock", "lyrics": "yeah"},
        ]
        validated = []
        for raw in raw_inputs:
            try:
                validated.append(AudioEntry.model_validate(raw))
            except ValidationError:
                continue
        assert len(validated) == 2
        assert validated[0].id == "ok1"
        assert validated[1].id == "ok2"


class TestAudioSidecarEntryShape:
    """AudioSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_model_dump_keys(self) -> None:
        entry = AudioSidecarEntry(
            input_id="a1",
            model="ace_step",
            mode="t2a",
            tags="pop",
            lyrics="la la",
            seed=31,
            bpm=120,
            duration=120,
            keyscale="C major",
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_AUDIO_SIDECAR_KEYS

    def test_model_dump_values(self) -> None:
        entry = AudioSidecarEntry(
            input_id="a2",
            model="ace_step",
            mode="t2a",
            tags="jazz",
            lyrics="scat",
            seed=41,
            bpm=90,
            duration=60,
            keyscale="D minor",
            generation_index=1,
        )
        d = entry.model_dump()
        assert d["input_id"] == "a2"
        assert d["model"] == "ace_step"
        assert d["mode"] == "t2a"
        assert d["tags"] == "jazz"
        assert d["lyrics"] == "scat"
        assert d["seed"] == 41
        assert d["bpm"] == 90
        assert d["duration"] == 60
        assert d["keyscale"] == "D minor"
        assert d["generation_index"] == 1

    def test_no_extra_keys(self) -> None:
        """model_dump() must not include keys beyond the expected set."""
        entry = AudioSidecarEntry(
            input_id="a1",
            model="ace_step",
            mode="t2a",
            tags="t",
            lyrics="l",
            seed=0,
            bpm=120,
            duration=120,
            keyscale="C major",
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_AUDIO_SIDECAR_KEYS
