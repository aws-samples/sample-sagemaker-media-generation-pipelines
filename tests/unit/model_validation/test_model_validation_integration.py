"""Integration tests for end-to-end shard flow.

Verifies round-trip serialization across pipeline boundaries:
- agent → visual (VisualEntry shard)
- captioning → visual (VisualEntry shard)
- agent → t2a (AudioEntry shard)

Also verifies sidecar model_dump() keys match expected DynamoDB schemas.
"""

import os

import pytest

from processing_job.common.models import (
    AudioEntry,
    AudioSidecarEntry,
    CaptioningSidecarEntry,
    ImageSidecarEntry,
    VideoSidecarEntry,
    VisualEntry,
)

pytestmark = pytest.mark.model_validation


# Expected DynamoDB key sets per sidecar model
EXPECTED_VIDEO_SIDECAR_KEYS = {
    "input_id",
    "model",
    "mode",
    "prompt",
    "source_filename",
    "seed",
    "generation_index",
}
EXPECTED_IMAGE_SIDECAR_KEYS = {
    "input_id",
    "model",
    "mode",
    "prompt",
    "seed",
    "generation_index",
}
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
EXPECTED_CAPTIONING_SIDECAR_KEYS = {
    "input_id",
    "model",
    "mode",
    "prompt",
    "source_filename",
    "generation_index",
}


class TestAgentToVisualRoundTrip:
    """Agent serializes VisualEntry → t2v/i2v reads it back."""

    def test_write_and_read_back(self, tmp_path) -> None:
        """Simulate agent writing a shard, visual container reading it."""
        entry = VisualEntry(id="img001", prompt="a flying car", image="car.png")

        shard_path = os.path.join(tmp_path, "img001.json")
        with open(shard_path, "w") as f:
            f.write(entry.model_dump_json())

        with open(shard_path) as f:
            restored = VisualEntry.model_validate_json(f.read())

        assert restored == entry

    def test_fields_preserved(self, tmp_path) -> None:
        entry = VisualEntry(id="v42", prompt="sunset over ocean", image="sunset.jpg")

        shard_path = os.path.join(tmp_path, "v42.json")
        with open(shard_path, "w") as f:
            f.write(entry.model_dump_json())

        with open(shard_path) as f:
            restored = VisualEntry.model_validate_json(f.read())

        assert restored.id == "v42"
        assert restored.prompt == "sunset over ocean"
        assert restored.image == "sunset.jpg"


class TestCaptioningToVisualRoundTrip:
    """Captioning serializes VisualEntry → i2v/flf2v reads it back."""

    def test_write_and_read_back(self, tmp_path) -> None:
        """Simulate captioning writing a shard, visual container reading it."""
        entry = VisualEntry(
            id="frame_001",
            prompt="A golden retriever playing in a park",
            image="frame_001.png",
        )

        shard_path = os.path.join(tmp_path, "frame_001.json")
        with open(shard_path, "w") as f:
            f.write(entry.model_dump_json())

        with open(shard_path) as f:
            restored = VisualEntry.model_validate_json(f.read())

        assert restored == entry

    def test_caption_as_prompt(self, tmp_path) -> None:
        """Captioning sets prompt to the generated caption text."""
        caption = "A serene mountain landscape at dawn with mist"
        entry = VisualEntry(id="mountain", prompt=caption, image="mountain.jpg")

        shard_path = os.path.join(tmp_path, "mountain.json")
        with open(shard_path, "w") as f:
            f.write(entry.model_dump_json())

        with open(shard_path) as f:
            restored = VisualEntry.model_validate_json(f.read())

        assert restored.prompt == caption


class TestAgentToT2ARoundTrip:
    """Agent serializes AudioEntry → t2a reads it back."""

    def test_write_and_read_back_defaults(self, tmp_path) -> None:
        """AudioEntry with default fields round-trips correctly."""
        entry = AudioEntry(id="audio1", tags="pop rock", lyrics="la la la")

        shard_path = os.path.join(tmp_path, "audio1.json")
        with open(shard_path, "w") as f:
            f.write(entry.model_dump_json())

        with open(shard_path) as f:
            restored = AudioEntry.model_validate_json(f.read())

        assert restored == entry
        assert restored.bpm == 120
        assert restored.duration == 120

    def test_write_and_read_back_custom(self, tmp_path) -> None:
        """AudioEntry with all custom fields round-trips correctly."""
        entry = AudioEntry(
            id="audio2",
            tags="jazz",
            lyrics="do re mi",
            bpm=90,
            duration=60,
            timesignature="3",
            language="es",
            keyscale="D minor",
        )

        shard_path = os.path.join(tmp_path, "audio2.json")
        with open(shard_path, "w") as f:
            f.write(entry.model_dump_json())

        with open(shard_path) as f:
            restored = AudioEntry.model_validate_json(f.read())

        assert restored == entry
        assert restored.bpm == 90
        assert restored.keyscale == "D minor"


class TestVideoSidecarShape:
    """VideoSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_keys_match(self) -> None:
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

    def test_no_extra_keys(self) -> None:
        entry = VideoSidecarEntry(
            input_id="v2",
            model="wan22",
            mode="i2v",
            prompt="sunset",
            source_filename="sun.png",
            seed=99,
            generation_index=1,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_VIDEO_SIDECAR_KEYS


class TestImageSidecarShape:
    """ImageSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_keys_match(self) -> None:
        entry = ImageSidecarEntry(
            input_id="i1",
            model="flux",
            mode="t2i",
            prompt="a dog",
            seed=7,
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_IMAGE_SIDECAR_KEYS


class TestAudioSidecarShape:
    """AudioSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_keys_match(self) -> None:
        entry = AudioSidecarEntry(
            input_id="a1",
            model="stable_audio",
            mode="t2a",
            tags="pop",
            lyrics="hey",
            seed=55,
            bpm=120,
            duration=120,
            keyscale="C major",
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_AUDIO_SIDECAR_KEYS


class TestCaptioningSidecarShape:
    """CaptioningSidecarEntry.model_dump() keys match expected DynamoDB schema."""

    def test_keys_match(self) -> None:
        entry = CaptioningSidecarEntry(
            input_id="c1",
            model="qwen3_5_9b",
            mode="captioning",
            prompt="A beautiful sunset",
            source_filename="sunset.png",
            generation_index=0,
        )
        assert set(entry.model_dump().keys()) == EXPECTED_CAPTIONING_SIDECAR_KEYS
