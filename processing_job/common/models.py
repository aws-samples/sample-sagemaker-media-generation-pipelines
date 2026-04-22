"""
Pydantic v2 models for SageMaker Processing Job pipeline data.

DAG-flow models define the JSON shards that flow between pipeline steps.
Sidecar models define per-asset metadata written to DynamoDB.
VBenchMetrics captures VBench evaluation scores (standalone).

All models use strict=True and extra='forbid' to catch type coercion
bugs and unexpected fields at validation time.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# DAG-flow models
# ---------------------------------------------------------------------------


class BaseEntry(BaseModel):
    """Shared base for all DAG-flow entries."""

    model_config = ConfigDict(strict=True, extra="forbid")

    id: str


class VisualEntry(BaseEntry):
    """Agent/captioning output → t2v/i2v/flf2v input."""

    prompt: str
    image: str = ""


class VragOutputEntry(VisualEntry):
    """vrag_llm output → retrieval input.

    Extends VisualEntry with LLM-refined fields for downstream steps.
    retrieval uses retrieval_query for AOSS search.
    retrieval passes video_prompt through as the VisualEntry.prompt for i2v.
    """

    retrieval_query: str
    video_prompt: str


class AudioEntry(BaseEntry):
    """Agent output → t2a input."""

    tags: str
    lyrics: str
    bpm: int = 120
    duration: int = 120
    timesignature: str = "4"
    language: str = "en"
    keyscale: str = "C major"


# ---------------------------------------------------------------------------
# Sidecar models (DynamoDB metadata)
# ---------------------------------------------------------------------------


class BaseSidecarEntry(BaseModel):
    """Common fields shared by all sidecar metadata entries."""

    model_config = ConfigDict(strict=True, extra="forbid")

    input_id: str
    model: str
    mode: str
    generation_index: int


class VideoSidecarEntry(BaseSidecarEntry):
    """Per-video metadata written to _video_metadata.json."""

    prompt: str
    source_filename: str = ""
    seed: int


class ImageSidecarEntry(BaseSidecarEntry):
    """Per-image metadata written to _image_metadata.json."""

    prompt: str
    seed: int


class AudioSidecarEntry(BaseSidecarEntry):
    """Per-audio metadata written to _audio_metadata.json."""

    prompt: str = ""
    tags: str
    lyrics: str
    seed: int
    bpm: int
    duration: int
    keyscale: str


class CaptioningSidecarEntry(BaseSidecarEntry):
    """Per-caption metadata written to _caption_metadata.json."""

    prompt: str
    source_filename: str


# ---------------------------------------------------------------------------
# Standalone model
# ---------------------------------------------------------------------------


class VBenchMetrics(BaseModel):
    """VBench evaluation scores. Not part of the DAG flow."""

    model_config = ConfigDict(strict=True, extra="forbid")

    subject_consistency: float | None = None
    background_consistency: float | None = None
    motion_smoothness: float | None = None
    dynamic_degree: float | None = None
    aesthetic_quality: float | None = None
    imaging_quality: float | None = None
    temporal_flickering: float | None = None
    temporal_style: float | None = None
    overall_consistency: float | None = None
    human_action: float | None = None
