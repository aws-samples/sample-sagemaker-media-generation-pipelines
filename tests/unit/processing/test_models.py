# Feature: vrag-llm-container, Property 2: VragOutputEntry schema enforces required fields and rejects extras
"""
Unit and property-based tests for VragOutputEntry model validation.

**Validates: Requirements 1b.1, 1b.2**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from processing_job.common.models import VisualEntry, VragOutputEntry

pytestmark = pytest.mark.processing


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_str_st = st.text(min_size=1, max_size=100)

_valid_vrag_output_st = st.builds(
    VragOutputEntry,
    id=_str_st,
    prompt=_str_st,
    image=st.just(""),
    retrieval_query=_str_st,
    video_prompt=_str_st,
)

# Strategy for valid VragOutputEntry dicts
_valid_vrag_dict_st = st.fixed_dictionaries(
    {
        "id": _str_st,
        "prompt": _str_st,
        "retrieval_query": _str_st,
        "video_prompt": _str_st,
    }
)

# Strategy for dicts missing retrieval_query
_missing_retrieval_query_st = st.fixed_dictionaries(
    {
        "id": _str_st,
        "prompt": _str_st,
        "video_prompt": _str_st,
    }
)

# Strategy for dicts missing video_prompt
_missing_video_prompt_st = st.fixed_dictionaries(
    {
        "id": _str_st,
        "prompt": _str_st,
        "retrieval_query": _str_st,
    }
)

# Strategy for dicts with extra fields
_extra_field_st = st.fixed_dictionaries(
    {
        "id": _str_st,
        "prompt": _str_st,
        "retrieval_query": _str_st,
        "video_prompt": _str_st,
        "unexpected": _str_st,
    }
)

# Strategy for dicts with wrong types (int instead of str)
_wrong_type_st = st.fixed_dictionaries(
    {
        "id": st.integers(min_value=0, max_value=1000),
        "prompt": _str_st,
        "retrieval_query": _str_st,
        "video_prompt": _str_st,
    }
)


# ---------------------------------------------------------------------------
# Unit tests: VragOutputEntry validation
# ---------------------------------------------------------------------------


class TestVragOutputEntryValid:
    """VragOutputEntry accepts dicts with all required fields."""

    def test_valid_with_all_fields(self) -> None:
        raw = {
            "id": "tokyo-rain",
            "prompt": "A slow cinematic shot",
            "retrieval_query": "tokyo alley rain neon",
            "video_prompt": "Slow tracking shot through rainy alley",
        }
        entry = VragOutputEntry.model_validate(raw)
        assert entry.id == "tokyo-rain"
        assert entry.prompt == "A slow cinematic shot"
        assert entry.retrieval_query == "tokyo alley rain neon"
        assert entry.video_prompt == "Slow tracking shot through rainy alley"
        assert entry.image == ""

    def test_valid_with_explicit_image(self) -> None:
        raw = {
            "id": "city-01",
            "prompt": "City skyline",
            "retrieval_query": "city skyline sunset",
            "video_prompt": "Panning shot of city skyline at sunset",
            "image": "",
        }
        entry = VragOutputEntry.model_validate(raw)
        assert entry.image == ""

    def test_image_defaults_to_empty_string(self) -> None:
        entry = VragOutputEntry(id="x", prompt="p", retrieval_query="rq", video_prompt="vp")
        assert entry.image == ""


class TestVragOutputEntryRejection:
    """VragOutputEntry rejects invalid inputs."""

    def test_missing_retrieval_query_raises(self) -> None:
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate({"id": "x", "prompt": "p", "video_prompt": "vp"})

    def test_missing_video_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate({"id": "x", "prompt": "p", "retrieval_query": "rq"})

    def test_missing_both_new_fields_raises(self) -> None:
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate({"id": "x", "prompt": "p"})

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate(
                {
                    "id": "x",
                    "prompt": "p",
                    "retrieval_query": "rq",
                    "video_prompt": "vp",
                    "foo": "bar",
                }
            )

    def test_int_id_rejected_strict(self) -> None:
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate(
                {
                    "id": 42,
                    "prompt": "p",
                    "retrieval_query": "rq",
                    "video_prompt": "vp",
                }
            )

    def test_int_retrieval_query_rejected_strict(self) -> None:
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate(
                {
                    "id": "x",
                    "prompt": "p",
                    "retrieval_query": 123,
                    "video_prompt": "vp",
                }
            )


class TestVragOutputEntryInheritance:
    """VragOutputEntry inherits from VisualEntry (id, prompt, image defaults)."""

    def test_is_subclass_of_visual_entry(self) -> None:
        assert issubclass(VragOutputEntry, VisualEntry)

    def test_instance_is_visual_entry(self) -> None:
        entry = VragOutputEntry(id="x", prompt="p", retrieval_query="rq", video_prompt="vp")
        assert isinstance(entry, VisualEntry)

    def test_inherits_id_prompt_image(self) -> None:
        entry = VragOutputEntry(id="abc", prompt="hello", retrieval_query="rq", video_prompt="vp")
        assert entry.id == "abc"
        assert entry.prompt == "hello"
        assert entry.image == ""

    def test_model_dump_keys(self) -> None:
        entry = VragOutputEntry(id="x", prompt="p", retrieval_query="rq", video_prompt="vp")
        expected = {"id", "prompt", "image", "retrieval_query", "video_prompt"}
        assert set(entry.model_dump().keys()) == expected

    def test_round_trip_json(self) -> None:
        entry = VragOutputEntry(id="rt", prompt="round trip", retrieval_query="rq", video_prompt="vp")
        restored = VragOutputEntry.model_validate_json(entry.model_dump_json())
        assert restored == entry


# ---------------------------------------------------------------------------
# Property 2: VragOutputEntry schema enforces required fields and rejects extras
# ---------------------------------------------------------------------------


class TestVragOutputEntryProperty:
    """Property-based tests for VragOutputEntry schema enforcement.

    **Validates: Requirements 1b.1, 1b.2**
    """

    # Feature: vrag-llm-container, Property 2: VragOutputEntry schema enforces required fields and rejects extras

    @given(data=_valid_vrag_dict_st)
    @settings(max_examples=100)
    def test_valid_dicts_accepted(self, data: dict) -> None:
        """Any dict with id, prompt, retrieval_query, video_prompt as strings validates."""
        entry = VragOutputEntry.model_validate(data)
        assert entry.id == data["id"]
        assert entry.prompt == data["prompt"]
        assert entry.retrieval_query == data["retrieval_query"]
        assert entry.video_prompt == data["video_prompt"]
        assert entry.image == ""

    @given(data=_missing_retrieval_query_st)
    @settings(max_examples=100)
    def test_missing_retrieval_query_rejected(self, data: dict) -> None:
        """Any dict missing retrieval_query raises ValidationError."""
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate(data)

    @given(data=_missing_video_prompt_st)
    @settings(max_examples=100)
    def test_missing_video_prompt_rejected(self, data: dict) -> None:
        """Any dict missing video_prompt raises ValidationError."""
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate(data)

    @given(data=_extra_field_st)
    @settings(max_examples=100)
    def test_extra_fields_rejected(self, data: dict) -> None:
        """Any dict with extra fields raises ValidationError."""
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate(data)

    @given(data=_wrong_type_st)
    @settings(max_examples=100)
    def test_wrong_types_rejected(self, data: dict) -> None:
        """Any dict with non-string id raises ValidationError (strict mode)."""
        with pytest.raises(ValidationError):
            VragOutputEntry.model_validate(data)

    @given(instance=_valid_vrag_output_st)
    @settings(max_examples=100)
    def test_round_trip_preserves_fields(self, instance: VragOutputEntry) -> None:
        """Serialize then deserialize produces an equal instance."""
        restored = VragOutputEntry.model_validate_json(instance.model_dump_json())
        assert restored == instance
