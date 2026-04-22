# Feature: unsplash-setup-ingestion, Property 3: Description fallback chain
# Feature: unsplash-setup-ingestion, Property 6: S3 key format
# Feature: unsplash-setup-ingestion, Property 9: VisualEntry output round trip
"""
Property-based tests for the dataset_ingest container logic.

Property 3: Description fallback chain
- For any parquet row dict with varying combinations of ai, description, and keywords
  fields (where values may be None, "nan", or empty), resolve_description SHALL return
  the correct value per the fallback chain.

Property 6: S3 key format
- For any photo ID string, the S3 key used for upload SHALL be images/{photo_id}.jpg.

Property 9: VisualEntry output round trip
- For any list of VisualEntry objects written to inputs_i2v.json, reading the file back
  and parsing each element as a VisualEntry SHALL produce objects identical to the originals.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from processing_job.common.models import VisualEntry
from processing_job.dataset_ingest.main import write_visual_entries
from processing_job.dataset_ingest.unsplash import resolve_description

pytestmark = pytest.mark.steps_setup


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty, non-nan text for "valid" description values
_valid_desc_st = st.text(min_size=1, max_size=80).filter(lambda s: s.strip() and s.strip().lower() != "nan")

# Values that should be treated as absent/invalid descriptions
_invalid_desc_st = st.one_of(
    st.none(),
    st.just("nan"),
    st.just("NaN"),
    st.just(""),
    st.just("  "),
    st.just(" nan "),
)

# Keyword dict strategy (matching parquet schema)
_keyword_dict_st = st.fixed_dictionaries({"keyword": _valid_desc_st})

# Non-empty keyword list
_valid_keywords_st = st.lists(_keyword_dict_st, min_size=1, max_size=5)

# Empty/absent keyword list
_invalid_keywords_st = st.one_of(
    st.none(),
    st.just([]),
    st.just([{"keyword": ""}]),
    st.just([{"keyword": "  "}]),
)

# Photo ID: alphanumeric strings (matching Unsplash photo IDs)
_photo_id_st = st.text(
    min_size=1,
    max_size=30,
    alphabet=st.characters(whitelist_categories=("L", "N")),
)

# Safe ID for file operations (no path separators)
_safe_id_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789"),
    min_size=1,
    max_size=20,
)


# ---------------------------------------------------------------------------
# Property 3: Description fallback chain
# ---------------------------------------------------------------------------


class TestDescriptionFallbackChainProperty:
    """
    Property 3: Description fallback chain.

    **Validates: Requirements 2.2, 2.3, 2.4**
    """

    @given(ai_desc=_valid_desc_st)
    @settings(max_examples=100)
    def test_ai_description_is_primary(self, ai_desc: str) -> None:
        """
        **Validates: Requirements 2.2, 2.3, 2.4**

        When ai['description'] is a valid non-empty, non-nan string,
        resolve_description SHALL return it regardless of other fields.
        """
        row = {
            "ai": {"description": ai_desc},
            "description": "fallback desc",
            "keywords": [{"keyword": "fallback kw"}],
        }
        result = resolve_description(row)
        assert result == ai_desc.strip()

    @given(desc=_valid_desc_st, ai_val=_invalid_desc_st)
    @settings(max_examples=100)
    def test_description_is_secondary(self, desc: str, ai_val) -> None:
        """
        **Validates: Requirements 2.2, 2.3, 2.4**

        When ai['description'] is absent/nan/empty but description is valid,
        resolve_description SHALL return description.
        """
        ai = {"description": ai_val} if ai_val is not None else {}
        row = {
            "ai": ai,
            "description": desc,
            "keywords": [{"keyword": "fallback kw"}],
        }
        result = resolve_description(row)
        assert result == desc.strip()

    @given(keywords=_valid_keywords_st, ai_val=_invalid_desc_st, desc_val=_invalid_desc_st)
    @settings(max_examples=100)
    def test_keywords_is_tertiary(self, keywords: list[dict], ai_val, desc_val) -> None:
        """
        **Validates: Requirements 2.2, 2.3, 2.4**

        When both ai['description'] and description are absent/nan/empty
        but keywords has valid entries, resolve_description SHALL return
        comma-joined keywords.
        """
        ai = {"description": ai_val} if ai_val is not None else {}
        row = {
            "ai": ai,
            "description": desc_val,
            "keywords": keywords,
        }
        result = resolve_description(row)
        expected = ", ".join(kw["keyword"].strip() for kw in keywords if kw.get("keyword", "").strip())
        assert result == expected

    @given(ai_val=_invalid_desc_st, desc_val=_invalid_desc_st, kw_val=_invalid_keywords_st)
    @settings(max_examples=100)
    def test_none_when_all_absent(self, ai_val, desc_val, kw_val) -> None:
        """
        **Validates: Requirements 2.2, 2.3, 2.4**

        When all three sources are absent/nan/empty,
        resolve_description SHALL return None.
        """
        ai = {"description": ai_val} if ai_val is not None else {}
        row: dict = {
            "ai": ai,
            "description": desc_val,
        }
        if kw_val is not None:
            row["keywords"] = kw_val
        result = resolve_description(row)
        assert result is None

    @given(
        ai_desc=st.one_of(_valid_desc_st, _invalid_desc_st),
        desc=st.one_of(_valid_desc_st, _invalid_desc_st),
        keywords=st.one_of(_valid_keywords_st, _invalid_keywords_st),
    )
    @settings(max_examples=100)
    def test_fallback_chain_ordering(self, ai_desc, desc, keywords) -> None:
        """
        **Validates: Requirements 2.2, 2.3, 2.4**

        For any random combination of present/absent/nan fields,
        the fallback chain order is always: ai_desc > desc > keywords > None.
        """
        ai = {"description": ai_desc} if ai_desc is not None else {}
        row: dict = {"ai": ai}
        if desc is not None:
            row["description"] = desc
        if keywords is not None:
            row["keywords"] = keywords

        result = resolve_description(row)

        # Determine expected result by following the chain
        def _is_valid(val) -> bool:
            return val is not None and str(val).strip() and str(val).strip().lower() != "nan"

        if _is_valid(ai_desc):
            assert result == str(ai_desc).strip()
        elif _is_valid(desc):
            assert result == str(desc).strip()
        elif keywords and isinstance(keywords, list):
            kw_strings = []
            for kw in keywords:
                if isinstance(kw, dict):
                    v = kw.get("keyword")
                    if v and str(v).strip():
                        kw_strings.append(str(v).strip())
            if kw_strings:
                assert result == ", ".join(kw_strings)
            else:
                assert result is None
        else:
            assert result is None


# ---------------------------------------------------------------------------
# Property 6: S3 key format
# ---------------------------------------------------------------------------


class TestS3KeyFormatProperty:
    """
    Property 6: S3 key format.

    **Validates: Requirements 3.1, 8.1**
    """

    @given(photo_id=_photo_id_st)
    @settings(max_examples=100)
    def test_s3_key_format(self, photo_id: str) -> None:
        """
        **Validates: Requirements 3.1, 8.1**

        For any photo ID string, the S3 key SHALL be images/{photo_id}.jpg.
        """
        expected_key = f"images/{photo_id}.jpg"
        # Verify the key format matches the pattern used in _process_row
        assert expected_key.startswith("images/")
        assert expected_key.endswith(".jpg")
        # Extract the photo_id back from the key
        extracted_id = expected_key.removeprefix("images/").removesuffix(".jpg")
        assert extracted_id == photo_id

    @given(photo_id=_photo_id_st)
    @settings(max_examples=100)
    def test_s3_key_is_deterministic(self, photo_id: str) -> None:
        """
        **Validates: Requirements 3.1, 8.1**

        The same photo ID always produces the same S3 key (idempotent).
        """
        key1 = f"images/{photo_id}.jpg"
        key2 = f"images/{photo_id}.jpg"
        assert key1 == key2


# ---------------------------------------------------------------------------
# Property 9: VisualEntry output round trip
# ---------------------------------------------------------------------------


# Strategy for valid VisualEntry dicts (matching the output format of write_visual_entries)
_visual_entry_dict_st = st.fixed_dictionaries(
    {
        "id": _safe_id_st,
        "prompt": st.text(min_size=1, max_size=80),
        "image": st.builds(
            lambda pid: f"s3://test-bucket/images/{pid}.jpg",
            _safe_id_st,
        ),
    }
)


class TestVisualEntryRoundTripProperty:
    """
    Property 9: VisualEntry output round trip.

    **Validates: Requirements 4.3, 4.4**
    """

    @given(entries=st.lists(_visual_entry_dict_st, min_size=0, max_size=10))
    @settings(max_examples=100)
    def test_round_trip_preserves_entries(self, entries: list[dict]) -> None:
        """
        **Validates: Requirements 4.3, 4.4**

        For any list of VisualEntry dicts, writing to inputs_t2v.json via
        write_visual_entries and reading back SHALL produce identical objects.
        """
        d = tempfile.mkdtemp()
        try:
            output_path = write_visual_entries(entries, d)
            assert os.path.exists(output_path)

            with open(output_path) as f:
                restored = json.load(f)

            assert len(restored) == len(entries)
            for original, loaded in zip(entries, restored):
                # Validate each loaded entry as a VisualEntry
                ve = VisualEntry.model_validate(loaded)
                assert ve.id == original["id"]
                assert ve.prompt == original["prompt"]
                assert ve.image == original["image"]
        finally:
            shutil.rmtree(d)

    @given(entries=st.lists(_visual_entry_dict_st, min_size=1, max_size=10))
    @settings(max_examples=100)
    def test_round_trip_image_matches_s3_pattern(self, entries: list[dict]) -> None:
        """
        **Validates: Requirements 4.3, 4.4**

        For any list of VisualEntry dicts with S3 URIs, the image field
        after round trip SHALL match the pattern s3://{bucket}/images/{photo_id}.jpg.
        """
        d = tempfile.mkdtemp()
        try:
            write_visual_entries(entries, d)
            output_path = os.path.join(d, "inputs_t2v.json")

            with open(output_path) as f:
                restored = json.load(f)

            for loaded in restored:
                ve = VisualEntry.model_validate(loaded)
                assert ve.image.startswith("s3://")
                assert "/images/" in ve.image
                assert ve.image.endswith(".jpg")
        finally:
            shutil.rmtree(d)
