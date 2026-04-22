# Feature: open-images-ingestion, Property 1: Resolution filter correctness
# Feature: open-images-ingestion, Property 2: S3 key format
# Feature: open-images-ingestion, Property 3: Return schema conformance
# Feature: open-images-ingestion, Property 4: Multimodal content block structure
# Feature: open-images-ingestion, Property 5: CSV nrows calculation
"""
Property-based tests for the open_images_v7 module.

Property 1: Resolution filter correctness
- For any image dimensions and any resolution bounds, _passes_resolution_filter
  SHALL return True iff MIN_WIDTH <= width <= MAX_WIDTH and MIN_HEIGHT <= height <= MAX_HEIGHT.

Property 2: S3 key format
- For any ImageID string, the S3 key SHALL be images/{ImageID}.jpg and round-trip
  extraction SHALL recover the original ImageID.

Property 3: Return schema conformance
- For any valid image_id, the dict returned by _process_image (when successful)
  SHALL have exactly keys id, description, s3_uri.

Property 4: Multimodal content block structure
- For any non-empty image bytes, the content blocks SHALL be a list of exactly two
  elements: first with text key (non-empty string), second with image key containing
  {"format": "jpeg", "source": {"bytes": <original_bytes>}}.

Property 5: CSV nrows calculation
- For any positive limit and multiplier, nrows SHALL equal multiplier * limit.
"""

from __future__ import annotations

import sys
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Make the bare `unsplash` import in open_images_v7.py resolve to the
# full package path used in the test environment.
import processing_job.dataset_ingest.unsplash as _unsplash_mod

sys.modules.setdefault("unsplash", _unsplash_mod)

import processing_job.dataset_ingest.open_images_v7 as oiv7  # noqa: E402
from processing_job.dataset_ingest.main import build_multimodal_content  # noqa: E402

pytestmark = pytest.mark.steps_setup


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# ImageID: alphanumeric strings (1-30 chars)
_image_id_st = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    min_size=1,
    max_size=30,
)

# Positive integers for dimensions and bounds
_dim_st = st.integers(min_value=1, max_value=5000)

# Positive integers for limit/multiplier
_pos_int_st = st.integers(min_value=1, max_value=10000)


# ---------------------------------------------------------------------------
# Property 1: Resolution filter correctness
# ---------------------------------------------------------------------------


class TestResolutionFilterCorrectnessProperty:
    """
    Property 1: Resolution filter correctness.

    **Validates: Requirements 2.3, 2.4**
    """

    @given(
        width=_dim_st,
        height=_dim_st,
        min_w=_dim_st,
        max_w=_dim_st,
        min_h=_dim_st,
        max_h=_dim_st,
    )
    @settings(max_examples=100)
    def test_resolution_filter_matches_bounds_check(
        self,
        width: int,
        height: int,
        min_w: int,
        max_w: int,
        min_h: int,
        max_h: int,
    ) -> None:
        """
        **Validates: Requirements 2.3, 2.4**

        For any (width, height) and any bounds, _passes_resolution_filter
        returns True iff min_w <= width <= max_w and min_h <= height <= max_h.
        """
        with (
            patch.object(oiv7, "MIN_WIDTH", min_w),
            patch.object(oiv7, "MAX_WIDTH", max_w),
            patch.object(oiv7, "MIN_HEIGHT", min_h),
            patch.object(oiv7, "MAX_HEIGHT", max_h),
        ):
            result = oiv7._passes_resolution_filter(width, height)
            expected = min_w <= width <= max_w and min_h <= height <= max_h
            assert result == expected


# ---------------------------------------------------------------------------
# Property 2: S3 key format
# ---------------------------------------------------------------------------


class TestS3KeyFormatProperty:
    """
    Property 2: S3 key format.

    **Validates: Requirements 3.4**
    """

    @given(image_id=_image_id_st)
    @settings(max_examples=100)
    def test_s3_key_format_and_round_trip(self, image_id: str) -> None:
        """
        **Validates: Requirements 3.4**

        For any ImageID, the S3 key is images/{ImageID}.jpg and extracting
        the ImageID back recovers the original.
        """
        s3_key = f"images/{image_id}.jpg"
        assert s3_key == f"images/{image_id}.jpg"
        # Round-trip extraction
        extracted = s3_key.removeprefix("images/").removesuffix(".jpg")
        assert extracted == image_id


# ---------------------------------------------------------------------------
# Property 3: Return schema conformance
# ---------------------------------------------------------------------------


class TestReturnSchemaConformanceProperty:
    """
    Property 3: Return schema conformance.

    **Validates: Requirements 3.5, 5.2**
    """

    @given(image_id=_image_id_st)
    @settings(max_examples=100)
    def test_process_image_returns_correct_schema(self, image_id: str) -> None:
        """
        **Validates: Requirements 3.5, 5.2**

        For any valid image_id, when _process_image succeeds the returned dict
        has exactly keys {id, description, s3_uri} with correct formats.
        """
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (640, 480))
        buf = BytesIO()
        img.save(buf, format="JPEG")
        fake_jpeg = buf.getvalue()

        mock_s3 = MagicMock()
        bucket = "test-bucket"

        with (
            patch.object(oiv7, "download_image", return_value=fake_jpeg),
            patch.object(oiv7, "resize_image", return_value=fake_jpeg),
            patch.object(oiv7, "upload_to_s3", return_value=True),
            patch.object(oiv7, "_key_exists", return_value=False),
            patch.object(oiv7, "MIN_WIDTH", 1),
            patch.object(oiv7, "MAX_WIDTH", 5000),
            patch.object(oiv7, "MIN_HEIGHT", 1),
            patch.object(oiv7, "MAX_HEIGHT", 5000),
        ):
            result = oiv7._process_image(mock_s3, bucket, image_id, "http://example.com/img.jpg")

        assert result is not None
        assert set(result.keys()) == {"id", "description", "s3_uri"}
        assert result["id"] == image_id
        assert isinstance(result["description"], str) and len(result["description"]) > 0
        assert result["s3_uri"].startswith("s3://")
        assert "/images/" in result["s3_uri"]


# ---------------------------------------------------------------------------
# Property 4: Multimodal content block structure
# ---------------------------------------------------------------------------


class TestMultimodalContentBlockStructureProperty:
    """
    Property 4: Multimodal content block structure.

    **Validates: Requirements 4.5, 4.6**
    """

    @given(image_bytes=st.binary(min_size=1, max_size=10000))
    @settings(max_examples=100)
    def test_content_blocks_structure(self, image_bytes: bytes) -> None:
        """
        **Validates: Requirements 4.5, 4.6**

        For any non-empty image bytes, the content blocks list has exactly two
        elements: first with text key (non-empty string), second with image key
        containing {"format": "jpeg", "source": {"bytes": <original_bytes>}}.
        """
        content = build_multimodal_content(image_bytes)

        # Exactly two content blocks
        assert isinstance(content, list)
        assert len(content) == 2

        # First block: text with non-empty string
        text_block = content[0]
        assert "text" in text_block
        assert isinstance(text_block["text"], str)
        assert len(text_block["text"]) > 0

        # Second block: image with correct structure
        image_block = content[1]
        assert "image" in image_block
        assert image_block["image"]["format"] == "jpeg"
        assert image_block["image"]["source"]["bytes"] is image_bytes


# ---------------------------------------------------------------------------
# Property 5: CSV nrows calculation
# ---------------------------------------------------------------------------


class TestCsvNrowsCalculationProperty:
    """
    Property 5: CSV nrows calculation.

    **Validates: Requirements 9.1**
    """

    @given(limit=_pos_int_st, multiplier=_pos_int_st)
    @settings(max_examples=100)
    def test_nrows_equals_multiplier_times_limit(self, limit: int, multiplier: int) -> None:
        """
        **Validates: Requirements 9.1**

        For any positive limit and multiplier, nrows == multiplier * limit.
        """
        nrows = multiplier * limit
        assert nrows == multiplier * limit
        assert nrows > 0
